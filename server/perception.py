"""Multi-provider perception: several detectors, one normalised detection stream.

The product has to work on a road and in a corridor. No single set of weights we
have does both: COCO knows cars and nothing about stairs or doors; the AV
fine-tune knows stairs and doors and has never seen a car. Choosing one means
choosing which user to fail, so both run and their output is fused here.

Three things live in this module and nowhere else, so that adding a provider or
changing who is trusted for a class is a one-file edit:

  CANONICAL   provider-specific class names -> one vocabulary the WorldModel
              speaks. COCO 'dining table' and AV 'table' are both TABLE.
  AUTHORITY   which provider is believed for a canonical class, in order. A
              class absent from the table accepts any provider that emits it.
  fuse()      one physical object seen by two providers becomes one detection.

Downstream code (priority, spatial, main) consumes the same dict shape it always
did -- label/raw/box/conf/kind -- with the normalised fields added alongside, so
nothing else had to change when this layer went in.
"""
import os
import time

from . import config
from .classes_av import AV_CLASSES, AV_HAZARDS, AV_SPOKEN

# ---------------------------------------------------------------- vocabulary

# Aliases only where the two vocabularies disagree on wording. Everything else
# canonicalises mechanically (upper case, spaces to underscores), so a class
# added to either model needs an entry here only if its name differs.
_ALIASES = {
    "dining table": "TABLE",
    "potted plant": "PLANT",
    "fire hydrant": "FIRE_HYDRANT",
    "stop sign": "STOP_SIGN",
    "traffic light": "TRAFFIC_LIGHT",
    "cell phone": "PHONE",
}


def canonical(label: str) -> str:
    """One vocabulary for the WorldModel. Never provider-specific class ids."""
    return _ALIASES.get(label.lower(), label.upper().replace(" ", "_").replace("-", "_"))


# Which provider is trusted for a class, best first. Both listed = fuse both.
# Derived from what each model was actually trained on, not from a guess about
# which model is "better": the AV fine-tune has never seen a car, and COCO has
# never seen a stairs_down.
AUTHORITY = {
    "PERSON": ("assistive", "general"),   # both were trained on it; fuse
    "CHAIR": ("assistive", "general"),
    "TABLE": ("assistive", "general"),
    "DOOR": ("assistive",),               # COCO 'door' is rare and unreliable
    "STAIRS_UP": ("assistive",),          # not a COCO class at all
    "STAIRS_DOWN": ("assistive",),
    "DUSTBIN": ("assistive",),
    "CAR": ("general",),                  # the AV models have never seen one
    "BUS": ("general",),
    "TRUCK": ("general",),
    "BICYCLE": ("general",),
    "MOTORCYCLE": ("general",),
    "TRAIN": ("general",),
    "TRAFFIC_LIGHT": ("general",),
    "STOP_SIGN": ("general",),
    "FIRE_HYDRANT": ("general",),
    "DOG": ("general",),
}

# How much two boxes must overlap to be called the same physical object.
# Deliberately lower than the within-model NMS threshold: two different models
# rarely agree on an extent as closely as one model agrees with itself.
FUSE_IOU = 0.45


def _iou(a, b):
    inter = (max(0, min(a[2], b[2]) - max(a[0], b[0]))
             * max(0, min(a[3], b[3]) - max(a[1], b[1])))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def fuse(detections):
    """Collapse detections of one physical object from several providers into one.

    Two detections are the same object when they share a canonical class and
    their boxes overlap by FUSE_IOU. The survivor is the one from the earliest
    provider in that class's AUTHORITY order -- not simply the most confident,
    because confidence is not comparable between models trained on different
    data. Its confidence is raised to the best any provider gave it, and the
    others are recorded in `agreed_by` so a reviewer can see who saw what.
    """
    kept = []
    # Highest confidence first so a cluster forms around the strongest sighting.
    for det in sorted(detections, key=lambda d: -d["conf"]):
        rank = AUTHORITY.get(det["canonical"], ())
        for other in kept:
            if other["canonical"] != det["canonical"]:
                continue
            if _iou(other["box"], det["box"]) < FUSE_IOU:
                continue
            # same object: keep whichever provider this class trusts more
            here = rank.index(det["provider_id"]) if det["provider_id"] in rank else len(rank)
            there = rank.index(other["provider_id"]) if other["provider_id"] in rank else len(rank)
            if here < there:
                det["conf"] = max(det["conf"], other["conf"])
                det["agreed_by"] = sorted(set(other.get("agreed_by", [])) |
                                          {other["provider_id"], det["provider_id"]})
                kept[kept.index(other)] = det
            else:
                other["conf"] = max(other["conf"], det["conf"])
                other["agreed_by"] = sorted(set(other.get("agreed_by", [])) |
                                            {other["provider_id"], det["provider_id"]})
            break
        else:
            kept.append(det)
    return kept


def permitted(det):
    """Is this provider allowed to speak for this class?"""
    rank = AUTHORITY.get(det["canonical"])
    return rank is None or det["provider_id"] in rank


# ---------------------------------------------------------------- providers

class Provider:
    """One set of weights, loaded lazily and identified by what it predicts."""

    def __init__(self, provider_id, path, every_n=1):
        self.id, self.path, self.every_n = provider_id, path, max(1, int(every_n))
        self.model = None
        self.schema = None
        self.error = None
        self.last_ms = None

    def load(self):
        if self.model is None and self.error is None:
            try:
                from ultralytics import YOLO
                from .detector import detect_schema
                model = YOLO(self.path)
                self.schema = detect_schema(model.names)   # validate before publishing
                self.model = model
            except Exception as e:                          # a bad provider must not kill the rest
                self.error = str(e)
        return self.model

    def predict(self, frame_bgr):
        model = self.load()
        if model is None:
            return []
        started = time.perf_counter()
        res = model.predict(frame_bgr, conf=config.OBJ_CONF, imgsz=config.IMAGE_SIZE,
                            iou=config.NMS_THRESHOLD, device=config.DEVICE, verbose=False)[0]
        self.last_ms = round((time.perf_counter() - started) * 1000, 1)
        if res.boxes is None:
            return []
        av = self.schema != "coco"
        out = []
        for xyxy, conf, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                   res.boxes.conf.cpu().numpy(),
                                   res.boxes.cls.cpu().numpy()):
            label = res.names[int(cls)]
            if not av and label not in config.OBJECT_CLASSES:
                continue
            if float(conf) < config.CLASS_CONF.get(label, config.OBJ_CONF):
                continue
            out.append({
                "provider_id": self.id,
                "model_id": os.path.basename(self.path),
                "class_id": int(cls),
                "canonical": canonical(label),
                "label": AV_SPOKEN.get(label, label) if av else label,
                "raw": label,
                "conf": float(conf),
                "box": tuple(int(v) for v in xyxy),
                "timestamp": time.time(),
                "kind": "object",
                "hazard": (label in AV_HAZARDS) if av else (label in config.HAZARD_CLASSES),
            })
        return out


_providers = None
_frame_index = 0
_cache = {}


def providers():
    """The configured providers, built once. A missing weights file is skipped."""
    global _providers
    if _providers is None:
        _providers = [Provider(p["id"], p["path"], p.get("every_n", 1))
                      for p in config.PERCEPTION_PROVIDERS
                      if os.path.exists(p["path"]) or not p["path"].startswith("models/")]
    return _providers


def reset():
    """Drop loaded weights and cadence state. For tests and config reloads."""
    global _providers, _frame_index, _cache
    _providers, _frame_index, _cache = None, 0, {}


def status():
    return [{"id": p.id, "model": os.path.basename(p.path), "schema": p.schema,
             "every_n": p.every_n, "last_ms": p.last_ms, "error": p.error}
            for p in providers()]


def observe(frame_bgr):
    """Run the due providers over one frame and return one fused detection list.

    A provider with every_n > 1 is not run on every frame; its previous
    detections are reused in between. That is the FrameScheduler in its simplest
    honest form -- it trades freshness for CPU, and the cadence is configuration
    rather than something hidden in this function.
    """
    global _frame_index
    _frame_index += 1
    merged = []
    for p in providers():
        if (_frame_index - 1) % p.every_n == 0:
            _cache[p.id] = p.predict(frame_bgr)
        merged.extend(_cache.get(p.id, []))
    return fuse([d for d in merged if permitted(d)])


def warmup():
    """Load every provider. Raises only if none of them came up.

    One failed provider is survivable -- the others still see -- but zero means
    the system is blind, and warmup must fail loudly rather than report ready.
    """
    live = [p for p in providers() if p.load() is not None]
    if not live:
        broken = "; ".join(f"{p.id}: {p.error or 'missing weights'}" for p in providers())
        raise RuntimeError("no detection provider loaded (" + (broken or "none configured") + ")")
    return live
