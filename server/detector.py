"""Object + obstacle detection using YOLOv8.

Output items: {label, box, conf, kind='object'|'hazard'}
A detection is a hazard if its class is in the active schema's hazard set AND
it is close (proximity computed later); the raw split here is by class only —
the priority engine re-ranks by proximity.
"""
import os

from . import config
from .classes_av import AV_CLASSES, AV_HAZARDS, AV_SPOKEN

_model = None

# Which class vocabulary a loaded model speaks. Everything downstream —
# hazard set, spoken names, class filtering — branches on this.
# Derived from the vocabulary, never written out as a literal: a hand-typed
# "av7" survived the schema shrinking to six classes and would have told
# /health, the web app and every report a class count the weights no longer
# have. The id cannot outlive the list it names.
SCHEMA_AV = f"av{len(AV_CLASSES)}"
SCHEMA_COCO = "coco"


def detect_schema(names) -> str:
    """Identify a model by its CLASS NAMES, never by its filename.

    This used to be inferred from the weights path: a file present at
    OBJECT_MODEL_CUSTOM was assumed to be the AV-6 fine-tune. A COCO-class
    model dropped there therefore switched hazard lookup to AV_HAZARDS, which
    contains no vehicles — 'car', 'bus', 'truck', 'bicycle' and 'motorcycle'
    all silently stopped being hazards, with no error and no log line.

    Schemas are now derived from what the model actually predicts, so the two
    vocabularies can never be crossed. An unrecognised schema is refused
    outright rather than guessed at: for a blind user, a detector whose hazard
    set we cannot establish is worse than no detector at all.
    """
    labels = {str(v) for v in names.values()}
    if labels == set(AV_CLASSES):
        return SCHEMA_AV
    if config.HAZARD_CLASSES <= labels:      # COCO, or any superset of it
        return SCHEMA_COCO
    missing = sorted(config.HAZARD_CLASSES - labels)
    raise ValueError(
        f"Unrecognised detector class schema ({len(labels)} classes). A model "
        f"must expose either the {len(AV_CLASSES)} AV classes or all COCO "
        f"hazard classes; this one is missing {missing[:6]}"
        f"{' …' if len(missing) > 6 else ''}. Refusing to load — an unknown "
        "schema would silently disable hazard alerts."
    )


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        path = config.OBJECT_MODEL_CUSTOM if os.path.exists(
            config.OBJECT_MODEL_CUSTOM) else config.OBJECT_MODEL
        model = YOLO(path)
        # validate BEFORE publishing to the global: a rejected model must not
        # leave a half-initialised _model behind for the next call to reuse
        model._av_schema = detect_schema(model.names)
        _model = model
    return _model


def active_schema():
    """Schema of the loaded model, or None if it hasn't been loaded yet."""
    return getattr(_model, "_av_schema", None) if _model is not None else None


def detect_objects(frame_bgr):
    model = _get_model()
    res = model.predict(frame_bgr, conf=config.OBJ_CONF, imgsz=config.IMAGE_SIZE, iou=config.NMS_THRESHOLD, device=config.DEVICE, verbose=False)[0]
    objects, hazards = [], []
    names = res.names
    if res.boxes is None:
        return objects, hazards
    av = model._av_schema == SCHEMA_AV
    for xyxy, conf, cls in zip(res.boxes.xyxy.cpu().numpy(),
                               res.boxes.conf.cpu().numpy(),
                               res.boxes.cls.cpu().numpy()):
        label = names[int(cls)]
        # COCO base model: filter to curated classes; AV-6 model: keep all
        if not av and label not in config.OBJECT_CLASSES:
            continue
        # noisy classes (person on blurry frames, soft furnishings) need
        # a stricter gate than the global threshold
        if float(conf) < config.CLASS_CONF.get(label, config.OBJ_CONF):
            continue
        spoken = AV_SPOKEN.get(label, label) if av else label
        item = {
            "label": spoken,
            "raw": label,
            "box": tuple(int(v) for v in xyxy),
            "conf": float(conf),
            "kind": "object",
        }
        hazard = (label in AV_HAZARDS) if av else (label in config.HAZARD_CLASSES)
        if hazard:
            hazards.append({**item, "kind": "hazard"})
        else:
            objects.append(item)
    return objects, hazards
