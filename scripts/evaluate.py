"""Evaluation harness: model capacity (Track A) + deployed-system reality (Track B).

WHY TWO TRACKS. `yolo val` scores the raw weights at conf~0.001 with its own
preprocessing. It never imports server/detector.py, so it sees none of
config.CLASS_CONF (person is gated at 0.62, not 0.45), none of the
OBJECT_CLASSES filter, none of the AV_HAZARDS/HAZARD_CLASSES split, no 960 px
resize, no blur gate, no spatial/priority — and it produces no latency. mAP is
therefore an upper bound on a MODEL, not a measurement of the SYSTEM. Track A
delegates to ultralytics anyway (its 101-point AP, 10-IoU TP matching and NMS
conventions are correct and a reviewer trusts them more than ours); Track B
drives the real FastAPI app over TestClient, so nothing here re-implements
pipeline logic and nothing here can drift out of sync with server/main.py.

WHY THE SELF-EVAL REFUSAL. datasets/av_raw's labels were WRITTEN by yolov8s.
Scoring yolov8n/8s against them measures agreement between two YOLO
checkpoints, not accuracy, and prints a flattering ~0.9 mAP. A run where both
the model and the ground truth speak COCO is refused unless --allow-selfeval,
and every table such a run writes is stamped as self-consistency. Until the
annotation pass lands, latency is the only honest number in the file.

Usage:
  python scripts/evaluate.py --weights yolov8n.pt --data datasets/av14_seed_split/data.yaml \
      --tag coco_baseline --limit 200
  python scripts/evaluate.py --compare runs/eval/coco_baseline.json runs/eval/av14.json
"""
import argparse
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

# Windows consoles here are cp1252: an em dash in a status line would otherwise
# kill the run with UnicodeEncodeError after the models were already loaded.
# Report files are written utf-8 explicitly and keep their real characters.
# line_buffering because a redirected 30-minute CPU run must show progress
# rather than sitting in an 8 KB block buffer until it exits.
for _s in (sys.stdout, sys.stderr):
    getattr(_s, "reconfigure", lambda **k: None)(errors="replace", line_buffering=True)

from prepare_split import video_of                        # noqa: E402
from server import config, detector                       # noqa: E402
from server.classes_av import AV_ALL_CLASSES, AV_CLASSES, AV_HAZARDS      # noqa: E402

# The subsets the paper actually claims. Macro mAP over all classes hides both:
# a sign_* class with 7 boxes moves it as much as person with 2512.
# The classes the paper's safety claim rests on. Retired ones drop out of the
# reported subset automatically and return the moment they are trained again,
# so the headline number never silently covers a class the model cannot emit.
SAFETY_ALL = ["stairs_down", "stairs_up", "pole", "door"]
SAFETY = [c for c in SAFETY_ALL if c in AV_CLASSES]
SIGNS = [c for c in AV_CLASSES if c.startswith("sign")]   # signboard + sign_*
# Guards a TYPO, not a schema change: names must exist in the annotation
# vocabulary even when they are not currently trained.
assert set(SAFETY_ALL) <= set(AV_ALL_CLASSES),     f"SAFETY names unknown: {set(SAFETY_ALL) - set(AV_ALL_CLASSES)}"
LOW_SUPPORT = 30                        # below this, a per-class AP is noise
STAGES = [("prep", None), ("detect", "detect_objects"), ("ocr", "detect_text"),
          ("spatial", "annotate"), ("priority", "build_speech"), ("total", None)]


# ------------------------------------------------------------------ pure bits
def pct(xs, q):
    """Nearest-rank percentile — no interpolation flavour to argue about."""
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def stats(xs):
    if not xs:
        return None
    return {"p50": round(pct(xs, .50), 1), "p90": round(pct(xs, .90), 1),
            "p99": round(pct(xs, .99), 1), "mean": round(sum(xs) / len(xs), 1),
            "max": round(max(xs), 1), "n": len(xs)}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, thr=0.5):
    """Greedy IoU match at the DEPLOYED confidence. preds [(name, box, conf)].

    Highest-confidence prediction claims the best free GT first, which is the
    same order NMS-style evaluation uses. Deliberately no AP integration here:
    that is Track A's job, and a hand-rolled PR curve is the classic way an
    eval harness quietly lies.
    """
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for name, _ in gts:
        counts[name]
    used = set()
    for name, box, _ in sorted(preds, key=lambda p: -p[2]):
        best, hit = thr, None
        for i, (gn, gb) in enumerate(gts):
            if i in used or gn != name:
                continue
            v = iou(box, gb)
            if v >= best:
                best, hit = v, i
        if hit is None:
            counts[name]["fp"] += 1
        else:
            used.add(hit)
            counts[name]["tp"] += 1
    for i, (gn, _) in enumerate(gts):
        if i not in used:
            counts[gn]["fn"] += 1
    return counts


def prf(c):
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": round(p, 3), "r": round(r, 3), "f1": round(f, 3)}


def leak_check(train_stems, val_stems):
    """Videos on BOTH sides of the split. 1 fps frames from one walkthrough are
    near-identical, so a single shared video turns mAP into fiction."""
    return sorted({video_of(s) for s in train_stems} & {video_of(s) for s in val_stems})


def table(head, rows):
    if not rows:
        return "_(none)_"
    return "\n".join(["| " + " | ".join(head) + " |",
                      "|" + "|".join("---" for _ in head) + "|"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def gt_boxes(path, names, w, h):
    """YOLO normalized cx,cy,w,h -> pixel xyxy in the POST-RESIZE frame."""
    out = []
    for line in path.read_text().splitlines() if path.exists() else []:
        p = line.split()
        if len(p) < 5:
            continue
        cls, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
        out.append((names.get(cls, str(cls)),
                    ((cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


# ------------------------------------------------------------------ data.yaml
def load_data(path):
    y = yaml.safe_load(Path(path).read_text())
    names = y["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else names
    return y, {int(k): str(v) for k, v in names.items()}, \
        Path(y.get("path") or Path(path).parent)


def split_dirs(y, base, split):
    rel = y.get(split)
    if not rel:
        return None, None
    img = Path(rel) if Path(rel).is_absolute() else base / rel
    return img, img.parent / "labels"


# ------------------------------------------------------------------ Track A
def track_a(model, data, split, device, out, tag, names, n_gt):
    # resolve(): a RELATIVE project is joined onto ultralytics' own settings
    # runs_dir, which buried our output at runs/detect/runs/eval/<tag>_ultralytics
    r = model.val(data=str(data), split=split, device=device, plots=False,
                  verbose=False, workers=0, project=str(Path(out).resolve()),
                  name=f"{tag}_ultralytics", exist_ok=True)
    b = r.box
    per = {}
    for i, ci in enumerate(b.ap_class_index):
        name = names.get(int(ci), str(int(ci)))
        per[name] = {"p": round(float(b.p[i]), 3), "r": round(float(b.r[i]), 3),
                     "ap50": round(float(b.ap50[i]), 3),
                     "ap50_95": round(float(b.ap[i]), 3),
                     "n_gt": n_gt.get(name, 0)}
    sup = sum(v["n_gt"] for v in per.values())

    def subset(classes):
        vals = [per[c]["ap50"] for c in classes if c in per]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {"map50": round(float(b.map50), 3), "map50_95": round(float(b.map), 3),
            # imbalance-honest second number: macro treats 7 boxes like 2512
            "map50_weighted": round(sum(v["ap50"] * v["n_gt"] for v in per.values())
                                    / sup, 3) if sup else None,
            "map50_safety": subset(SAFETY), "map50_signs": subset(SIGNS),
            "per_class": per}


# ------------------------------------------------------------------ Track B
def _timed(mod, name, sink):
    """Wrap a call site in place — measures the real pipeline without editing
    server code, so the harness cannot drift away from what ships."""
    fn = getattr(mod, name)

    def wrap(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            sink[name].append((time.perf_counter() - t) * 1000)
    setattr(mod, name, wrap)


def track_b(a, frames, lbl_dir, names, hazards, name_map):
    from fastapi.testclient import TestClient

    from server import main, priority, spatial, text_pipeline

    if a.no_ocr:      # isolates detector latency AND skips the EasyOCR load
        text_pipeline.detect_text = lambda img: []
        text_pipeline._get_reader = lambda: None

    sink = defaultdict(list)
    for mod, fn in ((detector, "detect_objects"), (text_pipeline, "detect_text"),
                    (spatial, "annotate"), (priority, "build_speech")):
        _timed(mod, fn, sink)

    lat, cls_counts = defaultdict(list), defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    hz, blur = Counter(), 0
    t0 = time.perf_counter()
    with TestClient(main.app) as client:          # lifespan starts main._warmup
        while True:
            h = client.get("/health").json()
            if h["error"]:
                raise SystemExit("model warmup failed: " + h["error"])
            if h["ready"]:
                break
            time.sleep(0.05)     # the poll interval quantises cold_start; keep it
            #                      well under the 0.1 s the number is reported to
        # cold start is a real deployment number and pure poison inside a mean
        cold = round(time.perf_counter() - t0, 1)
        schema = h["object_schema"]

        for i, img in enumerate(frames):
            mark = {k: len(v) for k, v in sink.items()}
            r = client.post("/analyze", data={"mode": "single"},
                            files={"frame": (img.name, img.read_bytes(), "image/jpeg")})
            if r.status_code != 200:
                raise SystemExit(f"/analyze {r.status_code} on {img.name}: {r.text[:200]}")
            js = r.json()
            if js.get("blur"):        # short-circuits before detect: not a sample
                blur += 1
                continue
            if i < a.warmup:          # first frames absorb lazy loads and autotune
                continue

            stage = {k: sum(v[mark.get(k, 0):]) for k, v in sink.items()}
            total = float(js["ms"])
            for nice, key in STAGES:
                if key:
                    lat[nice].append(stage.get(key, 0.0))
            lat["prep"].append(max(0.0, total - sum(stage.values())))
            lat["total"].append(total)

            if lbl_dir is None:
                continue
            w, h_px = js["frame"]["w"], js["frame"]["h"]
            gts = gt_boxes(lbl_dir / f"{img.stem}.txt", names, w, h_px)
            preds = [((p.get("raw") or p["label"]), tuple(p["box"]), p["conf"])
                     for p in js["hazards"] + js["objects"]]
            if name_map:
                # unmapped predictions are IGNORED, not counted FP: a COCO
                # model's "tv" is not a false positive against a vocabulary
                # that has no tv. Asymmetry noted in the table caption.
                preds = [(name_map[n], b, c) for n, b, c in preds if n in name_map]
            for k, v in match(preds, gts, a.iou).items():
                for f in ("tp", "fp", "fn"):
                    cls_counts[k][f] += v[f]
            # vocabulary-independent, so comparable across COCO and AV-7.
            # GT is gated by the SAME proximity rule the server uses: main.py
            # demotes "at a distance" hazards to plain objects on purpose, so
            # hazard_count never counts them. Scoring every distant hazard box as
            # a missed alert measures the system against behaviour it deliberately
            # does not have — on av_finetune/val 223 of 398 frames hold a
            # hazard-class box but only 160 hold a close one, capping recall at
            # 0.72 before the detector runs. Same function, so the two cannot drift.
            gt_hz = any(n in hazards and spatial.proximity(b, w, h_px) != "at a distance"
                        for n, b in gts)
            said_hz = js["hazard_count"] > 0
            hz["tp" if gt_hz and said_hz else "fp" if said_hz else
               "fn" if gt_hz else "tn"] += 1

    per_class = {k: {**v, **prf(v), "n_gt": v["tp"] + v["fn"]}
                 for k, v in sorted(cls_counts.items())}
    hzd = {"tp": hz["tp"], "fp": hz["fp"], "fn": hz["fn"], **prf(hz)}
    lat_out = {n: stats(lat[n]) for n, _ in STAGES}
    out = {"iou": a.iou, "conf": "deployed", "schema_seen": schema, "no_ocr": a.no_ocr,
           "per_class": per_class if lbl_dir else {},
           "hazard_frame": hzd if lbl_dir else {}, "latency_ms": lat_out,
           "cold_start_s": cold, "blur_rejected": blur,
           "fps_p50": round(1000 / lat_out["total"]["p50"], 2)
           if lat_out["total"] and lat_out["total"]["p50"] else None}
    if lat_out["total"] and lat_out["total"]["n"] < 100:
        out["p99_note"] = "n<100, p99 == max"
    return out


# ------------------------------------------------------------------ reporting
def markdown(d):
    m = [f"# eval: {d['tag']}", "",
         f"weights `{d['weights']}` · model schema **{d['model_schema']}** · "
         f"GT schema **{d['gt_schema']}** · data `{d['data']}`",
         f"host: {d['host']['cpu']} · python {d['host']['python']} · "
         f"torch {d['host']['torch']}", ""]
    if d.get("selfeval"):
        m += ["> **SELF-CONSISTENCY vs pre-labels, NOT ground truth.** The labels "
              "scored here were generated by yolov8s, so the detection numbers "
              "measure agreement between two YOLO checkpoints. Only the latency "
              "and cold-start rows are real.", ""]
    c = d["counts"]
    if d["gt_schema"] is None:
        # latency-only fallback: say so, or "0 boxes" reads as a measured zero
        m += [f"**No labelled split** — {c['val_frames']} frames used as a latency "
              f"corpus only ({c['frames_timed']} timed, {c['blur_rejected']} "
              "blur-rejected). Every accuracy section is absent, not zero.", ""]
    else:
        m += [f"val: {c['val_frames']} frames / {c['val_videos']} videos / "
              f"{c['val_boxes']} boxes / {c['empty_labels']} empty label files "
              f"({c['frames_timed']} timed, {c['blur_rejected']} blur-rejected) · "
              # --drop-empty filters the frames THIS script pushes; ultralytics
              # val re-reads the split directory itself, so it cannot be honoured
              # there. Say which table it applies to or the caption is a lie.
              + ("empty-label frames **excluded from the operating-point rows only** "
                 "(ultralytics val re-reads the whole split directory, so any "
                 "detection table below is unfiltered) "
                 if c['drop_empty'] else "empty-label frames **kept** ")
              + "(an empty file means the pre-labeller found nothing, not that a human "
              "verified nothing is there — quote no precision number without this)", ""]

    if d.get("detect"):
        det = d["detect"]
        # "n/a" not "None": the subset is undefined when the vocabulary has no
        # such classes, which must not read as a score of zero
        na = {k: ("n/a" if v is None else v) for k, v in det.items() if k != "per_class"}
        m += ["## Detection (ultralytics val, conf~0.001 — model capacity)", "",
              f"mAP50 **{na['map50']}** · mAP50-95 **{na['map50_95']}** · "
              f"support-weighted mAP50 **{na['map50_weighted']}** · "
              f"SAFETY **{na['map50_safety']}** · SIGNS **{na['map50_signs']}**", "",
              table(["class", "P", "R", "AP50", "AP50-95", "n_gt"],
                    [[k, v["p"], v["r"], v["ap50"], v["ap50_95"],
                      f"{v['n_gt']}{' ⚠' if v['n_gt'] < LOW_SUPPORT else ''}"]
                     for k, v in sorted(det["per_class"].items())]),
              "", f"⚠ = under {LOW_SUPPORT} boxes: treat that row as noise, not a result.", ""]
    elif d.get("detect_skipped"):
        m += ["## Detection (ultralytics val)", "", "**Skipped.** " + d["detect_skipped"], ""]

    p = d.get("pipeline")
    if p:
        if p["per_class"]:
            m += ["## Deployed operating point (server thresholds, "
                  f"IoU>={p['iou']} — system accuracy)", ""]
            m += [table(["class", "TP", "FP", "FN", "P", "R", "F1", "n_gt"],
                        [[k, v["tp"], v["fp"], v["fn"], v["p"], v["r"], v["f1"],
                          v["n_gt"]] for k, v in p["per_class"].items()]), ""]
            hzd = p["hazard_frame"]
            m += [f"**Frame-level hazard alert** P {hzd['p']} · R {hzd['r']} · "
                  f"F1 {hzd['f1']} (TP {hzd['tp']} / FP {hzd['fp']} / FN {hzd['fn']}) "
                  "— a frame counts as a hazard frame when it holds a hazard-class "
                  "box that is **not** \"at a distance\", which is exactly when the "
                  "server raises an alert (distant hazards are demoted to objects on "
                  "purpose). Vocabulary-independent: the one row comparable across "
                  "COCO and AV-7 runs.", ""]
        if d.get("names_map"):
            m += ["_Predictions with no entry in the name map are ignored rather "
                  "than counted as false positives; the COCO model structurally "
                  "cannot express most AV-7 classes, and that structural zero is "
                  "the finding, not a fair fight._", ""]
        dag = "†" if "p99_note" in p else ""
        m += ["## Latency (server-side; phone capture, upload and TTS excluded)", "",
              table(["stage", "p50", "p90", f"p99{dag}", "mean", "max", "n"],
                    [[n] + [p["latency_ms"][n][k] for k in
                            ("p50", "p90", "p99", "mean", "max", "n")]
                     for n, _ in STAGES if p["latency_ms"].get(n)]), "",
              f"throughput **{p['fps_p50']} fps** (1000/total p50) · cold start "
              f"**{p['cold_start_s']} s** "
              + ("(YOLO load only — **--no-ocr**, EasyOCR was stubbed out, so "
                 "the ocr row is 0 by construction)" if p.get("no_ocr") else
                 "(YOLO + EasyOCR load)")
              + ", measured once and excluded from every row above"]
        if dag:
            m += ["", "† fewer than 100 timed frames: p99 == max."]
    return "\n".join(m) + "\n"


def compare(paths, out):
    runs = [json.loads(Path(p).read_text()) for p in paths]
    tags = [r["tag"] for r in runs]
    m = ["# eval comparison", "", " vs ".join(f"`{t}`" for t in tags), ""]

    classes = sorted({c for r in runs for c in r.get("detect", {}).get("per_class", {})})
    if classes:
        # ASCII on purpose: this table is printed to a cp1252 console as well
        # as written to disk
        head = ["class"] + [f"AP50 {t}" for t in tags] + (["delta"] if len(runs) == 2 else []) + ["n_gt"]
        rows = []
        for c in classes:
            vals = [r.get("detect", {}).get("per_class", {}).get(c) for r in runs]
            aps = [v["ap50"] if v else None for v in vals]
            row = [c] + ["-" if a is None else a for a in aps]
            if len(runs) == 2:
                row.append(round(aps[1] - aps[0], 3) if None not in aps else "-")
            row.append(max((v["n_gt"] for v in vals if v), default=0))
            rows.append(row)
        # null means "this vocabulary has no such class", not zero — render it
        # as a dash so nobody averages it into a headline number
        def cell(r, key):
            v = r.get("detect", {}).get(key)
            return "-" if v is None else v

        summary = [[label] + [cell(r, key) for r in runs] for label, key in
                   (("mAP50", "map50"), ("weighted mAP50", "map50_weighted"),
                    ("SAFETY", "map50_safety"), ("SIGNS", "map50_signs"))]
        m += ["## Per-class AP50", "", table(head, rows), "",
              table(["metric"] + tags, summary), ""]

    lat_rows = [[n] + [(r.get("pipeline", {}).get("latency_ms", {}).get(n) or {}).get("p50", "-")
                       for r in runs] for n, _ in STAGES]
    lat_rows.append(["fps_p50"] + [r.get("pipeline", {}).get("fps_p50", "-") for r in runs])
    lat_rows.append(["cold_start_s"] + [r.get("pipeline", {}).get("cold_start_s", "-") for r in runs])
    m += ["## Latency p50 (ms)", "", table(["stage"] + tags, lat_rows), "",
          "## Frame-level hazard alert", "",
          table(["metric"] + tags,
                [[k] + [r.get("pipeline", {}).get("hazard_frame", {}).get(k, "-") for r in runs]
                 for k in ("p", "r", "f1", "tp", "fp", "fn")]), ""]
    if any(r.get("selfeval") for r in runs):
        m += ["> One or more runs are SELF-CONSISTENCY vs pre-labels, not ground "
              "truth. Detection rows are not accuracy.", ""]
    text = "\n".join(m)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out / 'compare.md'}")


# ------------------------------------------------------------------ main
def main(a):
    if a.compare:
        return compare(a.compare, a.out)

    weights = a.weights or (config.OBJECT_MODEL_CUSTOM
                            if Path(config.OBJECT_MODEL_CUSTOM).exists()
                            else config.OBJECT_MODEL)
    tag = a.tag or Path(weights).stem
    a.out.mkdir(parents=True, exist_ok=True)

    y = names = lbl_dir = None
    frames, train_stems = [], []
    if Path(a.data).exists():
        y, names, base = load_data(a.data)
        img_dir, lbl_dir = split_dirs(y, base, a.split)
        if img_dir and img_dir.exists():
            frames = sorted(img_dir.glob("*.jpg"))
        tr_img, _ = split_dirs(y, base, "train")
        train_stems = [p.stem for p in tr_img.glob("*.jpg")] if tr_img and tr_img.exists() else []
    if not frames:
        # degrade gracefully: no split yet is the normal state before the
        # annotation pass, and latency is measurable without any labels
        pool = [p for p in sorted((ROOT / "datasets").glob("*/images"))
                if any(p.glob("*.jpg"))]
        if not pool:
            raise SystemExit(f"no val images from {a.data} and no datasets/*/images "
                             "to fall back on — nothing to measure")
        frames, lbl_dir, names = sorted(pool[0].glob("*.jpg")), None, None
        print(f"! no {a.split} split at {a.data}: latency only, from {pool[0]}")

    # ---- counts + guards, before anything expensive loads ----
    n_gt, empty, boxes = Counter(), 0, 0
    # the counts describe the SPLIT, so they stay on the pre-drop list: reporting
    # "240 frames / 158 empty label files" right after dropping all 158 of them is
    # a contradiction, and frames_timed already says how many were measured.
    scanned = frames
    if lbl_dir:
        for f in frames:
            g = gt_boxes(lbl_dir / f"{f.stem}.txt", names, 1, 1)
            empty += not g
            boxes += len(g)
            n_gt.update(n for n, _ in g)
        if a.drop_empty:
            frames = [f for f in frames
                      if gt_boxes(lbl_dir / f"{f.stem}.txt", names, 1, 1)]
        bad = leak_check(train_stems, [f.stem for f in frames])
        if bad:
            raise SystemExit(f"SPLIT LEAK: {len(bad)} video(s) in both train and "
                             f"{a.split}: {bad[:8]} — near-identical frames on both "
                             "sides make every number below fiction")
        print(f"leak check: OK ({a.split} videos disjoint from train)")

    gt_schema = model_schema = None
    name_map = json.loads(Path(a.names_map).read_text()) if a.names_map else {}
    model = None
    if names:
        from ultralytics import YOLO
        model = YOLO(str(weights))
        model_schema = detector.detect_schema(model.names)   # raises on unknown
        gt_schema = detector.detect_schema(names)
        if model_schema == gt_schema == detector.SCHEMA_COCO and not a.allow_selfeval:
            raise SystemExit(
                "REFUSING: model and ground truth are both COCO. datasets/av_raw's "
                "labels were GENERATED by yolov8s, so this scores agreement between "
                "two YOLO checkpoints, not accuracy — expect a fantasy ~0.9 mAP. "
                "Correct the labels to AV-7 (docs/TRAINING.md Step 2) or pass "
                "--allow-selfeval to get the latency numbers with every detection "
                "table stamped as self-consistency.")
        if model_schema != gt_schema and not name_map:
            # scripts/remap_to_av14.AUTO is the already-reviewed COCO->AV-14
            # table; duplicating it into a json file would just let the two
            # drift. Imported here, not at module scope, so a rename there
            # costs this run a --names-map flag rather than the whole harness.
            try:
                from remap_to_av14 import AUTO
            except ImportError as e:
                raise SystemExit(f"schemas differ ({model_schema} vs {gt_schema}) "
                                 f"and no --names-map; remap_to_av14.AUTO is "
                                 f"unavailable ({e}) — pass --names-map")
            name_map = dict(AUTO)
            print(f"! schemas differ ({model_schema} vs {gt_schema}): using "
                  f"remap_to_av14.AUTO as the name map {name_map}")
        print(f"schema: model={model_schema} gt={gt_schema}   {a.split}: "
              f"{len(scanned)} frames / {len({video_of(f.stem) for f in scanned})} "
              f"videos / {boxes} boxes / {empty} empty labels"
              + (f" ({len(frames)} kept after --drop-empty)" if a.drop_empty else ""))

    d = {"tag": tag, "weights": str(weights), "model_schema": model_schema,
         "gt_schema": gt_schema, "data": str(a.data), "names_map": name_map or None,
         "selfeval": bool(names) and model_schema == gt_schema == detector.SCHEMA_COCO,
         "host": {"cpu": platform.processor() or platform.machine(),
                  "python": platform.python_version(), "torch": _torch_version()},
         "counts": {"val_frames": len(scanned),
                    "val_videos": len({video_of(f.stem) for f in scanned}),
                    "val_boxes": boxes, "empty_labels": empty,
                    "drop_empty": a.drop_empty}}

    # ultralytics val matches predictions to labels by class INDEX, so a COCO
    # model against AV-7 labels would score its "car" (2) against "table" (2).
    # No name map can fix that from outside; Track A is simply undefined across
    # vocabularies. Track B does the mapping by name and stays valid.
    cross_vocab = bool(names) and model_schema != gt_schema
    if a.task in ("detect", "all") and names and not cross_vocab:
        d["detect"] = track_a(model, a.data, a.split, a.device, a.out, tag, names, n_gt)
        det = d["detect"]
        print(f"mAP50 {det['map50']} | mAP50-95 {det['map50_95']} | weighted mAP50 "
              f"{det['map50_weighted']} | SAFETY {det['map50_safety']} | SIGNS "
              f"{det['map50_signs']}")
        thin = [f"{k}({v['n_gt']})" for k, v in sorted(det["per_class"].items())
                if v["n_gt"] < LOW_SUPPORT]
        if thin:
            print(f"low support (<{LOW_SUPPORT} boxes, treat as noise): {' '.join(thin)}")
    elif a.task in ("detect", "all") and cross_vocab:
        d["detect_skipped"] = (
            f"ultralytics val matches by class index, so a {model_schema} model "
            f"against {gt_schema} labels scores class i against a different class "
            "i. mAP is undefined across vocabularies — use the name-mapped "
            "operating-point and hazard-frame rows below.")
        print("! detection track SKIPPED: " + d["detect_skipped"])

    if a.task in ("pipeline", "all"):
        del model                                   # free the Track A copy first
        rng = random.Random(a.seed)
        pick = frames[:]
        rng.shuffle(pick)                           # else --limit samples one video
        if a.limit:
            pick = pick[:a.limit + a.warmup]
        hazards = AV_HAZARDS if gt_schema == detector.SCHEMA_AV7 else config.HAZARD_CLASSES
        config.OBJECT_MODEL = config.OBJECT_MODEL_CUSTOM = str(weights)
        d["pipeline"] = track_b(a, pick, lbl_dir, names, hazards, name_map)
        p = d["pipeline"]
        d["counts"]["blur_rejected"] = p.pop("blur_rejected")
        # every frame blur-gated is a legitimate outcome on shaky footage, and
        # it must not crash the report on the way out
        d["counts"]["frames_timed"] = (p["latency_ms"]["total"] or {}).get("n", 0)
        if d["model_schema"] is None:
            d["model_schema"] = p["schema_seen"]
        print(table(["stage", "p50", "p90", "p99", "max"],
                    [[n] + [p["latency_ms"][n][k] for k in ("p50", "p90", "p99", "max")]
                     for n, _ in STAGES if p["latency_ms"].get(n)]))
        print(f"fps_p50 {p['fps_p50']}   cold_start {p['cold_start_s']} s")
        if p["hazard_frame"]:
            h = p["hazard_frame"]
            print(f"hazard frame  P {h['p']}  R {h['r']}  F1 {h['f1']}   "
                  f"(blur-gated frames excluded: {d['counts']['blur_rejected']})")

    d["counts"].setdefault("blur_rejected", 0)
    d["counts"].setdefault("frames_timed", 0)
    js, md = a.out / f"{tag}.json", a.out / f"{tag}.md"
    js.write_text(json.dumps(d, indent=2), encoding="utf-8")
    md.write_text(markdown(d), encoding="utf-8")
    print(f"wrote {js}, {md}")


def _torch_version():
    try:
        import torch
        return torch.__version__
    except Exception:
        return "?"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", help="default: models/av_obstacle.pt if present "
                                      "else yolov8n.pt (mirrors detector._get_model)")
    ap.add_argument("--data", type=Path, default=Path("datasets/av14_seed_split/data.yaml"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--task", default="all", choices=["detect", "pipeline", "all"])
    ap.add_argument("--tag", help="output stem, default = weights stem")
    ap.add_argument("--out", type=Path, default=Path("runs/eval"))
    ap.add_argument("--limit", type=int, default=200, help="pipeline frames, 0 = all")
    ap.add_argument("--warmup", type=int, default=3, help="frames pushed and discarded")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--names-map", help='json {"<pred name>": "<gt name>"}')
    ap.add_argument("--drop-empty", action="store_true",
                    help="skip frames with an empty label file (both counts are "
                         "reported either way)")
    ap.add_argument("--no-ocr", action="store_true", help="stub EasyOCR out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-selfeval", action="store_true")
    ap.add_argument("--compare", nargs="+", metavar="RUN.json")
    main(ap.parse_args())
