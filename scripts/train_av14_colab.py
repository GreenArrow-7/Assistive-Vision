"""One Colab session -> models/av_obstacle.pt (AV-7 obstacle detector).

Local prep (this machine, no GPU needed):
  1. python scripts/merge_review_queue.py --out datasets/av14_merged
     python scripts/reindex_labels.py --src datasets/av14_merged \
         --out datasets/av7_merged
  2. python scripts/pull_open_datasets.py --per-class 400 --out datasets/oi_av14
     python scripts/reindex_labels.py --src datasets/oi_av14 --out datasets/oi_av7
  3. python scripts/prepare_split.py --src datasets/av7_merged \
         --out datasets/av7_split --schema av7 --extra datasets/oi_av7
  4. zip datasets/av7_split -> av7_split.zip -> upload to Google Drive
     (prepare_split.py already wrote it; verified to pass the gate below)

Then paste this file into a Colab cell (Runtime > T4 GPU) and run.
Cells are marked `# %%` so it also runs top-to-bottom as a script.
"""
# %% ---------------------------------------------------------------- config
DATASET_ZIP = "/content/drive/MyDrive/av7_split.zip"
EPOCHS = 120
IMGSZ = 832        # signs are small objects; 640 loses them at distance
BATCH = 12         # T4-safe at 832
MODEL = "yolov8s.pt"

# %% ---------------------------------------------------------------- setup
import subprocess
import sys
import zipfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                    "ultralytics", "pyyaml"], check=True)
    from google.colab import drive
    drive.mount("/content/drive")

work = Path("/content/av7" if IN_COLAB else "datasets/av7_colab")
if not (work / "data.yaml").exists():
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATASET_ZIP) as z:
        z.extractall(work)
    # the zip may nest the split dir one level down
    if not (work / "data.yaml").exists():
        inner = next(work.glob("*/data.yaml"))
        work = inner.parent
print("dataset root:", work)

# prepare_split.py writes no "path" key on purpose, so ultralytics roots the
# dataset at data.yaml's own directory and the zip trains wherever it lands.
# Pinning it here is belt-and-braces for exports that DO carry one: a Roboflow
# export ships the absolute path of the machine that built it, which resolves
# to nothing on Colab.
import yaml
cfg = yaml.safe_load((work / "data.yaml").read_text())
cfg["path"] = str(work.resolve())
(work / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
# The vocabulary comes from data.yaml, not a number baked in here: the
# schema is data-driven, and a trainer that disagrees with the split it was
# given would train names the server then refuses.
AV_NAMES = [cfg["names"][i] for i in range(len(cfg["names"]))]
assert AV_NAMES, "data.yaml declares no class names"
print(f"schema: {len(AV_NAMES)} classes -> {AV_NAMES}")

# %% ------------------------------------------------------- coverage gate
# Weights trained here carry the split's class names, so server/detector.py
# identifies them as "av7" and applies the full AV_HAZARDS set -- including
# stairs_down, the sole AV_CRITICAL class. A model that cannot emit a class it
# claims to speak is the exact silent-capability gap detect_schema exists to
# catch, arriving from the dataset side. Counting the boxes takes a second;
# discovering it after a 120-epoch T4 run does not.
ALLOW_SPARSE = False        # True: knowingly train a partial research model

from collections import Counter  # noqa: E402


def _hist(split):
    h = Counter()
    for lbl in (work / split / "labels").glob("*.txt"):
        for line in lbl.read_text().splitlines():
            if line.strip():
                h[int(line.split()[0])] += 1
    return h


_tr, _va = _hist("train"), _hist("val")
_names = AV_NAMES
print(f"{'class':<18}{'train':>8}{'val':>8}")
for _i, _n in enumerate(_names):
    print(f"{_n:<18}{_tr[_i]:>8}{_va[_i]:>8}")

_empty = [n for i, n in enumerate(_names) if not _tr[i] and not _va[i]]
# a class the model never sees in training cannot be learned; its only effect
# is to drag the val metrics down and make the run look worse than it is
_val_only = [n for i, n in enumerate(_names) if not _tr[i] and _va[i]]

_problems = []
if _empty:
    _problems.append(f"{len(_empty)} class(es) have NO boxes at all: "
                     + ", ".join(_empty))
if _val_only:
    _problems.append(f"{len(_val_only)} class(es) appear only in val, never in "
                     "train: " + ", ".join(_val_only))

COMPLETE = not _problems
if _problems:
    _msg = "\n".join("  - " + p for p in _problems)
    if not ALLOW_SPARSE:
        raise SystemExit(
            "Dataset coverage is incomplete:\n" + _msg + "\n\n"
            "Annotate the gaps, or set ALLOW_SPARSE = True to train anyway.\n"
            "A partial model is a research artifact ONLY: it must not be\n"
            "installed at models/av_obstacle.pt, where it would report\n"
            "object_schema 'av7' while silently never raising those hazards.")
    print("\nWARNING -- training a PARTIAL model:\n" + _msg)

# %% ---------------------------------------------------------------- train
from ultralytics import YOLO  # noqa: E402

model = YOLO(MODEL)
model.train(data=str(work / "data.yaml"), epochs=EPOCHS, imgsz=IMGSZ,
            batch=BATCH, patience=30, name="av7",
            degrees=8, hsv_v=0.5, fliplr=0.5, mosaic=1.0, close_mosaic=15)

best = Path(model.trainer.save_dir) / "weights" / "best.pt"

# per-class AP -> paper Table T3. Watch stairs_down and the sign_* classes:
# they are the classes that justify the whole contribution.
metrics = YOLO(str(best)).val(data=str(work / "data.yaml"), imgsz=IMGSZ)
print(f"mAP50 {metrics.box.map50:.3f} | mAP50-95 {metrics.box.map:.3f}")

# %% ------------------------------------------------------ verify + install
# same check the server runs at startup: the weights must speak the schema by NAME
m = YOLO(str(best))
names = set(str(v) for v in m.names.values())
assert names == set(AV_NAMES), (
    f"class names mismatch, server would refuse: {sorted(names)}")

# Matching names is necessary but NOT sufficient: they are exactly what makes a
# partial model dangerous, because they are what detect_schema trusts.
if COMPLETE:
    print("OK — rename to av_obstacle.pt, place at models/av_obstacle.pt,")
    print("restart the server; /health should report object_schema: av7")
else:
    print("DO NOT DEPLOY — this model was trained with ALLOW_SPARSE.")
    print("Its names match the schema, so the server would trust it with the")
    print("full hazard set while it can never emit the untrained classes.")
    print("Keep it as a research artifact / ablation row only.")

if IN_COLAB:
    from google.colab import files
    files.download(str(best))
