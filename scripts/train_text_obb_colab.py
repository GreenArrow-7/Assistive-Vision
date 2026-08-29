"""One Colab session -> models/text_obb.pt (YOLOv8-OBB oriented text detector).

Paste into a Colab cell (Runtime > Change runtime type > T4 GPU) or run locally.
Cells are marked `# %%` so it works either way.

ICDAR-2015 needs registration (rrc.cvc.uab.es), so it cannot be auto-downloaded.
Upload these four to Drive once, then point ICDAR_DIR at the folder holding them:
    ch4_training_images.zip          ch4_training_localization_transcription_gt.zip
    ch4_test_images.zip              Challenge4_Test_Task1_GT.zip

SynthText pretraining is optional (41 GB torrent). ICDAR alone trains a usable
detector; SynthText mainly helps recall on small/dense text.
"""
# %% ---------------------------------------------------------------- config
ICDAR_DIR = "/content/drive/MyDrive/icdar15"   # folder holding the 4 zips
SYNTH_ROOT = ""                                # optional: SynthText dir w/ gt.mat
SYNTH_MAX = 30000                              # images to use if pretraining
EPOCHS = 80
IMGSZ = 960                                    # text is small; 640 loses it
BATCH = 8                                      # T4-safe at 960
WORK = "/content/av_text"                      # scratch

# %% ---------------------------------------------------------------- setup
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                    "ultralytics", "scipy"], check=True)
    from google.colab import drive
    drive.mount("/content/drive")

REPO = Path(__file__).resolve().parent.parent if "__file__" in dir() \
    else Path("/content/Assistive Vision")
sys.path.insert(0, str(REPO / "scripts"))

work = Path(WORK)
work.mkdir(parents=True, exist_ok=True)

# %% ------------------------------------------------------ unzip + convert
ZIPS = {
    "train_images": "ch4_training_images.zip",
    "train_gts": "ch4_training_localization_transcription_gt.zip",
    "val_images": "ch4_test_images.zip",
    "val_gts": "Challenge4_Test_Task1_GT.zip",
}

raw = work / "icdar_raw"
for key, zname in ZIPS.items():
    dst = raw / key
    if dst.exists():
        continue
    src = Path(ICDAR_DIR) / zname
    if not src.exists():
        raise SystemExit(
            f"missing {src}\nUpload the four ICDAR-2015 zips to {ICDAR_DIR} "
            "(register at rrc.cvc.uab.es to download them)."
        )
    dst.mkdir(parents=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(dst)
    print(f"unzipped {zname} -> {dst}")

from convert_icdar_to_obb import convert as icdar_convert  # noqa: E402

data_root = work / "icdar15_obb"
icdar_convert(raw / "train_images", raw / "train_gts", data_root / "train")
icdar_convert(raw / "val_images", raw / "val_gts", data_root / "val")

# %% -------------------------------------------- optional SynthText pretrain
pretrain_yaml = None
if SYNTH_ROOT:
    from convert_synthtext_to_obb import convert as synth_convert  # noqa: E402

    synth_out = work / "synthtext_obb"
    if not (synth_out / "train" / "images").exists():
        synth_convert(Path(SYNTH_ROOT), synth_out, SYNTH_MAX)
    # reuse ICDAR val so pretrain and finetune are scored on the same yardstick
    pretrain_yaml = work / "synth.yaml"
    pretrain_yaml.write_text(
        f"path: {synth_out.resolve()}\ntrain: train/images\n"
        f"val: {(data_root / 'val' / 'images').resolve()}\nnames:\n  0: text\n")

# %% ---------------------------------------------------------------- yaml
# written at runtime: the checked-in scripts/icdar15_obb.yaml uses a repo-relative
# path that does not resolve under /content
data_yaml = work / "icdar15_obb.yaml"
data_yaml.write_text(
    f"path: {data_root.resolve()}\ntrain: train/images\nval: val/images\n"
    "names:\n  0: text\n")
print(data_yaml.read_text())

# %% ---------------------------------------------------------------- train
from ultralytics import YOLO  # noqa: E402

weights = "yolov8n-obb.pt"
if pretrain_yaml:
    print("=== stage 1: SynthText pretrain ===")
    m = YOLO(weights)
    m.train(data=str(pretrain_yaml), epochs=max(10, EPOCHS // 4), imgsz=IMGSZ,
            batch=BATCH, name="text_obb_pretrain", degrees=30, patience=15)
    weights = str(Path(m.trainer.save_dir) / "weights" / "best.pt")
    print("pretrained ->", weights)

print("=== stage 2: ICDAR-2015 finetune ===")
model = YOLO(weights)
model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
            name="text_obb", degrees=30, patience=20, close_mosaic=15)

best = Path(model.trainer.save_dir) / "weights" / "best.pt"
metrics = YOLO(str(best)).val(data=str(data_yaml), imgsz=IMGSZ)
print(f"\nmAP50 {metrics.box.map50:.3f} | mAP50-95 {metrics.box.map:.3f}")

# %% ------------------------------------------------------ verify + install
out = REPO / "models" / "text_obb.pt"
out.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(best, out)

# the server refuses a non-obb model here, so fail now rather than at runtime
check = YOLO(str(out))
assert check.task == "obb", f"trained model has task={check.task}, expected obb"
print(f"\nOK -> {out}")
print("Restart the server; /health should now report text_obb: true")

if IN_COLAB:
    from google.colab import files
    files.download(str(best))
