"""One Colab session -> models/text_obb.pt, with no dataset registration.

Paste into a Colab cell (Runtime > Change runtime type > T4 GPU) or run locally.
Cells are marked `# %%` so it works either way. This is Recipe C of
scripts/train_obb.md; train_text_obb_colab.py is the ICDAR-2015 sibling.

COCO-Text and COCO train2014 are both open downloads, so the whole run is
unattended -- no account, no Drive upload. The cost is domain: COCO-Text is
mostly small incidental text in everyday photos, so the detector is weaker on
the blurry rotated signboards ICDAR-15 is made of. Same class (0 = text), so
the ICDAR recipe later resumes from these weights and only improves them.
"""
# %% ---------------------------------------------------------------- config
EPOCHS = 60                                    # ~20x more images than ICDAR, so fewer passes
IMGSZ = 960                                    # text is small; 640 loses it
BATCH = 8                                      # T4-safe at 960
WORK = "/content/av_text"                      # scratch; needs ~20 GB free
REPO_DIR = "/content/Assistive-Vision"         # where the repo was cloned (paste-into-Colab only)

COCOTEXT_URL = "https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip"
IMAGES_URL = "http://images.cocodataset.org/zips/train2014.zip"  # 12.6 GB

# %% ---------------------------------------------------------------- setup
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                    "ultralytics"], check=True)

REPO = Path(__file__).resolve().parent.parent if "__file__" in dir() \
    else Path(REPO_DIR)
# Checked here, not at the import below: a wrong REPO_DIR otherwise fails as a bare ImportError.
if not (REPO / "scripts" / "convert_cocotext_to_obb.py").exists():
    raise SystemExit(f"{REPO} is not the Assistive-Vision repo. Clone it, or set REPO_DIR.")
sys.path.insert(0, str(REPO / "scripts"))

work = Path(WORK)
work.mkdir(parents=True, exist_ok=True)

# %% ---------------------------------------------------------------- download

def fetch(url: str, marker: Path):
    """Download+unzip into work/, unless marker already exists (Colab reconnects)."""
    if marker.exists():
        print(f"have {marker.name}, skipping download")
        return
    zp = work / url.rsplit("/", 1)[-1]
    if not zp.exists():
        # urllib, not wget: the script is documented to run locally too, and
        # Windows has no wget. Downloads to .part so a cut connection re-fetches.
        part = zp.with_suffix(zp.suffix + ".part")
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, part)
        part.rename(zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(work)
    zp.unlink()  # 12.6 GB: Colab disk is ~70 GB and the extracted copy is enough


fetch(COCOTEXT_URL, work / "cocotext.v2.json")
fetch(IMAGES_URL, work / "train2014")

# %% ---------------------------------------------------------------- convert
from convert_cocotext_to_obb import convert as cocotext_convert  # noqa: E402

data_root = work / "cocotext_obb"
if not (data_root / "val" / "images").exists():
    cocotext_convert(work / "cocotext.v2.json", work / "train2014", data_root)

# %% ---------------------------------------------------------------- yaml
# written at runtime: ultralytics resolves a relative `path:` against its own
# datasets_dir setting, so the checked-in scripts/cocotext_obb.yaml misses here
data_yaml = work / "cocotext_obb.yaml"
data_yaml.write_text(
    f"path: {data_root.resolve()}\ntrain: train/images\nval: val/images\n"
    "names:\n  0: text\n")
print(data_yaml.read_text())

# %% ---------------------------------------------------------------- train
from ultralytics import YOLO  # noqa: E402

model = YOLO("yolov8n-obb.pt")
model.train(data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
            name="text_obb_cocotext", degrees=30, patience=20, close_mosaic=15)

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
print("Download best.pt, drop it in models/text_obb.pt, restart the server;")
print("/health should then report text_obb: true")

if IN_COLAB:
    from google.colab import files
    files.download(str(best))
