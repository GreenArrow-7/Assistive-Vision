"""Convert ICDAR-2015 Incidental Scene Text ground truth to YOLOv8-OBB format.

ICDAR GT (gt_img_*.txt, UTF-8-BOM):  x1,y1,x2,y2,x3,y3,x4,y4,transcription
YOLO-OBB label:                      0 x1 y1 x2 y2 x3 y3 x4 y4   (normalized)

Instances transcribed as "###" (unreadable) are skipped — standard practice.

Usage:
  python scripts/convert_icdar_to_obb.py \
      --images data/icdar15/train_images \
      --gts    data/icdar15/train_gts \
      --out    datasets/icdar15_obb/train
  (repeat with val split, then train:
   yolo obb train data=scripts/icdar15_obb.yaml model=yolov8n-obb.pt \
        epochs=80 imgsz=960 degrees=30)
"""
import argparse
import shutil
from pathlib import Path

from PIL import Image


def convert(images_dir: Path, gts_dir: Path, out_dir: Path):
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_img, n_box, n_skip = 0, 0, 0
    for img_path in sorted(images_dir.glob("*.jpg")):
        gt_path = gts_dir / f"gt_{img_path.stem}.txt"
        if not gt_path.exists():
            continue
        w, h = Image.open(img_path).size
        lines_out = []
        for raw in gt_path.read_text(encoding="utf-8-sig").splitlines():
            parts = raw.strip().split(",")
            if len(parts) < 9:
                continue
            transcription = ",".join(parts[8:]).strip()
            if transcription == "###":          # unreadable -> ignore
                n_skip += 1
                continue
            try:
                coords = [float(v) for v in parts[:8]]
            except ValueError:
                continue
            norm = []
            for i, v in enumerate(coords):
                norm.append(min(1.0, max(0.0, v / (w if i % 2 == 0 else h))))
            lines_out.append("0 " + " ".join(f"{v:.6f}" for v in norm))
            n_box += 1
        if lines_out:
            shutil.copy(img_path, img_out / img_path.name)
            (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(lines_out))
            n_img += 1
    print(f"{out_dir}: {n_img} images, {n_box} text instances "
          f"({n_skip} unreadable skipped)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--gts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    convert(a.images, a.gts, a.out)
