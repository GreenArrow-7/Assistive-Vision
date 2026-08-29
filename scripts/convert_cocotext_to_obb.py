"""Convert COCO-Text v2 annotations to YOLOv8-OBB format — Stage 2 data
(real photos, complex scenes, mostly small/incidental text).

Needs:
  * cocotext.v2.json  — https://bgshih.github.io/cocotext/
  * COCO 2014 train images (train2014/) — https://cocodataset.org/#download
    (COCO-Text annotates a subset of COCO train2014)

Polygons ("mask") are rotated to a min-area rectangle -> 4-point OBB.
Illegible instances are skipped (same convention as ICDAR "###").

Usage:
  pip install opencv-python numpy
  python scripts/convert_cocotext_to_obb.py \
      --json data/cocotext.v2.json --images data/train2014 \
      --out datasets/cocotext_obb
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def convert(json_path: Path, images: Path, out: Path):
    data = json.loads(json_path.read_text())
    imgs, anns = data["imgs"], data["anns"]

    by_img = {}
    for a in anns.values():
        by_img.setdefault(str(a["image_id"]), []).append(a)

    counts = {"train": 0, "val": 0}
    n_box = n_skip = 0
    for img_id, img in imgs.items():
        split = "train" if img["set"] == "train" else "val"
        alist = by_img.get(img_id)
        if not alist:
            continue
        src = images / img["file_name"]
        if not src.exists():
            continue
        w, h = img["width"], img["height"]
        lines = []
        for a in alist:
            if a.get("legibility") != "legible":
                n_skip += 1
                continue
            poly = a.get("mask") or []
            if len(poly) >= 6:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                box = cv2.boxPoints(cv2.minAreaRect(pts))   # (4,2) rotated
            else:                                           # fallback: bbox
                x, y, bw, bh = a["bbox"]
                box = np.array([[x, y], [x + bw, y],
                                [x + bw, y + bh], [x, y + bh]], dtype=np.float32)
            box[:, 0] = np.clip(box[:, 0] / w, 0, 1)
            box[:, 1] = np.clip(box[:, 1] / h, 0, 1)
            lines.append("0 " + " ".join(f"{v:.6f}" for v in box.flatten()))
            n_box += 1
        if not lines:
            continue
        img_out = out / split / "images"
        lbl_out = out / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, img_out / img["file_name"])
        (lbl_out / f"{src.stem}.txt").write_text("\n".join(lines))
        counts[split] += 1
    print(f"DONE: {counts['train']} train / {counts['val']} val images, "
          f"{n_box} boxes ({n_skip} illegible skipped) -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path,
                    help="COCO train2014 image directory")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    convert(a.json, a.images, a.out)
