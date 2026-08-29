"""Convert SynthText ground truth (gt.mat) to YOLOv8-OBB format — Stage 1
pretraining data for the oriented text detector.

SynthText = 858k synthetic images (~41 GB) with word-level oriented boxes.
You do NOT need all of it: 30-60k images is plenty for pretraining on a free
Colab T4. Use --max-images.

Download: https://academictorrents.com/details/2dba9518166cbd141534cbf381aa3e99a087e83c
(or the official VGG link) -> extract to a folder containing gt.mat + image dirs.

Usage:
  pip install scipy pillow numpy
  python scripts/convert_synthtext_to_obb.py \
      --root data/SynthText --out datasets/synthtext_obb --max-images 50000
"""
import argparse
import random
import shutil
from pathlib import Path

import numpy as np


def convert(root: Path, out: Path, max_images: int, seed: int = 0):
    from scipy.io import loadmat

    print("Loading gt.mat (takes a minute)…")
    gt = loadmat(str(root / "gt.mat"), squeeze_me=True,
                 variable_names=["imnames", "wordBB"])
    imnames = np.atleast_1d(np.asarray(gt["imnames"], dtype=object))
    wordbbs = gt["wordBB"]
    if not (isinstance(wordbbs, np.ndarray) and wordbbs.dtype == object):
        tmp = np.empty(1, dtype=object)      # single-image mat gets squeezed
        tmp[0] = wordbbs
        wordbbs = tmp

    idxs = list(range(len(imnames)))
    random.Random(seed).shuffle(idxs)
    if max_images:
        idxs = idxs[:max_images]

    img_out = out / "train" / "images"
    lbl_out = out / "train" / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    n_img = n_box = 0
    for i in idxs:
        rel = str(imnames[i])
        src = root / rel
        if not src.exists():
            continue
        bb = np.array(wordbbs[i], dtype=float)     # (2,4) or (2,4,N)
        if bb.ndim == 2:
            bb = bb[:, :, None]
        try:
            w, h = Image.open(src).size
        except OSError:
            continue
        lines = []
        for k in range(bb.shape[2]):
            xs = np.clip(bb[0, :, k] / w, 0, 1)
            ys = np.clip(bb[1, :, k] / h, 0, 1)
            if xs.max() - xs.min() < 1e-3 or ys.max() - ys.min() < 1e-3:
                continue                            # degenerate box
            pts = np.empty(8)
            pts[0::2], pts[1::2] = xs, ys
            lines.append("0 " + " ".join(f"{v:.6f}" for v in pts))
            n_box += 1
        if not lines:
            continue
        name = rel.replace("/", "_")
        shutil.copy(src, img_out / name)
        (lbl_out / f"{Path(name).stem}.txt").write_text("\n".join(lines))
        n_img += 1
        if n_img % 5000 == 0:
            print(f"  {n_img} images…")
    print(f"DONE: {n_img} images, {n_box} word boxes -> {out}")
    print("Use ~2k of these as val, or point val at ICDAR/COCO-Text instead.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="folder containing gt.mat and the SynthText image dirs")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-images", type=int, default=50000)
    a = ap.parse_args()
    convert(a.root, a.out, a.max_images)
