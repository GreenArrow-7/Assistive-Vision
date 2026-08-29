"""Build a fine-tuning dataset from walkthrough videos.

Pipeline:
  1. Extract frames at --fps (default 1/s)
  2. Drop blurry frames (variance of Laplacian) and near-duplicates (aHash)
  3. Pre-label every kept frame with the current YOLO model -> YOLO-format
     .txt labels you IMPORT into Roboflow/CVAT and merely correct

Usage:
  python scripts/build_dataset.py --videos college.mp4 hospital.mp4 mall.mp4 \
      --out datasets/av_raw --fps 1

Then: upload datasets/av_raw/images + labels to Roboflow (YOLOv8 format),
fix wrong boxes, add missing classes (stairs, door, pole, signs),
export and train:
  yolo detect train model=yolov8s.pt data=<roboflow yaml> epochs=100 imgsz=640
"""
import argparse
import re
import unicodedata
from pathlib import Path

import cv2
import numpy as np

BLUR_MIN = 60.0          # variance of Laplacian below this = discard
HASH_DIST_MIN = 6        # aHash Hamming distance below this = duplicate
PRELABEL_CONF = 0.35     # loose on purpose; you delete extras in Roboflow


def ahash(gray):
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).flatten()


def slug(name: str) -> str:
    """ASCII-safe filename stem (YouTube titles carry emoji / fullwidth chars)."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", plain).strip("._-") or "video"


def imwrite(path: Path, img) -> None:
    """cv2.imwrite goes through the ANSI API on Windows: for a non-ASCII path it
    writes NOTHING and still returns True. Encode in memory, write via pathlib."""
    ok, buf = cv2.imencode(path.suffix, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError(f"JPEG encode failed for {path}")
    path.write_bytes(buf.tobytes())


def process(videos, out: Path, fps: float, prelabel: bool):
    img_dir = out / "images"
    lbl_dir = out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if prelabel:
        from ultralytics import YOLO
        model = YOLO("yolov8s.pt")
        names = model.names
        (out / "classes.txt").write_text(
            "\n".join(names[i] for i in sorted(names)))

    prev_hash = None
    kept = dropped_blur = dropped_dup = 0

    for vid in videos:
        cap = cv2.VideoCapture(str(vid))
        native = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(round(native / fps)))
        idx = 0
        stem = slug(Path(vid).stem)
        # stem, not vid: YouTube titles carry emoji, and a cp1252 console dies on them
        print(f"\n{stem}: {native:.0f} fps native, sampling every {step} frames")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step:
                idx += 1
                continue
            idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_MIN:
                dropped_blur += 1
                continue
            h = ahash(gray)
            if prev_hash is not None and int((h != prev_hash).sum()) < HASH_DIST_MIN:
                dropped_dup += 1
                continue
            prev_hash = h

            name = f"{stem}_{kept:05d}"
            imwrite(img_dir / f"{name}.jpg", frame)

            if model is not None:
                H, W = frame.shape[:2]
                res = model.predict(frame, conf=PRELABEL_CONF, verbose=False)[0]
                lines = []
                if res.boxes is not None:
                    for xyxy, cls in zip(res.boxes.xyxy.cpu().numpy(),
                                         res.boxes.cls.cpu().numpy()):
                        x1, y1, x2, y2 = xyxy
                        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                        bw, bh = (x2 - x1) / W, (y2 - y1) / H
                        lines.append(
                            f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                (lbl_dir / f"{name}.txt").write_text("\n".join(lines))
            kept += 1
            if kept % 100 == 0:
                print(f"  kept {kept}…")
        cap.release()

    print(f"\nDONE: {kept} frames kept | {dropped_blur} blurry dropped | "
          f"{dropped_dup} duplicates dropped\n-> {out}")
    print("Next: upload images/+labels/ to Roboflow (YOLOv8 format), correct, "
          "add classes stairs/door/pole/signs, export, train.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("datasets/av_raw"))
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--no-prelabel", action="store_true")
    a = ap.parse_args()
    process(a.videos, a.out, a.fps, not a.no_prelabel)
