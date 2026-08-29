"""Train a PRELIMINARY AV-7 detector on this CPU box, overnight.

This is NOT the paper model. On an i7-6600U the planned run (yolov8s, 832 px,
120 epochs) extrapolates to ~19 days; scripts/train_av14_colab.py on a free T4
does it overnight. This script exists so there IS a trained model to demo and
to measure against the COCO baseline while the GPU run is arranged.

What it does differently, and why:
  yolov8n not yolov8s   ~3x cheaper; the small model is the only one that
                        finishes here at all
  640 px not 832 px     832 costs 1.7x more per image; signboard (34 boxes)
                        will suffer either way, the volume classes will not
  ~12 epochs not 120    a floor, not a target -- expect low mAP and say so

It writes models/av_obstacle_candidate.pt, NOT models/av_obstacle.pt. The
server loads the latter, and a model this undertrained is very likely WORSE at
`person` than the COCO weights it would replace (COCO has millions of person
boxes; we have 2734). Compare with scripts/evaluate.py before deploying it.

  python scripts/train_local_baseline.py            # ~8 h, backgroundable
  python scripts/train_local_baseline.py --epochs 4 # quicker, weaker
"""
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.classes_av import AV_CLASSES  # noqa: E402


def coverage_gate(split: Path, names):
    """Same check the Colab trainer runs: never spend hours on a split that
    cannot populate a class it declares."""
    hist = {}
    for part in ("train", "val"):
        h = Counter()
        for lbl in (split / part / "labels").glob("*.txt"):
            for line in lbl.read_text().splitlines():
                if line.strip():
                    h[int(line.split()[0])] += 1
        hist[part] = h
    empty = [n for i, n in enumerate(names)
             if not hist["train"].get(i) and not hist["val"].get(i)]
    starved = [n for i, n in enumerate(names)
               if hist["val"].get(i) and not hist["train"].get(i)]
    if empty or starved:
        raise SystemExit(
            f"refusing to train: empty={empty} val_only={starved}. "
            "Fix the split (scripts/prepare_split.py) first.")
    print(f"{'class':<12}{'train':>8}{'val':>7}")
    for i, n in enumerate(names):
        print(f"{n:<12}{hist['train'][i]:>8}{hist['val'][i]:>7}")
    return hist


def main(a):
    split = a.data.resolve()
    yaml_path = split / "data.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"no {yaml_path}; run scripts/prepare_split.py first")
    coverage_gate(split, AV_CLASSES)

    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(
        data=str(yaml_path), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        workers=a.workers, project=str(a.project), name=a.name, exist_ok=True,
        patience=a.patience, plots=False, verbose=False,
        # a small run cannot afford heavy augmentation: it spends the few
        # epochs available on distorted frames instead of the real domain
        mosaic=0.5, close_mosaic=2, degrees=5, hsv_v=0.4, fliplr=0.5,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"training produced no weights at {best}")

    trained = YOLO(str(best))
    names = {str(v) for v in trained.names.values()}
    if names != set(AV_CLASSES):
        raise SystemExit(
            f"class-name mismatch, the server would refuse these weights: "
            f"{sorted(names)}")

    metrics = trained.val(data=str(yaml_path), imgsz=a.imgsz, plots=False)
    print(f"\nmAP50 {metrics.box.map50:.4f} | mAP50-95 {metrics.box.map:.4f}")
    for i, c in enumerate(metrics.box.ap_class_index):
        print(f"  {AV_CLASSES[int(c)]:<12} AP50 {metrics.box.ap50[i]:.4f}")

    out = Path("models/av_obstacle_candidate.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    print(f"\nwrote {out}")
    print("NOT deployed on purpose: models/av_obstacle.pt is what the server "
          "loads.\nCompare against the COCO baseline first:")
    print("  python scripts/evaluate.py --weights models/av_obstacle_candidate.pt "
          f"--data {yaml_path} --tag local_baseline")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("datasets/av7_split"))
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)   # 2 cores: workers hurt
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--project", type=Path, default=Path("runs/local"))
    ap.add_argument("--name", default="av7_baseline")
    main(ap.parse_args())
