"""Rewrite YOLO label indices from one class vocabulary to another, by NAME.

A YOLO .txt records only integers, so a label directory is meaningless without
knowing which vocabulary those integers index. Dropping a class from the middle
of a schema shifts every index above it: removing stairs_down(5) and pole(6)
from AV-14 moves dustbin 7 -> 5 and signboard 8 -> 6. Labels left untouched then
silently mean something else -- every dustbin box becomes a stairs_down box, with
nothing to show for it but a model that confidently announces the wrong thing.
This is the same failure that made the COCO pre-labels dangerous to import.

Source vocabulary comes from <src>/classes.txt (line number == class index, the
convention build_dataset.py and remap_to_av14.py already write). Target is
server.classes_av.AV_CLASSES. Boxes whose class does not exist in the target are
DROPPED, and frames left with no boxes are kept as background negatives.

  python scripts/reindex_labels.py --src datasets/av14_merged \
      --out datasets/av6_merged
"""
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.classes_av import AV_CLASSES  # noqa: E402


def read_classes(path: Path):
    """Line number IS the class index, so a blank line cannot be skipped --
    dropping one shifts every name below it and silently renames those boxes."""
    names = [n.strip() for n in path.read_text(encoding="utf-8").splitlines()]
    while names and not names[-1]:
        names.pop()                      # a trailing newline is not a class
    if not all(names):
        raise SystemExit(
            f"{path} line {names.index('') + 1} is blank, but line number is the "
            "class index -- skipping it would shift every name below it.")
    if not names:
        raise SystemExit(f"{path} is empty")
    return names


def build_map(src_names, dst_names):
    """old index -> new index, matched on name. None means the class is gone."""
    dst_index = {n: i for i, n in enumerate(dst_names)}
    return {i: dst_index.get(n) for i, n in enumerate(src_names)}


def reindex(src: Path, out: Path, dst_names=None):
    dst_names = list(dst_names or AV_CLASSES)
    cls_file = src / "classes.txt"
    if not cls_file.exists():
        raise SystemExit(
            f"no {cls_file}. Refusing to guess which vocabulary these labels "
            "use -- guessing wrong rewrites every box to a different class.")
    src_names = read_classes(cls_file)
    if out.resolve() == src.resolve():
        raise SystemExit("--out must differ from --src (this rewrites labels)")

    mapping = build_map(src_names, dst_names)
    dropped_names = [n for i, n in enumerate(src_names) if mapping[i] is None]

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    kept = Counter()
    dropped = Counter()
    n_frames = n_empty = 0
    for lbl in sorted((src / "labels").glob("*.txt")):
        lines = []
        for line in lbl.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            old = int(parts[0])
            if old >= len(src_names):
                raise SystemExit(
                    f"{lbl} uses class index {old} but {cls_file} defines only "
                    f"{len(src_names)}. These labels are not the vocabulary "
                    "classes.txt claims.")
            new = mapping[old]
            if new is None:
                dropped[src_names[old]] += 1
                continue
            kept[dst_names[new]] += 1
            lines.append(" ".join([str(new)] + parts[1:]))
        # a frame with no boxes is a background negative and must survive
        (out / "labels" / lbl.name).write_text(
            ("\n".join(lines) + "\n") if lines else "")
        n_frames += 1
        n_empty += 0 if lines else 1

    for img in (src / "images").glob("*.jpg"):
        dst = out / "images" / img.name
        if dst.exists():
            continue
        try:
            dst.hardlink_to(img)         # no extra disk, no admin rights
        except OSError:
            shutil.copy2(img, dst)

    (out / "classes.txt").write_text("\n".join(dst_names), encoding="utf-8")

    print(f"vocabulary: {len(src_names)} -> {len(dst_names)} classes")
    if dropped_names:
        print(f"  removed from schema: {', '.join(dropped_names)}")
    print(f"frames: {n_frames} ({n_empty} now empty / background)")
    print(f"boxes kept {sum(kept.values())} | dropped {sum(dropped.values())}")
    for n, c in kept.most_common():
        print(f"   keep  {n:<16} {c}")
    for n, c in dropped.most_common():
        print(f"   DROP  {n:<16} {c}")
    print(f"-> {out}")
    return kept, dropped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    reindex(a.src, a.out)
