"""Pull a ready-labeled booster set from Open Images V7 and convert it to the
AV-14 schema — thousands of labeled persons/chairs/tables/doors/stairs/dustbins
for free, no annotation needed.

Run ON YOUR LAPTOP or in Colab (needs internet, ~2-6 GB disk):

  pip install fiftyone ultralytics
  python scripts/pull_open_datasets.py --per-class 400 --out datasets/oi_av14

Output: datasets/oi_av14/images + labels in YOLO format with AV-14 class ids,
ready to upload to Roboflow ALONGSIDE your video frames (Roboflow merges them).

What it can and cannot give you:
  COVERED  : person, chair, table, door, stairs*, dustbin (Waste container)
  NOT HERE : pole, signboard, the 5 sign_* classes  -> those come from your
             video frames and the Roboflow Universe sets in DATA_SOURCING.md.
  *stairs caveat: Open Images has one "Stairs" class. This script imports them
   as stairs_up (id 4) and writes review_stairs.txt listing those images —
   in Roboflow, re-tag the descending ones as stairs_down. Do not skip this:
   stairs_down is the class the whole safety story depends on.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.classes_av import AV_ALL_CLASSES  # noqa: E402  single source of truth

# Open Images class -> AV class NAME. Everything in this map gets WRITTEN when
# it appears in a downloaded image.
#
# Names, never hardcoded integers. This map used to say "Waste container": 7,
# which was dustbin's index at the time; reordering AV_ALL_CLASSES so the seven
# trained classes hold 0-6 moved dustbin to 5 and put stairs_down at 7. The
# stale literal would have labelled a thousand bins as stairs_down -- the sole
# AV_CRITICAL class -- with nothing failing to announce it. Deriving the index
# from the name means a reorder can only ever raise, never mislabel.
OI_TO_AV_NAME = {
    "Person": "person",
    "Chair": "chair",
    "Table": "table",
    "Door": "door",
    "Stairs": "stairs_up",        # manually split later, see above
    "Waste container": "dustbin",
}

for _n in OI_TO_AV_NAME.values():
    if _n not in AV_ALL_CLASSES:
        raise SystemExit(f"{_n!r} is not in AV_ALL_CLASSES -- schema changed, "
                         "fix this table")

OI_TO_AV14 = {oi: AV_ALL_CLASSES.index(av) for oi, av in OI_TO_AV_NAME.items()}

# ...but only these decide WHICH images to download. The seed set already has
# 2512 person / 183 chair / 28 table boxes; selecting on Person too would spend
# the whole --per-class budget on the one class that needs nothing, since Person
# is orders of magnitude more common in Open Images than Stairs.
# Person/Chair/Table stay in the map above so that when they co-occur in a
# door/stairs/bin image they are still labelled — an unlabelled person in a
# training frame teaches the model to miss people.
SELECT_CLASSES = ["Door", "Stairs", "Waste container"]


def main(per_class: int, out: Path, splits=("train", "validation"),
         select=None):
    import fiftyone as fo
    import fiftyone.zoo as foz

    img_dir = out / "images"
    lbl_dir = out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # One load per class, so --per-class means per class. A single load with
    # classes=[...] and max_samples=N*len(classes) returns images matching ANY
    # of them, and Open Images has vastly more doors than staircases — the rare
    # classes we are here for would arrive as a rounding error.
    for split in splits:
      for oi_class in (select or SELECT_CLASSES):
        print(f"--- {split} / {oi_class} (up to {per_class}) ---")
        ds = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=[oi_class],
            max_samples=per_class,
            shuffle=True,
        )
        for sample in ds.iter_samples(progress=True):
            # fiftyone puts Open Images detections on 'ground_truth', not
            # 'detections' -- the latter raises AttributeError on every sample.
            gt = sample.get_field("ground_truth")
            dets = [d for d in (gt.detections if gt else [])
                    if d.label in OI_TO_AV14]
            if not dets:
                continue
            src = Path(sample.filepath)
            dst = img_dir / f"oi_{split[:2]}_{src.name}"
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            lines = []
            for d in dets:
                # FiftyOne boxes are [x_tl, y_tl, w, h] relative -> YOLO cxcywh
                x, y, w, h = d.bounding_box
                lines.append(f"{OI_TO_AV14[d.label]} "
                             f"{x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
            (lbl_dir / f"{dst.stem}.txt").write_text("\n".join(lines))
        fo.delete_dataset(ds.name)

    # Record the vocabulary these integers index. A YOLO .txt is meaningless
    # without it, and reindex_labels.py reads exactly this file; without it the
    # only thing saying "7 means dustbin" was whoever remembered. Line number
    # == class index, the convention build_dataset.py already writes.
    (out / "classes.txt").write_text("\n".join(AV_ALL_CLASSES))

    # Recomputed from the labels on disk, not accumulated in memory: this
    # script is now run one class (and one split) per process to keep peak
    # memory down, and an in-memory list would clobber earlier runs' entries.
    stairs_id = OI_TO_AV14["Stairs"]
    names = {q.stem: q.name for q in img_dir.iterdir()}
    stairs = sorted(
        names[q.stem] for q in lbl_dir.glob("*.txt") if q.stem in names
        and any(ln.split()[0] == str(stairs_id)
                for ln in q.read_text().splitlines() if ln.strip()))
    (out / "review_stairs.txt").write_text("\n".join(stairs))
    print(f"Done -> {out}")
    print(f"{len(stairs)} images contain stairs: re-tag descending ones "
          "as stairs_down in Roboflow (list: review_stairs.txt)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=400,
                    help="target samples per Open Images class per split")
    ap.add_argument("--out", type=Path, default=Path("datasets/oi_av14"))
    # Open Images' train metadata alone is a ~640 MB CSV; a dropped connection
    # mid-download leaves a truncated file that fiftyone treats as complete on
    # the next run (it only tests existence). Being able to run one split at a
    # time makes a failure cheap to retry -- start with validation.
    ap.add_argument("--splits", nargs="+", default=["train", "validation"],
                    choices=["train", "validation", "test"])
    # Open Images' train index is big enough that loading three classes in one
    # process peaked over 3 GB and segfaulted (exit 139) partway through the
    # second class. One class per process lets the OS reclaim it on exit.
    ap.add_argument("--classes", nargs="+", default=None, choices=SELECT_CLASSES,
                    metavar="OI_CLASS",
                    help="Open Images classes to select on (default: all three)")
    a = ap.parse_args()
    main(a.per_class, a.out, tuple(a.splits), a.classes)
