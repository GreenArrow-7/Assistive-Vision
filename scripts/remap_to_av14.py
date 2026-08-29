"""Rewrite datasets/av_raw's COCO pre-labels into AV-14 indices, into a NEW dir.

This REDUCES annotation work. It does NOT replace annotation. COCO has no
equivalent for stairs_up, stairs_down, door, signboard or any of the five
sign_* classes, so every box for those classes must still be drawn by hand --
that gap IS the reason AV-14 exists as its own schema. What this script buys
you is that the ~2700 person/chair/table boxes arrive pre-drawn and correctly
indexed, and that the ~1000 junk boxes (planters, cars, floor-tile "beds")
are gone before a human ever sees them.

Three verdicts, from a review of every COCO class present in the set:
  AUTO   -> class index rewritten to AV-14, box kept as drawn.
  MANUAL -> box NOT written to any label; listed in review_queue.csv instead,
            because the COCO class maps to two or more AV-14 classes (or to
            none) and only a human looking at the crop can decide.
  DROP   -> everything else. Not enumerated here on purpose: an unrecognised
            or newly-appearing COCO class must fall through to "discard",
            never to a guessed AV-14 index.

datasets/av_raw is opened read-only; images are hardlinked, never moved.

Usage:
  python scripts/remap_to_av14.py --src datasets/av_raw --out datasets/av14_seed
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent),            # scripts/
                str(Path(__file__).resolve().parents[1])]        # repo root

from prepare_split import link_or_copy      # noqa: E402  same hardlink policy
from server.classes_av import AV_ALL_CLASSES as AV_CLASSES    # noqa: E402  single source of truth

# COCO name -> AV-14 name. Panel verdict: person/table high confidence,
# chair medium (~4% junk, tolerable, and it is the only chair signal we have).
AUTO = {
    "person": "person",
    "chair": "chair",
    "dining table": "table",
}

# COCO name -> the AV-14 classes a human should choose between. Multi-valued
# entries are why these cannot be auto-mapped; single-valued ones (vase,
# cell phone, stop sign) are queued because the class they would poison has
# almost no other training data, so a 20% error rate would define it wrong.
MANUAL = {
    "tv": ("signboard",),                          # store signs vs ad panels vs real TVs
    "refrigerator": ("signboard", "door", "pole"),  # lift doors, notice boards, pillars
    "vase": ("dustbin",),                          # 11/14 bins, 3 planters/balloon
    "bench": ("chair", "table", "stairs_up"),      # only pre-drawn stairs in the set
    "cell phone": ("sign_lift",),                  # 2/6 are a lift call panel
    "stop sign": ("signboard",),                   # a "50% SALE" standee
    "parking meter": ("dustbin", "sign_lift"),
}

# Fail loudly if the schema is renamed under us rather than writing a wrong index.
for _name in [*AUTO.values(), *(c for t in MANUAL.values() for c in t)]:
    if _name not in AV_CLASSES:   # AV_ALL_CLASSES: annotation vocabulary
        raise SystemExit(f"{_name!r} is not in AV_CLASSES -- schema changed, fix this table")


def read_coco_names(src: Path) -> dict:
    """classes.txt written by build_dataset.py: line number == COCO index.

    Blank lines are skipped but still consume an index, because the index is
    what the label files reference. Names are stripped: a stray trailing space
    would miss the AUTO/MANUAL lookup and silently DROP every box of that class.
    """
    path = src / "classes.txt"
    if not path.exists():
        raise SystemExit(f"missing {path} -- needed to resolve COCO indices")
    return {i: n.strip() for i, n in enumerate(path.read_text().split("\n")) if n.strip()}


def refuse_to_clobber(out: Path, force: bool) -> None:
    """Never overwrite label files this script did not write.

    Labels are rewritten unconditionally while images are skipped if present, so
    a second run silently erases every box an annotator drew -- and the workflow
    is precisely 'run this, then annotate the output'. The same guard covers the
    forbidden typo `--out datasets/av_raw`, which would overwrite the original
    COCO pre-labels with AV-14 indices.
    """
    if force:
        return
    labelled = [p for p in (out / "labels").glob("*.txt") if p.read_text().strip()]
    if labelled:
        raise SystemExit(
            f"{out / 'labels'} already holds {len(labelled)} non-empty label files. "
            "Re-running would overwrite them (annotation work, or the av_raw "
            "originals). Point --out somewhere new, or pass --force if you are "
            "certain those labels are a previous run's output and nothing else."
        )


def remap(src: Path, out: Path, force: bool = False):
    coco = read_coco_names(src)
    av_index = {n: i for i, n in enumerate(AV_CLASSES)}

    refuse_to_clobber(out, force)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    auto, queued, dropped = Counter(), Counter(), Counter()
    queue_rows, negatives, held_back = [], 0, 0

    images = sorted((src / "images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images under {src / 'images'}")

    for img in images:
        lines, pending = [], 0
        lbl = src / "labels" / f"{img.stem}.txt"
        for line in lbl.read_text().splitlines() if lbl.exists() else []:
            parts = line.split()
            if len(parts) != 5:
                continue
            name = coco.get(int(parts[0]), "?")
            if name in AUTO:
                lines.append(" ".join([str(av_index[AUTO[name]]), *parts[1:]]))
                auto[name] += 1
            elif name in MANUAL:
                queued[name] += 1
                pending += 1
                queue_rows.append([img.name, name, "/".join(MANUAL[name]), *parts[1:]])
            else:
                dropped[name] += 1

        link_or_copy(img, out / "images" / img.name)
        if lines or not pending:
            # An empty label file is a POSITIVE assertion: "this frame contains
            # nothing" -- YOLO trains on it as a background negative, and needs
            # the file present to do so.
            (out / "labels" / f"{img.stem}.txt").write_text("\n".join(lines))
            negatives += not lines
        else:
            # ...which is a lie for a frame whose every box went to the review
            # queue: it holds a real object nobody has adjudicated yet. Calling
            # it background teaches the model "signboard region == background",
            # on exactly the classes with the least data. Omit the label file;
            # prepare_split.py pairs on label existence, so the frame is excluded
            # from the split until an annotator saves one.
            held_back += 1
            # --force over an older output would otherwise leave that run's empty
            # label file in place, resurrecting the false negative we just avoided.
            (out / "labels" / f"{img.stem}.txt").unlink(missing_ok=True)

    (out / "classes.txt").write_text("\n".join(AV_CLASSES))
    queue_path = out / "review_queue.csv"
    with queue_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "coco_class", "suspected_av14", "cx", "cy", "w", "h"])
        w.writerows(queue_rows)

    print(f"images: {len(images)} copied, {negatives} kept as background negatives "
          f"(empty label on purpose)")
    if held_back:
        print(f"  {held_back} frames have NO label file: every box in them is in the "
              "review queue,\n  so an empty label would train them as background. "
              "Annotate them and they rejoin the split.")
    print(f"boxes:  {sum(auto.values())} auto-mapped | {sum(queued.values())} queued "
          f"| {sum(dropped.values())} dropped")
    for label, c in (("auto", auto), ("queued", queued)):
        for name, n in c.most_common():
            tgt = AUTO[name] if label == "auto" else "/".join(MANUAL[name])
            print(f"  {label:6} {name:14} {n:5} -> {tgt}")
    print(f"wrote {queue_path} ({len(queue_rows)} boxes to adjudicate)")

    hand = [c for c in AV_CLASSES if c not in AUTO.values()]
    print(f"\nSTILL TO DRAW BY HAND ({len(hand)} of {len(AV_CLASSES)} classes): "
          f"{', '.join(hand)}")
    print("No COCO equivalent exists for these. This script is a head start, "
          "not a substitute for annotation.")
    return queue_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=Path("datasets/av_raw"))
    ap.add_argument("--out", type=Path, default=Path("datasets/av14_seed"))
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing label files in --out (destroys any "
                         "annotation already drawn there)")
    a = ap.parse_args()
    remap(a.src, a.out, a.force)
