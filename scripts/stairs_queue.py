"""Re-tag Open Images stairs as stairs_up / stairs_down, in the browser.

Open Images has ONE "Stairs" class, so pull_open_datasets.py writes every
staircase as stairs_up and lists the affected images in review_stairs.txt.
stairs_down is the sole AV_CRITICAL class -- the one that fires "Warning! Stop
and proceed carefully" -- so leaving them all as stairs_up means the highest
-urgency path in the system has no training data at all.

DATA_SOURCING.md said "re-tag the descending ones in Roboflow". That is a full
export/re-import round trip, and re-importing by class INDEX is exactly the
collision this repo has been bitten by. This keeps the re-tag inside the
by-name pipeline instead, and reuses review_crops.py as the viewer:

  python scripts/stairs_queue.py --make          # build the queue
  python scripts/review_crops.py --queue datasets/oi_av14/stairs_queue.csv \\
      --images datasets/oi_av14/images           # press 1 = up, 2 = down
  python scripts/stairs_queue.py --apply         # rewrite the labels

--apply only ever rewrites the CLASS of a box that is already stairs_up; it
never moves a box and never adds one. Decisions are matched on the box
geometry the queue recorded, so a re-run is idempotent.

Then, to actually train the class, un-retire it (done 2026-08-30):
  * move "stairs_down" from AV_RETIRED to AV_CLASSES in server/classes_av.py
  * python scripts/reindex_labels.py --src datasets/oi_av14 --out datasets/oi_av7
  * rebuild the split
Until that edit, reindex_labels DROPS every stairs_down box, because the
trained schema does not contain the class -- which is the honest behaviour, not
a bug.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.classes_av import AV_ALL_CLASSES  # noqa: E402

UP, DOWN = "stairs_up", "stairs_down"
FIELDS = ["image", "coco_class", "suspected_av14", "cx", "cy", "w", "h", "decision"]
DECISIONS = {UP, DOWN, "drop"}


def _index(name: str) -> int:
    return AV_ALL_CLASSES.index(name)


def make(src: Path, out: Path) -> int:
    """One row per stairs_up box in src/labels."""
    up = str(_index(UP))
    rows = []
    for lbl in sorted((src / "labels").glob("*.txt")):
        img = next((p.name for p in (src / "images").glob(f"{lbl.stem}.*")), None)
        if img is None:
            continue
        for line in lbl.read_text().splitlines():
            f = line.split()
            if len(f) == 5 and f[0] == up:
                rows.append({"image": img, "coco_class": "Stairs",
                             "suspected_av14": f"{UP}/{DOWN}",
                             "cx": f[1], "cy": f[2], "w": f[3], "h": f[4],
                             "decision": ""})
    if not rows:
        raise SystemExit(f"no {UP} boxes under {src}/labels -- nothing to review")
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} stairs boxes in {len({r['image'] for r in rows})} images "
          f"-> {out}")
    print("next: python scripts/review_crops.py "
          f"--queue {out} --images {src / 'images'}")
    return len(rows)


def apply(src: Path, queue: Path) -> Counter:
    """Rewrite decided boxes in place. Only stairs_up -> stairs_down/dropped."""
    with queue.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    bad = {d for r in rows if (d := (r.get("decision") or "").strip())
           if d not in DECISIONS}
    if bad:
        raise SystemExit(
            f"queue holds decisions this script cannot apply: {sorted(bad)}. "
            f"Valid: {sorted(DECISIONS)} (or blank to leave the box alone).")

    # geometry -> decision, keyed per image. The label file is the source of
    # truth for what is there; the CSV only says what to call it.
    want = {}
    for r in rows:
        d = (r.get("decision") or "").strip()
        if d:
            key = (Path(r["image"]).stem, r["cx"], r["cy"], r["w"], r["h"])
            want[key] = d

    up, down = str(_index(UP)), str(_index(DOWN))
    n = Counter()
    for lbl in sorted((src / "labels").glob("*.txt")):
        out, changed = [], False
        for line in lbl.read_text().splitlines():
            f = line.split()
            d = want.get((lbl.stem, *f[1:])) if len(f) == 5 and f[0] == up else None
            if d == DOWN:
                out.append(" ".join([down] + f[1:]))
                n[DOWN] += 1
                changed = True
            elif d == "drop":
                n["dropped"] += 1
                changed = True
            else:
                out.append(line)
                if d == UP:
                    n[UP] += 1
        if changed:
            lbl.write_text(("\n".join(out) + "\n") if out else "")
            n["files"] += 1
    undecided = sum(1 for r in rows if not (r.get("decision") or "").strip())
    print(f"{n[DOWN]} boxes -> {DOWN} | {n[UP]} kept {UP} | "
          f"{n['dropped']} dropped | {n['files']} files rewritten | "
          f"{undecided} still undecided")
    if n[DOWN]:
        print(f"\n{DOWN} now has boxes. To TRAIN it, move it from AV_RETIRED to "
              "AV_CLASSES\nin server/classes_av.py, then re-run reindex_labels "
              "and prepare_split.\nUntil then reindex_labels drops these boxes: "
              "the trained schema has no such class.")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=Path("datasets/oi_av14"))
    ap.add_argument("--queue", type=Path, default=None,
                    help="default: <src>/stairs_queue.csv")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--make", action="store_true", help="build the queue CSV")
    g.add_argument("--apply", action="store_true", help="fold decisions back in")
    a = ap.parse_args()
    q = a.queue or a.src / "stairs_queue.csv"
    if a.make:
        make(a.src, q)
    else:
        apply(a.src, q)
