"""Fold adjudicated review_queue.csv decisions back into a labelled dataset.

remap_to_av14.py auto-maps what it can and parks the ambiguous boxes in
review_queue.csv (e.g. COCO "tv": some are mall signboards that should feed OCR,
some are ad screens and actual televisions). Nothing read that file back, so the
219 parked boxes had no way home.

Workflow:
  1. python scripts/merge_review_queue.py --report          # what is left
  2. python scripts/review_crops.py                # browser, one key per row
     ...or edit the `decision` column by hand:
       an AV-14 class name  -> keep the box as that class
       drop                 -> discard the box
       (blank)              -> not yet decided; stays queued
     Hand-editing is the slow path: the crop is what decides these rows and a
     CSV editor cannot show it.
  3. python scripts/merge_review_queue.py --out datasets/av14_merged

Writes a NEW directory; the seed and raw datasets are never modified.
"""
import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.classes_av import AV_ALL_CLASSES  # noqa: E402

DECISION = "decision"
DROP = "drop"


def read_queue(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows and DECISION not in rows[0]:
        # first run: add the column so the file is ready to edit
        for r in rows:
            r[DECISION] = ""
        write_queue(path, rows)
        print(f"added a '{DECISION}' column to {path} — fill it in, then re-run")
    return rows


def write_queue(path: Path, rows):
    cols = ["image", "coco_class", "suspected_av14", "cx", "cy", "w", "h", DECISION]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)


def validate(rows, images_dir: Path):
    """Reject bad decisions before writing anything."""
    # the annotation vocabulary, not the trained one: a verdict of
    # stairs_down is valuable even while that class is untrained
    valid = set(AV_ALL_CLASSES) | {DROP}
    errors = []
    for i, r in enumerate(rows, start=2):          # +2: header is line 1
        d = (r.get(DECISION) or "").strip()
        if not d:
            continue
        if d not in valid:
            errors.append(f"line {i}: '{d}' is not an AV class or '{DROP}'")
        if not (images_dir / r["image"]).exists():
            errors.append(f"line {i}: image {r['image']} not found")
    return errors


def merge(seed: Path, out: Path, queue: Path):
    images_dir = seed / "images"
    rows = read_queue(queue)
    errors = validate(rows, images_dir)
    if errors:
        raise SystemExit("refusing to merge:\n  " + "\n  ".join(errors[:20]))

    decided = [r for r in rows if (r.get(DECISION) or "").strip()]
    keep = [r for r in decided if r[DECISION].strip() != DROP]

    if out.resolve() in (seed.resolve(), (seed / "..").resolve()):
        raise SystemExit(f"--out {out} would overwrite the seed dataset")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    # copy the seed across untouched, then append the adjudicated boxes
    for img in (seed / "images").glob("*.jpg"):
        dst = out / "images" / img.name
        if not dst.exists():
            try:
                dst.hardlink_to(img)
            except OSError:
                shutil.copy2(img, dst)
    for lbl in (seed / "labels").glob("*.txt"):
        shutil.copy2(lbl, out / "labels" / lbl.name)

    added = Counter()
    for r in keep:
        cls = AV_ALL_CLASSES.index(r[DECISION].strip())
        line = (f"{cls} {float(r['cx']):.6f} {float(r['cy']):.6f} "
                f"{float(r['w']):.6f} {float(r['h']):.6f}")
        lbl = out / "labels" / (Path(r["image"]).stem + ".txt")
        existing = lbl.read_text().splitlines() if lbl.exists() else []
        if line in existing:
            continue                    # idempotent: re-running adds nothing
        existing.append(line)
        lbl.write_text("\n".join(x for x in existing if x.strip()) + "\n")
        added[r[DECISION].strip()] += 1

    # a frame with no boxes is a background negative and must survive
    for img in (out / "images").glob("*.jpg"):
        lbl = out / "labels" / (img.stem + ".txt")
        if not lbl.exists():
            lbl.write_text("")

    print(f"queued {len(rows)} | decided {len(decided)} | "
          f"undecided {len(rows) - len(decided)}")
    print(f"boxes added {sum(added.values())} | dropped {len(decided) - len(keep)}")
    for k, v in added.most_common():
        print(f"   {k:<16} {v}")
    print(f"-> {out}")


def report(queue: Path):
    rows = read_queue(queue)
    done = Counter()
    todo = Counter()
    for r in rows:
        d = (r.get(DECISION) or "").strip()
        (done if d else todo)[r["coco_class"]] += 1
    total = len(rows)
    n_done = sum(done.values())
    print(f"review queue: {n_done}/{total} adjudicated "
          f"({n_done / total * 100:.0f}%)" if total else "queue is empty")
    if todo:
        print("\noutstanding, by source class:")
        for k, v in todo.most_common():
            print(f"   {k:<16} {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=Path, default=Path("datasets/av14_seed"))
    ap.add_argument("--out", type=Path, default=Path("datasets/av14_merged"))
    ap.add_argument("--queue", type=Path,
                    help="default: <seed>/review_queue.csv")
    ap.add_argument("--report", action="store_true",
                    help="show progress without writing anything")
    a = ap.parse_args()
    q = a.queue or a.seed / "review_queue.csv"
    if not q.exists():
        raise SystemExit(f"no review queue at {q}")
    report(q) if a.report else merge(a.seed, a.out, q)
