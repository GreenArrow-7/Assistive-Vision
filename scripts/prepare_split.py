"""Turn build_dataset.py's flat output into a YOLO train/val split.

Split is BY VIDEO, not by frame: consecutive frames from one walkthrough are
near-identical, so a random frame split leaks the val set into training and
reports a fantasy mAP. Whole videos are held out instead.

data.yaml must name the vocabulary the labels ACTUALLY use. This used to be
hardcoded to the 80-name COCO list, which is right only for raw yolov8s
pre-labels — run it on corrected AV-14 labels and you train a model whose
class names are COCO, and a COCO-class model at models/av_obstacle.pt is the
documented safety failure server/detector.detect_schema exists to catch.

Usage:
  python scripts/prepare_split.py --src datasets/av14_seed --out datasets/av14_seed_split --schema av7
  # COCO baseline set (regenerates what runs/eval/coco_baseline.* was
  # measured on). Kept out of the defaults on purpose: its data.yaml
  # declares COCO names, so training on it yields a COCO model -- exactly
  # the file that must never land at models/av_obstacle.pt.
  python scripts/prepare_split.py --src datasets/av_raw --out datasets/av_finetune
  python scripts/prepare_split.py --src datasets/av14 --out datasets/av14_split \
      --schema av7
"""
import argparse
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.classes_av import AV_CLASSES              # noqa: E402
# same schema names the server branches on, so --schema and detector.py speak
# one vocabulary
from server.detector import SCHEMA_AV, SCHEMA_COCO  # noqa: E402


def video_of(stem: str) -> str:
    """'MallTour_00123' -> 'MallTour' (build_dataset names frames <stem>_<n>)."""
    return re.sub(r"_\d+$", "", stem)


def max_class_index(labels) -> int:
    """Highest class id over all label files, or -1 if every one is empty."""
    hi = -1
    for lbl in labels:
        for line in lbl.read_text().splitlines():
            if line.strip():
                hi = max(hi, int(line.split()[0]))
    return hi


def resolve_schema(requested: str, labels) -> str:
    """Decide which vocabulary the labels speak.

    Auto-detection is honest in ONE direction only. A max class index above 13
    proves the labels are not AV-14 (AV-14 has 14 classes) — that is a real
    signal. The converse is not: AV-14's index range is a strict subset of
    COCO's, so a COCO set that happens to hold only person/bicycle/car/... (ids
    0-13) is indistinguishable from an AV-14 set. Nothing in a YOLO .txt file
    records the vocabulary.

    So we never guess 'av7'. Guessing it is precisely the dangerous guess: it
    stamps AV-14 names onto COCO-trained weights, and server/detector.py
    identifies models by their class NAMES, so the result loads as the AV-14
    fine-tune and silently swaps in AV_HAZARDS — which contains no vehicles.
    Auto therefore concludes 'coco' only on proof, and otherwise refuses and
    asks for --schema. AV-14 must be a human assertion; we still check that
    assertion against the labels.
    """
    hi = max_class_index(labels)
    if requested == SCHEMA_AV:
        if hi >= len(AV_CLASSES):
            raise SystemExit(
                f"--schema av7 but labels contain class index {hi}; the AV schema "
                f"defines only 0-{len(AV_CLASSES) - 1}. These look like uncorrected "
                "COCO pre-labels: remap them before splitting."
            )
        return SCHEMA_AV
    if requested == SCHEMA_COCO:
        return SCHEMA_COCO
    if hi >= len(AV_CLASSES):
        return SCHEMA_COCO
    raise SystemExit(
        f"cannot infer label schema: highest class index is {hi}, which fits "
        f"both AV (0-{len(AV_CLASSES) - 1}) and a COCO subset. Pass "
        "--schema av7 or --schema coco; guessing would put the wrong names "
        "in data.yaml."
    )


def coco_names(src: Path):
    """COCO names in index order.

    build_dataset.py writes the exact vocabulary it pre-labelled with to
    classes.txt next to the labels, so prefer that over downloading yolov8n.pt:
    it costs no network, and it cannot disagree with the indices in the files.

    Line number IS the class index, so a blank line cannot be skipped: dropping
    one shifts every later name down an index and silently renames every box
    above it (potted plant -> tv, and so on) with nothing to show for it but a
    data.yaml that reads plausibly. Refuse instead of guessing which the author
    meant. remap_to_av14.read_coco_names keeps the same invariant.
    """
    txt = src / "classes.txt"
    if txt.exists():
        names = [n.strip() for n in txt.read_text().splitlines()]
        if not all(names):
            raise SystemExit(
                f"{txt} line {names.index('') + 1} is blank, but line number is "
                "the class index — skipping it would shift every name below it."
            )
        return names
    from ultralytics import YOLO
    names = YOLO("yolov8n.pt").names
    return [names[i] for i in sorted(names)]


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)          # hardlink: no extra disk, no admin rights
    except OSError:
        shutil.copy2(src, dst)


# pull_open_datasets.py stamps the upstream Open Images split into every
# filename it writes (oi_tr_*, oi_va_*).
EXTRA_SPLIT_PREFIX = {"oi_tr_": "train", "oi_va_": "val"}


def collect_extra(extra: Path, keep_empty: bool):
    """Gather a source that already carries its own train/val assignment.

    These are honoured rather than re-split. Open Images' train/validation
    split is disjoint by construction, and unlike walkthrough frames these are
    unrelated web photos -- there is no temporal correlation for the by-video
    holdout to protect against, and video_of() would make each one its own
    single-frame "video" anyway, which is what makes the holdout meaningless.
    """
    pairs, n_empty, unknown = [], 0, []
    for img in sorted((extra / "images").glob("*.jpg")):
        split = next((sp for pre, sp in EXTRA_SPLIT_PREFIX.items()
                      if img.name.startswith(pre)), None)
        if split is None:
            unknown.append(img.name)
            continue
        lbl = extra / "labels" / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        if not lbl.read_text().strip():
            n_empty += 1
            if not keep_empty:
                continue
        pairs.append((split, img, lbl))
    if unknown:
        raise SystemExit(
            f"{len(unknown)} file(s) under {extra}/images carry no known split "
            f"prefix ({', '.join(sorted(EXTRA_SPLIT_PREFIX))}): "
            f"{unknown[:3]}. Refusing to guess which side they belong on -- a "
            "wrong guess is train/val leakage that no metric would reveal.")
    if not pairs:
        raise SystemExit(f"no usable image/label pairs under {extra}")
    return pairs, n_empty


def class_histogram(labels):
    """boxes per class id over a set of label files."""
    hist = defaultdict(int)
    for lbl in labels:
        for line in lbl.read_text().splitlines():
            if line.strip():
                hist[int(line.split()[0])] += 1
    return hist


def prepare(src: Path, out: Path, val_frac: float, schema: str = "auto",
            allow_sparse: bool = False, keep_empty: bool = False,
            extra: Path = None):
    pairs, n_empty = [], 0
    for img in sorted((src / "images").glob("*.jpg")):
        lbl = src / "labels" / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        # An empty label file from a pre-labeller asserts "nothing here" on the
        # model's behalf, but all it records is that a COCO detector found none
        # of ITS classes. datasets/av14_seed was 45% empty files, and COCO
        # cannot express stairs, doors, poles or pictogram signs at all -- so
        # those frames were training the model to ignore precisely the classes
        # AV-14 exists for. Dropped unless a human has verified them.
        if not lbl.read_text().strip():
            n_empty += 1
            if not keep_empty:
                continue
        pairs.append((img, lbl))
    if not pairs:
        if n_empty:
            raise SystemExit(
                f"all {n_empty} label files under {src} are empty -- nothing to "
                "train on. An empty pre-label means the detector found none of "
                "ITS classes, not that the frame is empty; pass --keep-empty "
                "only once a human has verified them.")
        raise SystemExit(f"no image/label pairs under {src}")

    extra_pairs, extra_empty = ((), 0) if extra is None else         collect_extra(extra, keep_empty)
    n_empty += extra_empty

    # resolve before copying a single file: a wrong vocabulary is worth failing
    # on early, not after writing a whole split. Extra labels are included so a
    # source in a different vocabulary is caught here, not discovered in mAP.
    schema = resolve_schema(schema, [lbl for _, lbl in pairs]
                            + [lbl for _, _, lbl in extra_pairs])

    by_video = defaultdict(list)
    for img, lbl in pairs:
        by_video[video_of(img.stem)].append((img, lbl))

    # Hold out whole videos, smallest first. Stop at the subset CLOSEST to the
    # target rather than the first one past it: "add until n >= target" always
    # overshoots by up to a whole video, which turned --val-frac 0.2 into an
    # actual 32.8% val split. Whole-video granularity means an exact hit is
    # usually impossible, so aim for nearest and print what was achieved.
    videos = sorted(by_video, key=lambda v: (-len(by_video[v]), v))
    target = len(pairs) * val_frac
    val_videos, n = set(), 0
    for v in reversed(videos):          # prefer smaller videos for val
        take = len(by_video[v])
        if val_videos and abs(n + take - target) >= abs(n - target):
            break                       # adding this one lands further away
        if len(val_videos) + 1 == len(videos):
            break                       # never hold out every video
        val_videos.add(v)
        n += take
    if not val_videos or len(val_videos) == len(videos):
        raise SystemExit("need >=2 videos to split by video")

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (out / split / sub).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for video, items in by_video.items():
        split = "val" if video in val_videos else "train"
        for img, lbl in items:
            link_or_copy(img, out / split / "images" / img.name)
            link_or_copy(lbl, out / split / "labels" / lbl.name)
            counts[split] += 1

    for split, img, lbl in extra_pairs:
        link_or_copy(img, out / split / "images" / img.name)
        link_or_copy(lbl, out / split / "labels" / lbl.name)
        counts[split] += 1

    # server/classes_av.AV_CLASSES is the single source of truth for the schema.
    # scripts/av7.yaml hand-duplicates the same list (it is uploaded to
    # Roboflow/Colab standalone); tests/test_prepare_split.py fails if the two
    # ever drift. classes_av.data_yaml() is not reused here because it emits the
    # train/valid/test layout, and this script writes train/val only.
    names = AV_CLASSES if schema == SCHEMA_AV else coco_names(src)
    yaml_path = out / "data.yaml"
    # No "path" key on purpose. It used to be str(out.resolve()), which baked
    # an absolute Windows path into the file -- unusable on Colab, which is
    # where TRAINING.md says to train. Ultralytics resolves the dataset root as
    #   data.get("path") or Path(data["yaml_file"]).parent
    # so omitting it roots the split at the directory the yaml lives in, and
    # the whole folder can be zipped and unpacked anywhere. Writing "path: ."
    # would NOT work: a relative path resolves against ultralytics' DATASETS_DIR.
    yaml_path.write_text(yaml.safe_dump({
        "train": "train/images",
        "val": "val/images",
        "names": dict(enumerate(names)),
    }, sort_keys=False))

    achieved = counts["val"] / (counts["train"] + counts["val"])
    print(f"schema: {schema} ({len(names)} classes)")
    print(f"videos: {len(videos)} total, {len(val_videos)} held out for val")
    print(f"  val videos: {sorted(val_videos)}")
    print(f"frames: {counts['train']} train / {counts['val']} val "
          f"(val {achieved:.1%}, requested {val_frac:.0%})")
    if n_empty:
        kept = "KEPT as background" if keep_empty else "dropped"
        print(f"empty label files: {n_empty} {kept}"
              + ("" if keep_empty else "  (--keep-empty to retain)"))
    print(f"wrote {yaml_path}")

    # A data.yaml declaring 14 names whose labels only ever use 3 of them trains
    # a model that PASSES detector.detect_schema (its names match AV_CLASSES) yet
    # structurally cannot emit the other 11. That is the same silent-capability
    # gap detect_schema exists to prevent, arriving from the dataset side.
    if schema == SCHEMA_AV:
        # measured on what was WRITTEN, per split -- a class can be present
        # overall yet absent from train, which a combined count hides
        h_train = class_histogram((out / "train" / "labels").glob("*.txt"))
        h_val = class_histogram((out / "val" / "labels").glob("*.txt"))
        empty = [n for i, n in enumerate(names)
                 if not h_train.get(i) and not h_val.get(i)]
        # 0 in train but present in val is strictly WORSE than absent: the model
        # cannot learn the class, yet val still scores it, so it reports 0 AP for
        # a class that never had a chance and drags the mAP down with it.
        starved = [n for i, n in enumerate(names)
                   if h_val.get(i) and not h_train.get(i)]
        problems = []
        if empty:
            problems.append(
                f"{len(empty)} of {len(names)} AV classes have NO boxes: "
                f"{', '.join(empty)}.\nTraining on this yields a model that "
                "cannot detect them, while still identifying itself as the trained schema.")
        if starved:
            problems.append(
                f"{len(starved)} class(es) appear only in val, never in train: "
                f"{', '.join(starved)}.\nThe model cannot learn them yet val "
                "will score them 0 AP. With so few boxes a by-video holdout "
                "cannot place them on both sides -- annotate more of them, or "
                "drop the class from the schema.")
        if problems:
            msg = "\n\n".join(problems)
            if not allow_sparse:
                raise SystemExit(msg + "\n\nPass --allow-sparse to build the "
                                       "split anyway.")
            print("WARNING: " + msg)
    return yaml_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("datasets/av14_seed"))
    ap.add_argument("--out", type=Path, default=Path("datasets/av14_seed_split"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--schema", choices=["auto", SCHEMA_AV, SCHEMA_COCO],
                    default="auto", help="label vocabulary; auto only ever "
                    "concludes coco, see resolve_schema()")
    ap.add_argument("--allow-sparse", action="store_true",
                    help="build an av14 split even if some classes have no boxes")
    ap.add_argument("--extra", type=Path, default=None,
                    help="merge in a source that already encodes its own "
                         "train/val split in each filename, e.g. "
                         "datasets/oi_av14 from pull_open_datasets.py")
    ap.add_argument("--keep-empty", action="store_true",
                    help="keep empty label files as background frames; only do "
                         "this once a human has confirmed they are truly empty")
    a = ap.parse_args()
    prepare(a.src, a.out, a.val_frac, a.schema, a.allow_sparse, a.keep_empty,
            a.extra)
