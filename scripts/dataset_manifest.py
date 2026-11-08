"""Manifest a YOLO dataset and check its splits for leakage, by source.

File hashes are not enough when the images are video frames. Frame 00699 and
frame 00700 of the same walkthrough are different bytes and near-identical
content, so hashing says "no leakage" while the model is effectively validated
on its training data. This script groups images by the source they came from and
reports any source appearing on both sides of a split.

Source is read from the filename prefix the ingest pipeline writes:
`<videoid>_<title>_<frame>.jpg` for video frames, `oi_<split>_<imageid>.jpg` for
Open Images stills (each of which is its own source -- they are unrelated
photographs, not frames of one recording).

    python scripts/dataset_manifest.py datasets/av7_split --id assistive_av7 --version 1
"""
import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

FRAME = re.compile(r"^(?P<src>.+?)_(?P<frame>\d{4,6})$")


def source_of(stem):
    """The recording or photograph this image came from."""
    if stem.startswith("oi_"):
        return stem           # an independent still; its own source
    m = FRAME.match(stem)
    return m.group("src") if m else stem


def digest(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(root, split):
    images = os.path.join(root, split, "images")
    labels = os.path.join(root, split, "labels")
    out = []
    for name in sorted(os.listdir(images)):
        stem = os.path.splitext(name)[0]
        lab = os.path.join(labels, stem + ".txt")
        boxes = Counter()
        if os.path.exists(lab):
            for line in open(lab):
                if line.split():
                    boxes[int(line.split()[0])] += 1
        out.append({"file": name, "source": source_of(stem),
                    "image_md5": digest(os.path.join(images, name)),
                    "label_md5": digest(lab) if os.path.exists(lab) else None,
                    "boxes": dict(sorted(boxes.items()))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--id", required=True)
    ap.add_argument("--version", default="1")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    splits = {s: scan(a.root, s) for s in a.splits.split(",")}
    by_source = {s: {r["source"] for r in rows} for s, rows in splits.items()}
    by_hash = {s: {r["image_md5"] for r in rows} for s, rows in splits.items()}

    names = list(splits)
    leaks = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            shared_src = sorted(by_source[x] & by_source[y])
            shared_img = sorted(by_hash[x] & by_hash[y])
            if shared_src or shared_img:
                leaks.append({"between": [x, y],
                              "shared_sources": shared_src,
                              "identical_images": len(shared_img)})

    manifest = {
        "dataset_id": a.id,
        "version": a.version,
        "root": a.root,
        "taxonomy": open(os.path.join(a.root, "data.yaml")).read(),
        "splits": {s: {"images": len(rows),
                       "sources": len(by_source[s]),
                       "boxes_per_class": dict(sorted(
                           Counter(k for r in rows for k, v in r["boxes"].items()
                                   for _ in range(v)).items()))}
                   for s, rows in splits.items()},
        "leakage_policy": "splits are isolated by SOURCE RECORDING, not by file hash",
        "leakage": leaks or "none",
        "files": {s: rows for s, rows in splits.items()},
    }
    path = a.out or os.path.join(a.root, "MANIFEST.json")
    json.dump(manifest, open(path, "w"), indent=1)

    for s in names:
        d = manifest["splits"][s]
        print(f"{s:6} {d['images']:5} images  {d['sources']:4} sources  boxes {d['boxes_per_class']}")
    print("leakage:", manifest["leakage"])
    print("wrote", path)
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
