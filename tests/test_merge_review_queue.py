"""Tests for merge_review_queue: validation, idempotency, background negatives.

The queue is the only path back for boxes remap_to_av14 could not decide, so a
silent drop here loses annotation work with nothing to show it happened.
"""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import merge_review_queue as mrq  # noqa: E402
from server.classes_av import AV_CLASSES  # noqa: E402

COLS = ["image", "coco_class", "suspected_av14", "cx", "cy", "w", "h", "decision"]


def make_seed(tmp_path, frames=("a", "b"), seed_boxes=None):
    seed = tmp_path / "seed"
    (seed / "images").mkdir(parents=True)
    (seed / "labels").mkdir(parents=True)
    for f in frames:
        (seed / "images" / f"{f}.jpg").write_bytes(b"")
        (seed / "labels" / f"{f}.txt").write_text(seed_boxes or "")
    return seed


def make_queue(seed, rows):
    q = seed / "review_queue.csv"
    with q.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return q


def row(image, decision="", cls="tv", cx=0.5, cy=0.5, w=0.2, h=0.2):
    return {"image": image, "coco_class": cls, "suspected_av14": "signboard",
            "cx": cx, "cy": cy, "w": w, "h": h, "decision": decision}


def test_decided_box_is_written_with_correct_class_index(tmp_path):
    seed = make_seed(tmp_path)
    q = make_queue(seed, [row("a.jpg", "signboard")])
    out = tmp_path / "out"

    mrq.merge(seed, out, q)

    line = (out / "labels" / "a.txt").read_text().strip()
    assert line.split()[0] == str(AV_CLASSES.index("signboard"))
    assert [float(x) for x in line.split()[1:]] == [0.5, 0.5, 0.2, 0.2]


def test_drop_decision_discards_the_box(tmp_path):
    seed = make_seed(tmp_path)
    q = make_queue(seed, [row("a.jpg", "drop")])
    mrq.merge(seed, tmp_path / "out", q)
    assert (tmp_path / "out" / "labels" / "a.txt").read_text().strip() == ""


def test_blank_decision_stays_queued(tmp_path):
    seed = make_seed(tmp_path)
    q = make_queue(seed, [row("a.jpg", "")])
    mrq.merge(seed, tmp_path / "out", q)
    # undecided is NOT the same as dropped: no box written, row still in queue
    assert (tmp_path / "out" / "labels" / "a.txt").read_text().strip() == ""
    assert len(list(csv.DictReader(q.open(newline="")))) == 1


def test_unknown_class_is_refused_before_writing(tmp_path):
    seed = make_seed(tmp_path)
    q = make_queue(seed, [row("a.jpg", "signbord")])      # typo
    with pytest.raises(SystemExit, match="not an AV class"):
        mrq.merge(seed, tmp_path / "out", q)
    assert not (tmp_path / "out").exists(), "must not write a partial dataset"


def test_missing_image_is_refused(tmp_path):
    seed = make_seed(tmp_path)
    q = make_queue(seed, [row("ghost.jpg", "signboard")])
    with pytest.raises(SystemExit, match="not found"):
        mrq.merge(seed, tmp_path / "out", q)


def test_merge_is_idempotent(tmp_path):
    seed = make_seed(tmp_path)
    q = make_queue(seed, [row("a.jpg", "signboard")])
    out = tmp_path / "out"

    mrq.merge(seed, out, q)
    first = (out / "labels" / "a.txt").read_text()
    mrq.merge(seed, out, q)
    assert (out / "labels" / "a.txt").read_text() == first


def test_seed_boxes_are_preserved_and_appended_to(tmp_path):
    seed = make_seed(tmp_path, seed_boxes="0 0.1 0.1 0.1 0.1")
    q = make_queue(seed, [row("a.jpg", "signboard")])
    mrq.merge(seed, tmp_path / "out", q)
    lines = (tmp_path / "out" / "labels" / "a.txt").read_text().strip().splitlines()
    assert len(lines) == 2 and lines[0].startswith("0 0.1")


def test_background_negatives_keep_an_empty_label_file(tmp_path):
    seed = make_seed(tmp_path, frames=("a", "b", "c"))
    q = make_queue(seed, [row("a.jpg", "signboard")])
    out = tmp_path / "out"
    mrq.merge(seed, out, q)
    # b and c have no boxes but must stay in the dataset as negatives
    for f in ("b", "c"):
        assert (out / "labels" / f"{f}.txt").exists()
        assert (out / "images" / f"{f}.jpg").exists()


def test_seed_dataset_is_never_modified(tmp_path):
    seed = make_seed(tmp_path, seed_boxes="0 0.1 0.1 0.1 0.1")
    q = make_queue(seed, [row("a.jpg", "signboard")])
    before = (seed / "labels" / "a.txt").read_text()
    mrq.merge(seed, tmp_path / "out", q)
    assert (seed / "labels" / "a.txt").read_text() == before


def test_zero_decisions_is_a_no_op_not_a_wipe(tmp_path):
    """The state the user actually starts in: 219 rows, none decided."""
    seed = make_seed(tmp_path, seed_boxes="0 0.1 0.1 0.1 0.1")
    q = make_queue(seed, [row("a.jpg"), row("b.jpg")])
    out = tmp_path / "out"
    mrq.merge(seed, out, q)
    assert (out / "labels" / "a.txt").read_text().strip() == "0 0.1 0.1 0.1 0.1"
    assert len(list((out / "images").glob("*.jpg"))) == 2
