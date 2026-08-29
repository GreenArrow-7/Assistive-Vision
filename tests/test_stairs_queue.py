"""stairs_queue.py: the stairs_up -> stairs_down re-tag must touch nothing else.

stairs_down is the sole AV_CRITICAL class, so a re-tag that moves a box, drops
a neighbouring class, or double-applies is worse than no re-tag at all.
"""
import csv
import sys
from pathlib import Path

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts"),
                str(Path(__file__).resolve().parents[1])]

import stairs_queue as sq  # noqa: E402
from server.classes_av import AV_ALL_CLASSES  # noqa: E402

UP, DOWN = str(AV_ALL_CLASSES.index("stairs_up")), str(AV_ALL_CLASSES.index("stairs_down"))
PERSON = str(AV_ALL_CLASSES.index("person"))
BOX_A = "0.500000 0.500000 0.200000 0.300000"
BOX_B = "0.100000 0.200000 0.050000 0.060000"


def _ds(tmp_path, labels):
    for sub in ("images", "labels"):
        (tmp_path / sub).mkdir()
    for stem, lines in labels.items():
        (tmp_path / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        (tmp_path / "images" / f"{stem}.jpg").write_bytes(b"")
    return tmp_path


def _decide(queue, verdicts):
    rows = list(csv.DictReader(queue.open(encoding="utf-8", newline="")))
    for r, d in zip(rows, verdicts):
        r["decision"] = d
    with queue.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sq.FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def test_make_queues_only_stairs_boxes(tmp_path):
    src = _ds(tmp_path, {"a": [f"{PERSON} {BOX_A}", f"{UP} {BOX_B}"],
                         "b": [f"{PERSON} {BOX_A}"]})
    q = src / "q.csv"
    assert sq.make(src, q) == 1
    rows = list(csv.DictReader(q.open(encoding="utf-8", newline="")))
    assert [r["image"] for r in rows] == ["a.jpg"]
    assert rows[0]["suspected_av14"] == "stairs_up/stairs_down"
    assert rows[0]["decision"] == ""


def test_apply_retags_only_the_decided_box(tmp_path):
    src = _ds(tmp_path, {"a": [f"{PERSON} {BOX_A}", f"{UP} {BOX_B}"],
                         "b": [f"{UP} {BOX_A}"]})
    q = src / "q.csv"
    sq.make(src, q)
    _decide(q, ["stairs_down", "stairs_up"])   # a -> down, b stays up
    sq.apply(src, q)
    assert (src / "labels" / "a.txt").read_text().split("\n")[:2] == [
        f"{PERSON} {BOX_A}", f"{DOWN} {BOX_B}"]          # person untouched
    assert (src / "labels" / "b.txt").read_text().strip() == f"{UP} {BOX_A}"


def test_apply_is_idempotent(tmp_path):
    """A second run must not re-match: the box is no longer stairs_up."""
    src = _ds(tmp_path, {"a": [f"{UP} {BOX_A}"]})
    q = src / "q.csv"
    sq.make(src, q)
    _decide(q, ["stairs_down"])
    sq.apply(src, q)
    once = (src / "labels" / "a.txt").read_text()
    sq.apply(src, q)
    assert (src / "labels" / "a.txt").read_text() == once


def test_undecided_rows_leave_the_label_alone(tmp_path):
    src = _ds(tmp_path, {"a": [f"{UP} {BOX_A}"]})
    q = src / "q.csv"
    sq.make(src, q)
    sq.apply(src, q)                                     # no verdicts entered
    assert (src / "labels" / "a.txt").read_text().strip() == f"{UP} {BOX_A}"


def test_drop_removes_the_box_but_keeps_the_frame(tmp_path):
    """An empty label file is a background negative and must survive."""
    src = _ds(tmp_path, {"a": [f"{UP} {BOX_A}"]})
    q = src / "q.csv"
    sq.make(src, q)
    _decide(q, ["drop"])
    sq.apply(src, q)
    assert (src / "labels" / "a.txt").exists()
    assert (src / "labels" / "a.txt").read_text().strip() == ""


def test_unknown_decision_is_refused(tmp_path):
    """A typo must not silently leave every box as stairs_up."""
    src = _ds(tmp_path, {"a": [f"{UP} {BOX_A}"]})
    q = src / "q.csv"
    sq.make(src, q)
    _decide(q, ["stairs-down"])
    with pytest.raises(SystemExit, match="cannot apply"):
        sq.apply(src, q)
    assert (src / "labels" / "a.txt").read_text().strip() == f"{UP} {BOX_A}"
