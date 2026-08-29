"""Tests for reindex_labels: the index-shift that silently relabels boxes.

Removing a class from the middle of a schema moves every index above it. A
label file left untouched then means something else entirely -- this is the
same class of bug that made importing raw COCO pre-labels dangerous.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import reindex_labels as ri  # noqa: E402

OLD = ["person", "chair", "table", "door", "stairs_up", "stairs_down",
       "pole", "dustbin", "signboard"]
NEW = ["person", "chair", "table", "door", "stairs_up", "dustbin", "signboard"]


def make_src(tmp_path, labels, names=OLD):
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    for stem, text in labels.items():
        (src / "images" / f"{stem}.jpg").write_bytes(b"")
        (src / "labels" / f"{stem}.txt").write_text(text)
    (src / "classes.txt").write_text("\n".join(names))
    return src


def test_indices_above_a_removed_class_shift_down(tmp_path):
    # dustbin 7 -> 5 and signboard 8 -> 6 once stairs_down/pole are removed
    src = make_src(tmp_path, {"a": "7 0.5 0.5 0.2 0.2\n8 0.1 0.1 0.1 0.1"})
    out = tmp_path / "out"
    ri.reindex(src, out, NEW)
    lines = (out / "labels" / "a.txt").read_text().split("\n")
    assert lines[0].split()[0] == "5", lines
    assert lines[1].split()[0] == "6", lines


def test_box_geometry_is_untouched(tmp_path):
    src = make_src(tmp_path, {"a": "7 0.123456 0.234567 0.2 0.3"})
    out = tmp_path / "out"
    ri.reindex(src, out, NEW)
    assert (out / "labels" / "a.txt").read_text().split()[1:] == \
        ["0.123456", "0.234567", "0.2", "0.3"]


def test_removed_class_boxes_are_dropped(tmp_path):
    src = make_src(tmp_path, {"a": "5 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.1 0.1"})
    out = tmp_path / "out"
    kept, dropped = ri.reindex(src, out, NEW)
    assert dropped["stairs_down"] == 1
    assert kept["person"] == 1
    assert (out / "labels" / "a.txt").read_text().strip() == "0 0.1 0.1 0.1 0.1"


def test_frame_emptied_by_the_drop_survives_as_background(tmp_path):
    """Deleting the file would lose a hard negative the model needs."""
    src = make_src(tmp_path, {"a": "6 0.5 0.5 0.2 0.2"})   # pole only
    out = tmp_path / "out"
    ri.reindex(src, out, NEW)
    assert (out / "labels" / "a.txt").exists()
    assert (out / "labels" / "a.txt").read_text().strip() == ""
    assert (out / "images" / "a.jpg").exists()


def test_refuses_without_a_recorded_vocabulary(tmp_path):
    src = make_src(tmp_path, {"a": "0 0.5 0.5 0.2 0.2"})
    (src / "classes.txt").unlink()
    with pytest.raises(SystemExit, match="Refusing to guess"):
        ri.reindex(src, tmp_path / "out", NEW)


def test_refuses_index_beyond_the_declared_vocabulary(tmp_path):
    """Catches labels that are not the vocabulary classes.txt claims."""
    src = make_src(tmp_path, {"a": "62 0.5 0.5 0.2 0.2"})   # a COCO index
    with pytest.raises(SystemExit, match="defines only"):
        ri.reindex(src, tmp_path / "out", NEW)


def test_refuses_to_overwrite_the_source(tmp_path):
    src = make_src(tmp_path, {"a": "0 0.5 0.5 0.2 0.2"})
    with pytest.raises(SystemExit, match="must differ"):
        ri.reindex(src, src, NEW)


def test_blank_line_in_classes_txt_is_refused(tmp_path):
    src = make_src(tmp_path, {"a": "0 0.5 0.5 0.2 0.2"})
    (src / "classes.txt").write_text("person\n\nchair")
    with pytest.raises(SystemExit, match="blank"):
        ri.reindex(src, tmp_path / "out", NEW)


def test_output_records_the_new_vocabulary(tmp_path):
    src = make_src(tmp_path, {"a": "0 0.5 0.5 0.2 0.2"})
    out = tmp_path / "out"
    ri.reindex(src, out, NEW)
    assert (out / "classes.txt").read_text().splitlines() == NEW


def test_identity_remap_is_a_faithful_copy(tmp_path):
    src = make_src(tmp_path, {"a": "3 0.5 0.5 0.2 0.2"}, names=NEW)
    out = tmp_path / "out"
    ri.reindex(src, out, NEW)
    assert (out / "labels" / "a.txt").read_text().strip() == "3 0.5 0.5 0.2 0.2"
