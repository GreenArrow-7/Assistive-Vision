"""Tests for scripts/remap_to_av14.py (COCO pre-labels -> AV-14 indices)."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import remap_to_av14 as rm  # noqa: E402
from server.classes_av import AV_ALL_CLASSES as AV_CLASSES  # noqa: E402

# deliberately NOT the real COCO indices: the script must resolve by NAME
COCO = ["car", "person", "vase", "dining table", "tv", "zebra"]
BOX = "0.500000 0.400000 0.100000 0.200000"


def make_src(tmp_path, labels: dict) -> Path:
    src = tmp_path / "raw"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    (src / "classes.txt").write_text("\n".join(COCO))
    for stem, lines in labels.items():
        (src / "images" / f"{stem}.jpg").write_bytes(b"jpeg")
        (src / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    return src


def run(tmp_path, labels):
    out = tmp_path / "out"
    rm.remap(make_src(tmp_path, labels), out)
    return out


def label_lines(out: Path, stem: str):
    return (out / "labels" / f"{stem}.txt").read_text().splitlines()


def test_auto_rewrites_index_and_keeps_box(tmp_path):
    out = run(tmp_path, {"f_0": [f"1 {BOX}", f"3 {BOX}"]})   # person, dining table
    assert label_lines(out, "f_0") == [
        f"{AV_CLASSES.index('person')} {BOX}",
        f"{AV_CLASSES.index('table')} {BOX}",       # dining table -> table
    ]


def test_dropped_classes_vanish_entirely(tmp_path):
    out = run(tmp_path, {"f_0": [f"0 {BOX}", f"5 {BOX}", f"1 {BOX}"]})  # car, zebra, person
    assert label_lines(out, "f_0") == [f"{AV_CLASSES.index('person')} {BOX}"]
    with (out / "review_queue.csv").open() as fh:
        assert list(csv.DictReader(fh)) == []       # dropped != queued


def test_unknown_coco_index_drops_rather_than_guessing(tmp_path):
    out = run(tmp_path, {"f_0": [f"79 {BOX}"]})     # id past the end of classes.txt
    assert label_lines(out, "f_0") == []


def test_review_queue_lists_manual_boxes_and_omits_them_from_labels(tmp_path):
    out = run(tmp_path, {"f_0": [f"2 {BOX}", f"4 {BOX}"]})   # vase, tv
    # nothing auto-written, and no empty label either -- see
    # test_frame_with_only_queued_boxes_gets_no_label_file
    assert not (out / "labels" / "f_0.txt").exists()
    with (out / "review_queue.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["image"], r["coco_class"], r["suspected_av14"]) for r in rows] == [
        ("f_0.jpg", "vase", "dustbin"),
        ("f_0.jpg", "tv", "signboard"),
    ]
    assert [rows[0][k] for k in ("cx", "cy", "w", "h")] == BOX.split()


def test_review_queue_shows_every_candidate_for_multi_way_classes(tmp_path):
    src = make_src(tmp_path, {"f_0": []})
    (src / "classes.txt").write_text("refrigerator")
    (src / "labels" / "f_0.txt").write_text(f"0 {BOX}")
    out = tmp_path / "out"
    rm.remap(src, out)
    with (out / "review_queue.csv").open() as fh:
        assert next(csv.DictReader(fh))["suspected_av14"] == "signboard/door/pole"


def test_empty_frames_survive_as_background_negatives(tmp_path):
    out = run(tmp_path, {"f_0": [], "f_1": [f"0 {BOX}"]})    # empty, car-only
    for stem in ("f_0", "f_1"):
        assert (out / "images" / f"{stem}.jpg").exists()
        assert (out / "labels" / f"{stem}.txt").read_text() == ""


def test_source_of_truth_is_av_classes(tmp_path):
    out = run(tmp_path, {"f_0": [f"1 {BOX}"]})
    # classes.txt is AV_CLASSES verbatim, in index order, for Roboflow
    assert (out / "classes.txt").read_text().splitlines() == AV_CLASSES
    # and every mapping target is a real AV-14 class, not a stale name
    targets = [*rm.AUTO.values(), *(c for t in rm.MANUAL.values() for c in t)]
    assert set(targets) <= set(AV_CLASSES)


def test_indices_follow_av_classes_order_not_a_hardcoded_copy(tmp_path, monkeypatch):
    # rotate the schema: if any index were hardcoded, the output would not move
    rotated = AV_CLASSES[1:] + AV_CLASSES[:1]
    monkeypatch.setattr(rm, "AV_CLASSES", rotated)
    out = run(tmp_path, {"f_0": [f"1 {BOX}"]})              # person
    assert label_lines(out, "f_0") == [f"{rotated.index('person')} {BOX}"]
    assert (out / "classes.txt").read_text().splitlines() == rotated


def test_frame_with_only_queued_boxes_gets_no_label_file(tmp_path):
    # an empty label = "this frame is background". A frame whose every box is
    # awaiting adjudication is NOT background; writing one trains the rarest
    # classes as negatives.
    out = run(tmp_path, {"pending": [f"4 {BOX}"],          # tv -> queued
                         "negative": [f"0 {BOX}"]})        # car -> dropped
    assert not (out / "labels" / "pending.txt").exists()
    assert (out / "images" / "pending.jpg").exists()       # annotator still sees it
    assert (out / "labels" / "negative.txt").read_text() == ""   # real negative


def test_frame_keeps_label_when_it_has_both_auto_and_queued_boxes(tmp_path):
    out = run(tmp_path, {"f_0": [f"1 {BOX}", f"4 {BOX}"]})  # person + tv
    assert label_lines(out, "f_0") == [f"{AV_CLASSES.index('person')} {BOX}"]


def test_rerun_refuses_to_overwrite_annotation(tmp_path):
    out = run(tmp_path, {"f_0": [f"1 {BOX}"]})
    hand_drawn = f"{AV_CLASSES.index('stairs_down')} {BOX}"
    (out / "labels" / "f_0.txt").write_text(hand_drawn)
    with pytest.raises(SystemExit):
        rm.remap(tmp_path / "raw", out)
    assert (out / "labels" / "f_0.txt").read_text() == hand_drawn
    rm.remap(tmp_path / "raw", out, force=True)            # escape hatch works
    assert label_lines(out, "f_0") == [f"{AV_CLASSES.index('person')} {BOX}"]


def test_force_clears_a_stale_empty_label_for_a_now_held_back_frame(tmp_path):
    src = make_src(tmp_path, {"f_0": [f"4 {BOX}"]})        # tv -> queued
    out = tmp_path / "out"
    (out / "labels").mkdir(parents=True)
    (out / "labels" / "f_0.txt").write_text("")            # older run's false negative
    rm.remap(src, out, force=True)
    assert not (out / "labels" / "f_0.txt").exists()


def test_refuses_to_write_back_into_the_source_dataset(tmp_path):
    src = make_src(tmp_path, {"f_0": [f"1 {BOX}"]})        # av_raw is untouchable
    with pytest.raises(SystemExit):
        rm.remap(src, src)
    assert (src / "labels" / "f_0.txt").read_text() == f"1 {BOX}"


def test_class_names_are_stripped_before_lookup(tmp_path):
    # a trailing space in classes.txt must not silently reroute a class to DROP
    src = make_src(tmp_path, {"f_0": [f"0 {BOX}"]})
    (src / "classes.txt").write_text("person \n")
    rm.remap(src, tmp_path / "out")
    assert label_lines(tmp_path / "out", "f_0") == [f"{AV_CLASSES.index('person')} {BOX}"]


def test_source_dataset_is_not_modified(tmp_path):
    labels = {"f_0": [f"0 {BOX}", f"1 {BOX}"]}
    src = make_src(tmp_path, labels)
    before = {p: p.read_bytes() for p in src.rglob("*") if p.is_file()}
    rm.remap(src, tmp_path / "out")
    assert {p: p.read_bytes() for p in src.rglob("*") if p.is_file()} == before


# --- pull_open_datasets writes into the same vocabulary -------------------
import pull_open_datasets as pod  # noqa: E402  (fiftyone is imported lazily)


def test_open_images_map_derives_indices_from_names():
    """The map used to hardcode "Waste container": 7, correct when dustbin sat
    at 7. Reordering AV_ALL_CLASSES so the trained classes hold 0-6 moved
    dustbin to 5 and put stairs_down at 7 -- the stale literal would have
    labelled a thousand bins as the sole AV_CRITICAL class, silently."""
    for oi, name in pod.OI_TO_AV_NAME.items():
        assert pod.OI_TO_AV14[oi] == AV_CLASSES.index(name), oi
    assert AV_CLASSES[pod.OI_TO_AV14["Waste container"]] == "dustbin"
    assert AV_CLASSES[pod.OI_TO_AV14["Stairs"]] == "stairs_up"


def test_open_images_map_only_names_real_classes():
    for name in pod.OI_TO_AV_NAME.values():
        assert name in AV_CLASSES, name


def test_selected_classes_are_all_in_the_write_map():
    """Selecting on a class absent from the map would download images whose
    target object is then left unlabelled -- a trained false negative."""
    for oi in pod.SELECT_CLASSES:
        assert oi in pod.OI_TO_AV_NAME, oi
