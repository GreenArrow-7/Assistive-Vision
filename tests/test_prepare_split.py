"""Tests for prepare_split: schema resolution and the by-video holdout.

The schema tests exist because data.yaml is what fixes a trained model's class
names, and server/detector.py identifies models by their names — wrong names
here become a detector with the wrong hazard set. The holdout test exists
because a leaked val split reports a mAP nobody can reproduce on real footage.
"""
import re
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import prepare_split as ps  # noqa: E402
from server.classes_av import AV_CLASSES  # noqa: E402


def make_src(tmp_path, videos, class_ids):
    """Fake build_dataset output: {video_stem: n_frames}, one box per class id.

    Images are empty files — nothing under test decodes them, only hardlinks.
    """
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    for stem, n in videos.items():
        for i in range(n):
            name = f"{stem}_{i:05d}"
            (src / "images" / f"{name}.jpg").write_bytes(b"")
            (src / "labels" / f"{name}.txt").write_text(
                "\n".join(f"{c} 0.5 0.5 0.2 0.2" for c in class_ids))
    return src


def test_av14_labels_produce_av14_names(tmp_path):
    src = make_src(tmp_path, {"mall": 8, "hospital": 4}, range(len(AV_CLASSES)))
    out = tmp_path / "out"
    ps.prepare(src, out, 0.25, schema="av7", allow_sparse=True)
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert [cfg["names"][i] for i in range(len(cfg["names"]))] == AV_CLASSES


def test_coco_labels_autodetect_and_use_classes_txt(tmp_path):
    src = make_src(tmp_path, {"mall": 8, "hospital": 4}, [0, 58, 62])
    (src / "classes.txt").write_text("\n".join(f"c{i}" for i in range(80)))
    out = tmp_path / "out"
    ps.prepare(src, out, 0.25)          # no --schema: index 62 proves COCO
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert len(cfg["names"]) == 80 and cfg["names"][62] == "c62"


def test_blank_line_in_classes_txt_does_not_shift_every_name(tmp_path):
    """Line number is the class index. Filtering blanks out renamed every class
    above the gap (index 62 came back 'c63') and data.yaml still looked fine."""
    src = make_src(tmp_path, {"mall": 8, "hospital": 4}, [0, 58, 62])
    names = [f"c{i}" for i in range(80)]
    names[12] = ""
    (src / "classes.txt").write_text("\n".join(names))
    with pytest.raises(SystemExit, match="line 13 is blank"):
        ps.prepare(src, tmp_path / "out", 0.25)


def test_auto_refuses_to_guess_av14_from_low_indices(tmp_path):
    """A COCO set holding only ids 0-13 looks exactly like AV-14. Guessing
    'av7' would put AV-14 names on COCO-trained weights, so it must refuse."""
    src = make_src(tmp_path, {"mall": 8, "hospital": 4}, [0, 2, len(AV_CLASSES) - 1])
    with pytest.raises(SystemExit, match="cannot infer label schema"):
        ps.prepare(src, tmp_path / "out", 0.25)


def test_av14_flag_rejects_out_of_range_labels(tmp_path):
    src = make_src(tmp_path, {"mall": 8, "hospital": 4}, [0, 62])
    with pytest.raises(SystemExit, match="class index 62"):
        ps.prepare(src, tmp_path / "out", 0.25, schema="av7", allow_sparse=True)


def test_empty_labels_alone_are_not_enough_to_infer(tmp_path):
    """402 of the real label files are empty; an all-empty set says nothing.

    Empty files are now dropped before schema resolution (they carry no class
    ids, so they could never have informed it), so the refusal arrives as the
    more specific all-empty message rather than "cannot infer label schema".
    The contract under test is unchanged: an all-empty set must be refused.
    """
    src = make_src(tmp_path, {"mall": 8, "hospital": 4}, [])
    with pytest.raises(SystemExit, match="are empty -- nothing to train on"):
        ps.prepare(src, tmp_path / "out", 0.25)


def test_holdout_never_splits_a_video_across_train_and_val(tmp_path):
    src = make_src(tmp_path, {"mall": 20, "hosp": 12, "campus": 9, "uni": 7},
                   range(len(AV_CLASSES)))
    out = tmp_path / "out"
    ps.prepare(src, out, 0.25, schema="av7", allow_sparse=True)

    def videos_in(split):
        return {ps.video_of(p.stem)
                for p in (out / split / "images").glob("*.jpg")}

    train, val = videos_in("train"), videos_in("val")
    assert train and val
    assert not (train & val)
    assert train | val == {"mall", "hosp", "campus", "uni"}
    # every frame landed somewhere, none duplicated
    n = sum(len(list((out / s / "images").glob("*.jpg"))) for s in ("train", "val"))
    assert n == 48


def test_av14_yaml_has_not_drifted_from_the_schema():
    """scripts/av7.yaml duplicates AV_CLASSES by hand; index order is
    load-bearing, so a silent divergence would mislabel every prediction."""
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "scripts" / "av7.yaml").read_text())
    assert [cfg["names"][i] for i in range(len(cfg["names"]))] == AV_CLASSES


def test_val_frac_lands_near_target_not_past_it(tmp_path):
    # 100 frames, request 20% -> videos of 5/10/30/55. The old "add until
    # n >= target" took 5+10+30 = 45%; nearest-subset stops at 5+10 = 15%.
    src = make_src(tmp_path, {"a": 55, "b": 30, "c": 10, "d": 5}, range(len(AV_CLASSES)))
    out = tmp_path / "out"
    ps.prepare(src, out, 0.20, schema="av7", allow_sparse=True)
    n_val = len(list((out / "val" / "images").glob("*.jpg")))
    n_train = len(list((out / "train" / "images").glob("*.jpg")))
    assert n_val + n_train == 100
    assert abs(n_val / 100 - 0.20) <= abs(45 / 100 - 0.20), n_val


def test_sparse_av14_split_is_refused(tmp_path):
    # only person/chair/table populated: the state datasets/av14_seed is in
    src = make_src(tmp_path, {"mall": 8, "hospital": 4}, [0, 1, 2])
    with pytest.raises(SystemExit, match=f"{len(AV_CLASSES) - 3} of {len(AV_CLASSES)} AV classes have NO boxes"):
        ps.prepare(src, tmp_path / "out", 0.25, schema="av7")


def test_sparse_av14_split_builds_with_override(tmp_path):
    src = make_src(tmp_path, {"mall": 8, "hospital": 4}, [0, 1, 2])
    out = tmp_path / "out"
    ps.prepare(src, out, 0.25, schema="av7", allow_sparse=True)
    names = yaml.safe_load((out / "data.yaml").read_text())["names"]
    assert list(names.values()) == list(AV_CLASSES)


def test_coco_split_not_subject_to_sparse_guard(tmp_path):
    # COCO pre-labels legitimately cover a fraction of the 80 names
    src = make_src(tmp_path, {"mall": 8, "hospital": 4}, [0, 62])
    (src / "classes.txt").write_text("\n".join(f"c{i}" for i in range(80)))
    ps.prepare(src, tmp_path / "out", 0.25)      # must not raise


def test_data_yaml_is_portable_not_machine_specific(tmp_path):
    r"""data.yaml used to carry path: C:\Users\...\datasets\av14_seed_split.

    TRAINING.md trains on Colab, where that path does not exist, so the split
    could not be used on the one machine it was built for. Ultralytics roots the
    dataset at Path(yaml_file).parent when "path" is absent, so leaving it out
    makes the directory relocatable. "path: ." would NOT do: a relative path
    resolves against ultralytics' DATASETS_DIR, not the yaml's own location.
    """
    src = make_src(tmp_path, {"mall": 8, "hospital": 4}, range(len(AV_CLASSES)))
    out = tmp_path / "out"
    ps.prepare(src, out, 0.25, schema="av7", allow_sparse=True)
    raw = (out / "data.yaml").read_text()
    cfg = yaml.safe_load(raw)
    assert "path" not in cfg, cfg.get("path")
    assert str(tmp_path) not in raw          # no absolute path anywhere
    assert cfg["train"] == "train/images" and cfg["val"] == "val/images"


def _split_counts(out):
    return (len(list((out / "train" / "labels").glob("*.txt"))),
            len(list((out / "val" / "labels").glob("*.txt"))))


def test_empty_pre_label_files_are_dropped_by_default(tmp_path):
    """A pre-labeller's empty file records that a COCO detector found none of
    ITS classes -- not that the frame is empty. av14_seed was 45% such files,
    training the model to ignore the stairs/doors/signs COCO cannot express."""
    src = make_src(tmp_path, {"mall": 8, "hospital": 8}, [0, 1, 2])
    # empty 4 frames of ONE video, so both videos survive the by-video holdout
    for lbl in sorted((src / "labels").glob("mall_*.txt"))[:4]:
        lbl.write_text("")
    out = tmp_path / "out"
    ps.prepare(src, out, 0.25, schema="av7", allow_sparse=True)
    assert sum(_split_counts(out)) == 12, _split_counts(out)


def test_keep_empty_retains_them_as_background(tmp_path):
    src = make_src(tmp_path, {"mall": 8, "hospital": 8}, [0, 1, 2])
    for lbl in sorted((src / "labels").glob("mall_*.txt"))[:4]:
        lbl.write_text("")
    out = tmp_path / "out"
    ps.prepare(src, out, 0.25, schema="av7", allow_sparse=True, keep_empty=True)
    assert sum(_split_counts(out)) == 16, _split_counts(out)


def make_extra(tmp_path, n_train, n_val, class_ids):
    """Fake pull_open_datasets output: split encoded in the filename prefix."""
    ex = tmp_path / "extra"
    (ex / "images").mkdir(parents=True)
    (ex / "labels").mkdir(parents=True)
    for pre, n in (("oi_tr_", n_train), ("oi_va_", n_val)):
        for i in range(n):
            name = f"{pre}{i:04d}"
            (ex / "images" / f"{name}.jpg").write_bytes(b"")
            (ex / "labels" / f"{name}.txt").write_text(
                "\n".join(f"{c} 0.5 0.5 0.2 0.2" for c in class_ids))
    return ex


def test_extra_source_honours_its_own_split(tmp_path):
    """Open Images' train/validation split is disjoint by construction, and its
    photos have no temporal correlation for the by-video holdout to protect.
    video_of() would make each one its own single-frame 'video'."""
    src = make_src(tmp_path, {"mall": 8, "hospital": 8}, [0, 1, 2])
    ex = make_extra(tmp_path, 10, 4, [3, 4, 5])
    out = tmp_path / "out"
    ps.prepare(src, out, 0.25, schema="av7", allow_sparse=True, extra=ex)
    tr = {p.name for p in (out / "train" / "images").glob("*.jpg")}
    va = {p.name for p in (out / "val" / "images").glob("*.jpg")}
    assert sum(n.startswith("oi_tr_") for n in tr) == 10
    assert sum(n.startswith("oi_va_") for n in va) == 4
    assert not any(n.startswith("oi_va_") for n in tr)
    assert not any(n.startswith("oi_tr_") for n in va)
    assert not (tr & va)


def test_extra_file_without_a_split_prefix_is_refused(tmp_path):
    """Guessing a side would be train/val leakage no metric would reveal."""
    src = make_src(tmp_path, {"mall": 8, "hospital": 8}, [0, 1, 2])
    ex = make_extra(tmp_path, 4, 2, [3])
    (ex / "images" / "mystery.jpg").write_bytes(b"")
    (ex / "labels" / "mystery.txt").write_text("3 0.5 0.5 0.2 0.2")
    with pytest.raises(SystemExit, match="no known split prefix"):
        ps.prepare(src, tmp_path / "out", 0.25, schema="av7",
                   allow_sparse=True, extra=ex)


def test_extra_labels_count_towards_the_sparse_guard(tmp_path):
    """The guard must see the merged set: classes the extra source supplies are
    no longer missing, and it should stop complaining about them."""
    src = make_src(tmp_path, {"mall": 8, "hospital": 8}, [0, 1, 2])
    ex = make_extra(tmp_path, 6, 2, [3, 4, 5])
    with pytest.raises(SystemExit) as e:      # still sparse: signboard short
        ps.prepare(src, tmp_path / "out", 0.25, schema="av7", extra=ex)
    msg = str(e.value)
    assert f"of {len(AV_CLASSES)} AV classes have NO boxes" in msg, msg
    for supplied in ("door", "stairs_up", "dustbin"):
        assert supplied not in msg, msg


def make_src_per_video(tmp_path, spec):
    """spec = {video_stem: (n_frames, [class_ids])} -- lets a class live in
    exactly one video, which is how a train-starved class actually arises."""
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)
    for stem, (n, ids) in spec.items():
        for i in range(n):
            name = f"{stem}_{i:05d}"
            (src / "images" / f"{name}.jpg").write_bytes(b"")
            (src / "labels" / f"{name}.txt").write_text(
                "\n".join(f"{c} 0.5 0.5 0.2 0.2" for c in ids))
    return src


def test_class_present_only_in_val_is_refused(tmp_path):
    """0 boxes in train but present in val is worse than absent: the model
    cannot learn the class, yet val scores it 0 AP and drags mAP down."""
    src = make_src_per_video(tmp_path, {
        "mall": (8, list(range(len(AV_CLASSES) - 1))),   # everything except 11
        "hospital": (4, list(range(len(AV_CLASSES)))),          # the only home of class 11
    })
    with pytest.raises(SystemExit, match="only in val, never in train"):
        ps.prepare(src, tmp_path / "out", 0.25, schema="av7")


def test_starved_class_named_in_the_message(tmp_path):
    src = make_src_per_video(tmp_path, {
        "mall": (8, list(range(len(AV_CLASSES) - 1))),
        "hospital": (4, list(range(len(AV_CLASSES)))),
    })
    with pytest.raises(SystemExit, match=re.escape(AV_CLASSES[-1])):
        ps.prepare(src, tmp_path / "out", 0.25, schema="av7")


def test_balanced_classes_do_not_trip_the_starved_guard(tmp_path):
    src = make_src_per_video(tmp_path, {
        "mall": (8, list(range(len(AV_CLASSES)))),
        "hospital": (4, list(range(len(AV_CLASSES)))),
    })
    ps.prepare(src, tmp_path / "out", 0.25, schema="av7")   # must not raise
