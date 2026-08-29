"""End-to-end checks for the OBB label converters, on synthetic fixtures.

Real ICDAR-2015 / SynthText need registration + a 41 GB download, so these
build a minimal dataset in tmp_path and run the converters for real.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import convert_icdar_to_obb as icdar  # noqa: E402
import convert_synthtext_to_obb as synth  # noqa: E402


def _img(path: Path, w=200, h=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), "white").save(path)


def test_icdar_converts_and_normalizes(tmp_path):
    _img(tmp_path / "img" / "img_1.jpg")           # 200x100
    (tmp_path / "gt").mkdir()
    (tmp_path / "gt" / "gt_img_1.txt").write_text(
        "﻿20,10,100,10,100,50,20,50,EXIT\n"   # BOM, as ICDAR ships it
        "0,0,50,0,50,25,0,25,###\n",               # unreadable -> skipped
        encoding="utf-8")

    icdar.convert(tmp_path / "img", tmp_path / "gt", tmp_path / "out")

    label = (tmp_path / "out" / "labels" / "img_1.txt").read_text().split("\n")
    assert len(label) == 1, "### instance must be skipped"
    v = label[0].split()
    assert v[0] == "0"
    assert [float(x) for x in v[1:3]] == [0.1, 0.1]      # 20/200, 10/100
    assert [float(x) for x in v[5:7]] == [0.5, 0.5]      # 100/200, 50/100
    assert (tmp_path / "out" / "images" / "img_1.jpg").exists()


def test_icdar_clamps_out_of_frame_coords(tmp_path):
    # ICDAR boxes routinely run a few px past the edge; YOLO rejects >1.0
    _img(tmp_path / "img" / "img_2.jpg")
    (tmp_path / "gt").mkdir()
    (tmp_path / "gt" / "gt_img_2.txt").write_text(
        "-5,-5,260,-5,260,140,-5,140,SIGN\n", encoding="utf-8")

    icdar.convert(tmp_path / "img", tmp_path / "gt", tmp_path / "out")

    vals = [float(x) for x in
            (tmp_path / "out" / "labels" / "img_2.txt").read_text().split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in vals), vals


def test_icdar_skips_image_without_gt(tmp_path):
    _img(tmp_path / "img" / "img_3.jpg")
    (tmp_path / "gt").mkdir()
    icdar.convert(tmp_path / "img", tmp_path / "gt", tmp_path / "out")
    assert not list((tmp_path / "out" / "images").glob("*.jpg"))


def _gt_mat(path, names, bbs):
    """Write a gt.mat shaped like the real one: wordBB is a CELL array, one
    (2,4,N) entry per image. np.array([...], dtype=object) will not do — with
    uniform shapes numpy collapses it into a numeric array instead of a cell."""
    from scipy.io import savemat

    cell = np.empty(len(bbs), dtype=object)
    for i, b in enumerate(bbs):
        cell[i] = b
    nm = np.empty(len(names), dtype=object)
    for i, n in enumerate(names):
        nm[i] = n
    savemat(str(path), {"imnames": nm, "wordBB": cell})


def test_synthtext_converts_multi_box(tmp_path):
    pytest.importorskip("scipy.io")
    for n in ("a", "b"):
        _img(tmp_path / "root" / "set1" / f"{n}.jpg")      # 200x100
    # wordBB is (2, 4, N): [x|y][corner][word]
    bb = np.zeros((2, 4, 2))
    bb[0, :, 0], bb[1, :, 0] = [20, 100, 100, 20], [10, 10, 50, 50]
    bb[0, :, 1], bb[1, :, 1] = [120, 180, 180, 120], [60, 60, 90, 90]
    other = np.zeros((2, 4, 1))
    other[0, :, 0], other[1, :, 0] = [0, 40, 40, 0], [0, 0, 20, 20]
    _gt_mat(tmp_path / "root" / "gt.mat", ["set1/a.jpg", "set1/b.jpg"], [bb, other])

    synth.convert(tmp_path / "root", tmp_path / "out", max_images=0)

    lines = (tmp_path / "out" / "train" / "labels" / "set1_a.txt").read_text().splitlines()
    assert len(lines) == 2, lines
    v = [float(x) for x in lines[0].split()[1:]]
    assert v[0:2] == [0.1, 0.1]           # 20/200, 10/100 — x,y interleaved
    assert all(0.0 <= x <= 1.0 for x in v)
    assert (tmp_path / "out" / "train" / "images" / "set1_a.jpg").exists()
    # the second image's single box also lands
    assert len((tmp_path / "out" / "train" / "labels" / "set1_b.txt")
               .read_text().splitlines()) == 1


def test_synthtext_drops_degenerate_box(tmp_path):
    pytest.importorskip("scipy.io")
    for n in ("a", "b"):
        _img(tmp_path / "root" / "set1" / f"{n}.jpg")
    zero = np.zeros((2, 4, 1))            # zero-area: all corners identical
    good = np.zeros((2, 4, 1))
    good[0, :, 0], good[1, :, 0] = [20, 100, 100, 20], [10, 10, 50, 50]
    _gt_mat(tmp_path / "root" / "gt.mat", ["set1/a.jpg", "set1/b.jpg"], [zero, good])

    synth.convert(tmp_path / "root", tmp_path / "out", max_images=0)

    lbl = tmp_path / "out" / "train" / "labels"
    assert not (lbl / "set1_a.txt").exists()      # degenerate -> no file at all
    assert (lbl / "set1_b.txt").exists()


# ---------- the /health -> text_obb wiring ----------
def test_non_obb_model_is_refused(tmp_path, monkeypatch):
    """A detect model at models/text_obb.pt made detect_text return [] for
    every frame, silently killing text detection."""
    from server import config, text_pipeline

    import ultralytics

    class _Detect:                       # stand-in for a plain detect model
        task = "detect"

    fake = tmp_path / "text_obb.pt"      # the file only has to EXIST
    fake.write_bytes(b"")
    monkeypatch.setattr(ultralytics, "YOLO", lambda path: _Detect())
    monkeypatch.setattr(text_pipeline, "_obb", None)
    monkeypatch.setattr(config, "TEXT_OBB_MODEL", str(fake))
    with pytest.raises(ValueError, match="not 'obb'"):
        text_pipeline._get_obb()


def test_missing_model_falls_back_quietly(tmp_path, monkeypatch):
    from server import config, text_pipeline

    monkeypatch.setattr(text_pipeline, "_obb", None)
    monkeypatch.setattr(config, "TEXT_OBB_MODEL", str(tmp_path / "nope.pt"))
    assert text_pipeline._get_obb() is None       # -> EasyOCR CRAFT path
