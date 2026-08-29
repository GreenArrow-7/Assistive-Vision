"""Regression tests for the frame writer (needs cv2, unlike the logic tests)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_dataset as bd  # noqa: E402


def test_slug_strips_non_ascii():
    assert bd.slug("AIIMS Jammu ( Inside Tour )\U0001F3E8") == "AIIMS_Jammu_Inside_Tour"
    assert bd.slug("Delhi University｜｜Library") == "Delhi_University_Library"
    assert bd.slug("\U0001F3E8\U0001F3E8") == "video"       # never returns empty
    assert bd.slug("plain-name_01") == "plain-name_01"      # ASCII untouched


def test_imwrite_writes_non_ascii_path(tmp_path):
    # cv2.imwrite returns True but silently writes nothing here on Windows
    img = np.zeros((16, 16, 3), np.uint8)
    dst = tmp_path / "hospital_\U0001F3E8_0001.jpg"
    bd.imwrite(dst, img)
    assert dst.exists() and dst.stat().st_size > 0
