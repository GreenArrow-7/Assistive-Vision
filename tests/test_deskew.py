"""The quad-ordering half of oriented text reading.

YOLO-OBB's xyxyxyxy starts at an arbitrary corner, so the rectified crop used to
come out mirrored or upside down depending on which corner that was. Detection
metrics cannot see that defect: the box is right and the OCR is empty.
"""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.text_pipeline import _deskew, _order_quad  # noqa: E402


def marked_frame():
    """Black frame with a white mark inside the top-left of the text region."""
    f = np.zeros((200, 200, 3), np.uint8)
    f[60:110, 20:180] = 40           # the "text" region
    f[66:80, 26:50] = 255            # its top-left corner mark
    return f


UPRIGHT = [[20, 60], [180, 60], [180, 110], [20, 110]]


@pytest.mark.parametrize("start", range(4))
def test_any_start_corner_gives_the_same_crop(start):
    frame = marked_frame()
    rolled = UPRIGHT[start:] + UPRIGHT[:start]
    assert np.array_equal(_deskew(frame, rolled), _deskew(frame, UPRIGHT)), \
        "crop depends on which corner the detector happened to emit first"


def test_crop_is_upright_not_flipped():
    crop = _deskew(marked_frame(), UPRIGHT)
    h, w = crop.shape[:2]
    quadrant = lambda ys, xs: crop[ys, xs].mean()
    assert quadrant(slice(0, h // 2), slice(0, w // 2)) > quadrant(slice(h // 2, h), slice(0, w // 2))
    assert quadrant(slice(0, h // 2), slice(0, w // 2)) > quadrant(slice(0, h // 2), slice(w // 2, w))


def test_rotated_text_keeps_its_baseline_horizontal():
    # a 30-degree tilted sign: the long edge must become the crop's width
    c, ang = np.array([100.0, 85.0]), np.radians(30)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    quad = [(R @ (np.array(p, float) - c) + c).tolist() for p in UPRIGHT]
    crop = _deskew(np.zeros((200, 200, 3), np.uint8), quad)
    assert crop.shape[1] > crop.shape[0], "baseline was not mapped to the horizontal"


def test_vertical_text_does_not_crash():
    # no edge runs left-to-right, so the longest-edge fallback has to fire
    vertical = [[90, 20], [110, 20], [110, 180], [90, 180]]
    assert _order_quad(vertical).shape == (4, 2)
    assert _deskew(np.zeros((200, 200, 3), np.uint8), vertical).size > 0
