"""Spatial guidance: direction (left/center/right) and proximity from bounding boxes.

Pure functions — no ML dependencies — so they are trivially unit-testable.
Boxes are (x1, y1, x2, y2) in pixel coordinates.
"""
import math

from . import config
from .classes_av import AV_HEIGHTS

_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
          7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def steps_phrase(n: int) -> str:
    return _WORDS.get(n, str(n))


def _height_of(label: str):
    return config.KNOWN_HEIGHTS.get(label) or AV_HEIGHTS.get(label)


def estimate_steps(label: str, box, frame_w: int, frame_h: int,
                   pitch_deg: float = 0.0, vfov: float = None):
    """Walking-steps distance from TWO independent monocular estimators, fused
    conservatively (min = closer = safer for a blind user).

    E1 height model:   Z = f · H_class / h_box     (needs known class height,
                       invalid when the box is truncated top AND bottom)
    E2 ground plane:   Z = h_cam / tan(θ), θ = atan((y_bottom − c_y)/f) + pitch
                       — uses the object's floor-contact point, corrected by
                       the phone's downward tilt reported by its gyroscope.
    Special case: box touching the frame bottom with large area = at the
    user's feet -> 1 step.
    """
    x1, y1, x2, y2 = box
    m = config.EDGE_MARGIN_PX
    touches_top = y1 <= m
    touches_bottom = y2 >= frame_h - m
    h_px = max(1, y2 - y1)
    v = vfov if (vfov and 25.0 <= vfov <= 90.0) else config.CAMERA_VFOV_DEG
    f_px = (frame_h / 2.0) / math.tan(math.radians(v / 2.0))
    c_y = frame_h / 2.0

    estimates = []

    # E1 — class-height model
    H = _height_of(label)
    if H and not (touches_top and touches_bottom):
        estimates.append(f_px * H / h_px)

    # E2 — ground-plane model with pitch correction
    if not touches_bottom:
        theta = math.atan2(y2 - c_y, f_px) + math.radians(max(0.0, pitch_deg))
        if theta > math.radians(2.0):          # contact point below horizon
            estimates.append(config.CAMERA_HEIGHT_M / math.tan(theta))

    # object at the user's feet
    if touches_bottom:
        # frame area from the ACTUAL width. This assumed a 16:9 frame derived
        # from frame_h alone, which overstates the denominator by 33% on 4:3
        # and by 216% on a portrait 9:16 frame — the ratio came out ~3x too
        # small, so a phone held upright (the normal way to hold it) never
        # cleared FEET_AREA_RATIO and the at-your-feet trip warning never fired.
        area_ratio = ((x2 - x1) * h_px) / float(frame_w * frame_h)
        if area_ratio > config.FEET_AREA_RATIO or (H and h_px > 0.55 * frame_h):
            return 1

    if not estimates:
        return None
    z_m = min(estimates)                       # conservative: assume closer
    return int(max(1, min(99, round(z_m / config.STEP_LENGTH_M))))


def direction(box, frame_w: int) -> str:
    """Divide the frame into thirds and report where the box center falls."""
    cx = (box[0] + box[2]) / 2.0
    if cx < frame_w / 3:
        return "on your left"
    if cx < 2 * frame_w / 3:
        return "straight ahead"
    return "on your right"


def direction_short(d: str) -> str:
    if "left" in d:
        return "LEFT"
    if "right" in d:
        return "RIGHT"
    return "AHEAD"


def proximity(box, frame_w: int, frame_h: int) -> str:
    """Estimate proximity from bbox area relative to the frame."""
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    ratio = area / float(frame_w * frame_h)
    if ratio >= config.NEAR_AREA_RATIO:
        return "very close"
    if ratio >= config.MID_AREA_RATIO:
        return "nearby"
    return "at a distance"


def annotate(items, frame_w: int, frame_h: int, pitch_deg: float = 0.0,
             vfov: float = None):
    """Attach direction/proximity/steps to dicts that carry 'box' (+'label')."""
    for it in items:
        it["direction"] = direction(it["box"], frame_w)
        it["proximity"] = proximity(it["box"], frame_w, frame_h)
        key = it.get("raw") or it.get("label", "")
        it["steps"] = estimate_steps(key, it["box"], frame_w, frame_h,
                                     pitch_deg, vfov)
        if it["steps"] is None:
            it["steps"] = estimate_steps(it.get("label", ""), it["box"],
                                         frame_w, frame_h, pitch_deg, vfov)
    return items
