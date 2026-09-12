"""Oriented text detection + OCR.

Strategy:
  * If a fine-tuned YOLO-OBB text model exists at config.TEXT_OBB_MODEL,
    use it for oriented region proposals, deskew each region, OCR with EasyOCR.
  * Otherwise fall back to EasyOCR end-to-end (its CRAFT detector natively
    returns quadrilaterals for rotated text) — same output schema.

Output items: {label, box=(x1,y1,x2,y2), conf, angle, kind='text'}
"""
import math
import os
import re

import cv2
import numpy as np

from . import config

_reader = None
_obb = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(config.OCR_LANGS, gpu=config.OCR_GPU, verbose=False)
    return _reader


def _get_obb():
    """Load optional fine-tuned YOLO-OBB text detector.

    Refuses a non-OBB model: detect_text reads res.obb, which is None for a
    plain detect/segment model, so the wrong file here silently returns zero
    text for every frame instead of failing.
    """
    global _obb
    if _obb is None and os.path.exists(config.TEXT_OBB_MODEL):
        from ultralytics import YOLO
        m = YOLO(config.TEXT_OBB_MODEL)
        if m.task != "obb":
            raise ValueError(
                f"{config.TEXT_OBB_MODEL} is a '{m.task}' model, not 'obb'. "
                "Train with `yolo obb train model=yolov8n-obb.pt`, or delete "
                "the file to fall back to EasyOCR's CRAFT detector."
            )
        _obb = m
    return _obb


def _quad_to_aabb(quad):
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def _quad_angle(quad):
    (x1, y1), (x2, y2) = quad[0], quad[1]
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _deskew(frame, quad):
    """Rotate an oriented quad region to horizontal for cleaner OCR."""
    quad = np.array(quad, dtype=np.float32)
    w = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3])))
    h = int(max(np.linalg.norm(quad[1] - quad[2]), np.linalg.norm(quad[3] - quad[0])))
    w, h = max(w, 8), max(h, 8)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(frame, M, (w, h))


def useful_text(text):
    """Conservative garbage filter; digits/room numbers and short signs survive."""
    text = ' '.join(text.split())
    chars = [c for c in text if not c.isspace()]
    return (2 <= len(chars) <= 180 and
            sum(c.isalnum() for c in chars) / len(chars) >= .6 and
            not re.search(r'(.)\1{4,}', text))


def prepare_text_frame(frame):
    # Improve low contrast locally without changing coordinates or inventing detail.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 65 or gray.std() < 25:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return frame


def merge_text_lines(items):
    """Join adjacent words on the same oriented baseline, preserving a quad.

    CRAFT often returns separate words on a tilted sign, in top-to-bottom order.
    A phrase search needs reading order, not arbitrary OCR output order.
    """
    pending = list(items)
    merged = []
    while pending:
        seed = pending.pop(0)
        angle = math.radians(seed['angle'])
        u = np.array([math.cos(angle), math.sin(angle)])
        v = np.array([-math.sin(angle), math.cos(angle)])

        def bounds(item):
            q = np.asarray(item['quad'])
            x, y = q @ u, q @ v
            return float(x.min()), float(x.max()), float(y.min()), float(y.max())

        group = [seed]
        changed = True
        while changed and len(group) < 8:
            changed = False
            for candidate in list(pending):
                delta = abs((candidate['angle'] - seed['angle'] + 90) % 180 - 90)
                if delta > 12:
                    continue
                a, b, c, d = bounds(candidate)
                for member in group:
                    e, f, g, h = bounds(member)
                    height = min(d-c, h-g)
                    gap = max(a-f, e-b)
                    if height > 0 and abs((c+d-g-h)/2) < .35 * height and -.2 * height <= gap < height:
                        group.append(candidate)
                        pending.remove(candidate)
                        changed = True
                        break
        if len(group) == 1:
            merged.append(seed)
            continue
        group.sort(key=lambda item: bounds(item)[0])
        extents = [bounds(item) for item in group]
        left, right = min(x[0] for x in extents), max(x[1] for x in extents)
        top, bottom = min(x[2] for x in extents), max(x[3] for x in extents)
        quad = [(u*x + v*y).tolist() for x,y in
                [(left,top),(right,top),(right,bottom),(left,bottom)]]
        merged.append({**seed, 'label': ' '.join(i['label'] for i in group),
                       'conf': min(i['conf'] for i in group), 'quad': quad,
                       'box': _quad_to_aabb(quad)})
    return merged


def detect_text(frame_bgr):
    """Return list of recognized text items with oriented-box metadata."""
    frame_bgr = prepare_text_frame(frame_bgr)
    try:
        obb = _get_obb()
    except Exception:
        # An invalid optional detector must not disable the CRAFT/OCR fallback.
        obb = None
    reader = _get_reader()
    items = []

    if obb is not None:
        # --- Path A: fine-tuned YOLO-OBB proposals -> deskew -> OCR ---
        res = obb.predict(frame_bgr, conf=config.TEXT_CONF, imgsz=config.IMAGE_SIZE, device=config.DEVICE, verbose=False)[0]
        if res.obb is not None:
            for poly, conf in zip(res.obb.xyxyxyxy.cpu().numpy(),
                                  res.obb.conf.cpu().numpy()):
                quad = poly.reshape(4, 2).tolist()
                roi = _deskew(frame_bgr, quad)
                # Region is already detected and rectified; do not run CRAFT again.
                out = reader.recognize(roi, detail=1, paragraph=False)
                text = " ".join(t for _, t, c in out if c >= config.TEXT_CONF).strip()
                if useful_text(text):
                    items.append({
                        "label": text,
                        "box": _quad_to_aabb(quad),
                        "conf": min(float(conf), min(c for _, t, c in out if c >= config.TEXT_CONF)),
                        "detector_conf": float(conf),
                        "quad": [[float(x), float(y)] for x, y in quad],
                        "angle": round(_quad_angle(quad), 1),
                        "kind": "text",
                    })
        return merge_text_lines(items)

    # --- Path B: EasyOCR end-to-end (CRAFT detector handles rotation) ---
    results = reader.readtext(frame_bgr, detail=1, paragraph=False)
    retries = 0
    for quad, text, conf in results[:64]:
        text = text.strip()
        # Bounded orientation retry only for uncertain regions; no second detector.
        if .15 <= conf < config.TEXT_CONF and retries < 2:
            retries += 1
            out = reader.recognize(_deskew(frame_bgr, quad), detail=1,
                                   paragraph=False, rotation_info=[90, 270])
            if out:
                _, candidate, score = max(out, key=lambda x: x[2])
                if score > conf: text, conf = candidate.strip(), score
        if conf >= config.TEXT_CONF and useful_text(text):
            items.append({
                "label": text,
                "box": _quad_to_aabb(quad),
                "conf": float(conf),
                "quad": [[float(x), float(y)] for x, y in quad],
                "angle": round(_quad_angle(quad), 1),
                "kind": "text",
            })
    return merge_text_lines(items)
