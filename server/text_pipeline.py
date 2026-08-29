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


def detect_text(frame_bgr):
    """Return list of recognized text items with oriented-box metadata."""
    obb = _get_obb()
    reader = _get_reader()
    items = []

    if obb is not None:
        # --- Path A: fine-tuned YOLO-OBB proposals -> deskew -> OCR ---
        res = obb.predict(frame_bgr, conf=0.35, verbose=False)[0]
        if res.obb is not None:
            for poly, conf in zip(res.obb.xyxyxyxy.cpu().numpy(),
                                  res.obb.conf.cpu().numpy()):
                quad = poly.reshape(4, 2).tolist()
                roi = _deskew(frame_bgr, quad)
                out = reader.readtext(roi, detail=1, paragraph=False)
                text = " ".join(t for _, t, c in out if c >= config.TEXT_CONF).strip()
                if len(text) > 1:
                    items.append({
                        "label": text,
                        "box": _quad_to_aabb(quad),
                        "conf": float(conf),
                        "angle": round(_quad_angle(quad), 1),
                        "kind": "text",
                    })
        return items

    # --- Path B: EasyOCR end-to-end (CRAFT detector handles rotation) ---
    results = reader.readtext(frame_bgr, detail=1, paragraph=False)
    for quad, text, conf in results:
        text = text.strip()
        if conf >= config.TEXT_CONF and len(text) > 1:
            items.append({
                "label": text,
                "box": _quad_to_aabb(quad),
                "conf": float(conf),
                "angle": round(_quad_angle(quad), 1),
                "kind": "text",
            })
    return items
