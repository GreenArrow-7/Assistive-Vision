"""Evaluate supplied oriented quads and paired OCR transcriptions, not invented data.

JSON input: [{"truth": [{"quad": [[x,y],...], "text": "EXIT"}],
              "predictions": [{"quad": [[x,y],...], "text": "EXIT"}]}]
This is fixed-threshold detection F1, not AP. Use YOLO val for OBB mAP.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

def rotated_iou(a, b):
    def polygon(points):
        p = np.asarray(points, dtype=np.float32)
        if p.shape != (4, 2) or not np.isfinite(p).all():
            raise ValueError('Expected four finite polygon points')
        return cv2.convexHull(p)
    a, b = polygon(a), polygon(b)
    area_a, area_b = cv2.contourArea(a), cv2.contourArea(b)
    if not area_a or not area_b:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a, b)
    return float(inter / max(area_a + area_b - inter, 1e-9))

def edit_distance(a, b):
    row = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1]+1, row[j]+1, row[j-1]+(x != y)))
        row = new
    return row[-1]

def evaluate(frames, threshold=.5):
    tp = fp = fn = chars = errors = exact = words = word_errors = 0
    for frame in frames:
        truth, predictions = frame['truth'], frame['predictions']
        pairs = sorted(((rotated_iou(t['quad'], p['quad']), i, j)
                        for i,t in enumerate(truth) for j,p in enumerate(predictions)), reverse=True)
        used_t, used_p = set(), set()
        for score, i, j in pairs:
            if score < threshold or i in used_t or j in used_p:
                continue
            used_t.add(i); used_p.add(j); tp += 1
            a, b = truth[i]['text'], predictions[j]['text']
            chars += len(a); errors += edit_distance(a, b); exact += a == b
            words += len(a.split()); word_errors += edit_distance(a.split(), b.split())
        fp += len(predictions)-len(used_p)
        fn += len(truth)-len(used_t)
    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/(tp+fn) if tp+fn else 0
    return dict(frames=len(frames), tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0,
                matched_ocr_accuracy=exact/tp if tp else None,
                matched_cer=errors/chars if chars else None,
                matched_wer=word_errors/words if words else None,
                note='OCR metrics cover IoU-matched pairs only; misses are reflected in recall.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.input.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
