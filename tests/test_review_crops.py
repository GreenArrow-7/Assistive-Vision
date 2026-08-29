"""End-to-end check for the review-queue adjudicator.

The 219 parked boxes are only decidable by looking at the crop, so this drives
the real HTTP handler: fetch a row, post a verdict, confirm it reached the CSV.
A verdict that does not survive to disk silently loses annotation work.
"""
import csv
import json
import sys
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import review_crops as rc  # noqa: E402

FIELDS = ["image", "coco_class", "suspected_av14", "cx", "cy", "w", "h",
          "decision"]


def make_queue(tmp_path, rows):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in {r["image"] for r in rows}:
        cv2.imwrite(str(img_dir / name),
                    np.full((240, 320, 3), 127, dtype=np.uint8))
    q = tmp_path / "review_queue.csv"
    with q.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return q, img_dir


def row(image="a.jpg", coco="tv", suspected="signboard", decision=""):
    return {"image": image, "coco_class": coco, "suspected_av14": suspected,
            "cx": "0.5", "cy": "0.5", "w": "0.25", "h": "0.25",
            "decision": decision}


@pytest.fixture
def server(tmp_path):
    q, img_dir = make_queue(tmp_path, [
        row("a.jpg", "tv", "signboard"),
        row("b.jpg", "refrigerator", "signboard/door/pole"),
        row("c.jpg", "vase", "dustbin"),
    ])
    rc.Handler.rows, rc.Handler.fields = rc.load(q)
    rc.Handler.queue, rc.Handler.img_dir = q, img_dir
    rc.Handler.skipped, rc.Handler.history = set(), []
    httpd = HTTPServer(("127.0.0.1", 0), rc.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", q
    httpd.shutdown()


def get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return r.read(), r.headers.get("Content-Type")


def post(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def decisions(q):
    with q.open(encoding="utf-8", newline="") as fh:
        return [r["decision"] for r in csv.DictReader(fh)]


def test_candidates_splits_multi_valued_suggestions():
    assert rc.candidates(row(suspected="signboard/door/pole")) == \
        ["signboard", "door", "pole"]
    assert rc.candidates(row(suspected="")) == []


def test_verdict_reaches_the_csv_and_advances(server):
    base, q = server
    first = json.loads(get(base, "/next")[0])
    assert first["row"]["coco_class"] == "tv"
    assert first["candidates"] == ["signboard"]
    post(base, "/decide", {"i": first["i"], "decision": "signboard"})
    assert decisions(q)[0] == "signboard"          # survived to disk
    assert json.loads(get(base, "/next")[0])["row"]["coco_class"] == \
        "refrigerator"                             # advanced past it


def test_drop_is_recorded_not_left_blank(server):
    base, q = server
    i = json.loads(get(base, "/next")[0])["i"]
    post(base, "/decide", {"i": i, "decision": "drop"})
    assert decisions(q)[0] == "drop"


def test_skip_leaves_the_row_undecided_for_a_later_pass(server):
    base, q = server
    post(base, "/skip")
    assert decisions(q)[0] == ""                   # still blank on disk
    assert json.loads(get(base, "/next")[0])["row"]["coco_class"] == \
        "refrigerator"                             # but not re-offered now


def test_back_undoes_the_previous_verdict(server):
    base, q = server
    i = json.loads(get(base, "/next")[0])["i"]
    post(base, "/decide", {"i": i, "decision": "signboard"})
    assert decisions(q)[0] == "signboard"
    post(base, "/decide", {"i": None, "decision": "__back__"})
    assert decisions(q)[0] == ""                   # cleared again
    assert json.loads(get(base, "/next")[0])["row"]["coco_class"] == "tv"


def test_crop_and_full_frame_both_render_jpeg(server):
    base, _ = server
    i = json.loads(get(base, "/next")[0])["i"]
    for zoom in ("1", "0"):
        body, ctype = get(base, f"/img?i={i}&zoom={zoom}")
        assert ctype == "image/jpeg"
        assert cv2.imdecode(np.frombuffer(body, np.uint8),
                            cv2.IMREAD_COLOR) is not None


def test_finished_queue_reports_no_row(server):
    base, q = server
    for _ in range(3):
        nxt = json.loads(get(base, "/next")[0])
        post(base, "/decide", {"i": nxt["i"], "decision": "drop"})
    assert json.loads(get(base, "/next")[0])["row"] is None
    assert decisions(q) == ["drop", "drop", "drop"]


def test_missing_image_is_refused_up_front(tmp_path):
    """A queue pointed at the wrong image dir must fail before the browser
    opens, not render 404s for every row."""
    q, img_dir = make_queue(tmp_path, [row("a.jpg")])
    (img_dir / "a.jpg").unlink()
    with pytest.raises(SystemExit, match="are not under"):
        rc.main(q, img_dir, 8799, open_browser=False)


def test_save_preserves_every_column(tmp_path):
    q, _ = make_queue(tmp_path, [row("a.jpg", "vase", "dustbin")])
    rows, fields = rc.load(q)
    rows[0]["decision"] = "dustbin"
    rc.save(q, rows, fields)
    with q.open(encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        assert r.fieldnames == FIELDS
        got = next(iter(r))
    assert got["decision"] == "dustbin" and got["cx"] == "0.5"
