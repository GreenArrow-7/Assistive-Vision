"""Pure-logic checks for the eval harness — no model, no dataset, no network.

These are the places evaluate.py could quietly produce a wrong number: the YOLO
centre-form -> pixel-corner conversion, percentile indexing, IoU, the greedy
match order, P/R/F1, and the split-leak guard. Everything else in that file is
delegation (ultralytics) or transport (TestClient).

Each case is written to FAIL under a plausible mutation of the function it
covers — a test that passes whether or not the logic is right is worse than no
test, because it reads as coverage.
"""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[1] / "scripts")]

from evaluate import gt_boxes, iou, leak_check, match, pct, prf, stats, table  # noqa: E402


def test_gt_boxes_converts_yolo_centre_form_to_pixel_corners(tmp_path):
    """Every P/R number rides on this: a cx/cy-vs-x1/y1 mixup shifts every GT box
    by half its size and quietly halves recall with no error anywhere."""
    p = tmp_path / "f.txt"
    p.write_text("2 0.5 0.25 0.5 0.5\n9 0.1 0.1 0.2 0.2\nmalformed\n")
    out = gt_boxes(p, {2: "table", 9: "sign_washroom"}, 200, 400)
    #                cx 0.5*200=100, w 0.5*200=100 -> x 50..150
    #                cy 0.25*400=100, h 0.5*400=200 -> y 0..200
    assert out[0] == ("table", (50.0, 0.0, 150.0, 200.0))
    assert out[1][0] == "sign_washroom"
    assert len(out) == 2                       # the short line is dropped, not crashed
    assert gt_boxes(tmp_path / "missing.txt", {}, 1, 1) == []   # unlabelled frame
    # an index the data.yaml does not name must stay traceable, not become a name
    assert gt_boxes(p, {}, 1, 1)[0][0] == "2"


def test_percentiles_are_nearest_rank_and_p99_degrades_to_max():
    xs = list(range(101))                          # 0..100, value == percentile
    assert pct(xs, 0.5) == 50 and pct(xs, 0.9) == 90 and pct(xs, 0.99) == 99
    # n < 100: there is no 99th sample, so p99 must land on max, not interpolate
    assert pct([5, 1, 3], 0.99) == 5 == max([5, 1, 3])
    assert stats([]) is None                        # graceful: no samples, no row
    assert stats([10.0])["p99"] == stats([10.0])["max"] == 10.0


def test_iou_matches_a_hand_computed_overlap():
    # 10x10 boxes offset by 5 in both axes: intersection 25, union 175
    assert abs(iou((0, 0, 10, 10), (5, 5, 15, 15)) - 25 / 175) < 1e-9
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_greedy_match_gives_the_contested_gt_to_the_higher_confidence_pred():
    """The two GTs overlap, and only `loose` can reach G2 at IoU>=0.5.

    The obvious version of this test (one real GT, one decoy pred that never
    clears the threshold) scores the same under ANY sort order, so it does not
    test the ordering at all. This one flips 2 TP into 1 TP + 1 FP if the sort
    is reversed, which is the whole point of matching by confidence.
    """
    gts = [("door", (0, 0, 10, 10)), ("door", (4, 0, 14, 10))]
    tight = (0, 0, 10, 10)          # IoU 1.0 on G1, 0.43 on G2 -> G1 or nothing
    loose = (1.5, 0, 11.5, 10)      # IoU 0.74 on G1, 0.60 on G2 -> either, prefers G1
    assert match([("door", tight, 0.9), ("door", loose, 0.5)], gts, 0.5)["door"] \
        == {"tp": 2, "fp": 0, "fn": 0}
    # same boxes, confidences swapped: loose takes G1 first and strands tight
    c = match([("door", tight, 0.5), ("door", loose, 0.9)], gts, 0.5)["door"]
    assert c == {"tp": 1, "fp": 1, "fn": 1}
    assert prf(c) == {"p": 0.5, "r": 0.5, "f1": 0.5}
    # asymmetric on purpose: p==r cannot tell precision from recall, and the
    # deployed thresholds make exactly the high-P/low-R shape this must report
    assert prf({"tp": 3, "fp": 1, "fn": 6}) == {"p": 0.75, "r": 0.333, "f1": 0.462}
    assert prf({"tp": 0, "fp": 0, "fn": 0}) == {"p": 0.0, "r": 0.0, "f1": 0.0}


def test_leak_check_catches_a_video_on_both_sides_of_the_split():
    assert leak_check(["MallTour_00001"], ["Campus_00002"]) == []
    assert leak_check(["MallTour_00001", "Campus_9"],
                      ["MallTour_00742"]) == ["MallTour"]


def test_table_renders_a_header_even_and_says_so_when_empty():
    assert table(["a", "b"], [[1, 2]]).splitlines()[0] == "| a | b |"
    assert table(["a"], []) == "_(none)_"
