"""Unit tests for the pure-logic modules (no ML dependencies)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import config, detector, priority, spatial, symbols  # noqa: E402
from server.classes_av import AV_CLASSES  # noqa: E402

W, H = 900, 600


def box(cx, cy, w=60, h=40):
    return (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)


def test_direction_thirds():
    assert spatial.direction(box(100, 300), W) == "on your left"
    assert spatial.direction(box(450, 300), W) == "straight ahead"
    assert spatial.direction(box(800, 300), W) == "on your right"


def test_proximity_scaling():
    tiny = box(450, 300, 30, 20)
    huge = box(450, 300, 500, 400)
    assert spatial.proximity(tiny, W, H) == "at a distance"
    assert spatial.proximity(huge, W, H) == "very close"


def test_clean_query():
    assert symbols.clean_query("Where is the exit?") == "exit"
    assert symbols.clean_query("navigate to the reception please") == "reception"
    assert symbols.clean_query("find washroom") == "washroom"


def test_symbols_from_text_keywords():
    texts = [{"label": "FIRE EXIT", "box": box(100, 100), "kind": "text"}]
    syms = symbols.symbols_from_texts(texts)
    assert syms and syms[0]["label"] == "emergency exit"


def test_keyword_match_prefers_text():
    texts = [{"label": "EXIT →", "box": box(800, 100), "kind": "text"}]
    m = symbols.match_keyword("exit", texts, [], [])
    assert m is texts[0]


def test_priority_order_hazard_first():
    hz = [{"label": "car", "box": box(450, 300, 500, 400), "kind": "hazard",
           "direction": "straight ahead", "proximity": "very close"}]
    txt = [{"label": "PHARMACY", "box": box(800, 100), "kind": "text",
            "direction": "on your right", "proximity": "nearby"}]
    out = priority.build_speech(hz, [], txt, [], keyword=None, match=None)
    assert out["speech"].startswith("Caution.")
    assert "car" in out["speech"] and "PHARMACY" in out["speech"]
    assert out["hazard_count"] == 1


def test_priority_keyword_not_found():
    out = priority.build_speech([], [], [], [], keyword="exit", match=None)
    assert "exit not found" in out["speech"]


def test_priority_keyword_found_message():
    m = {"label": "washroom", "box": box(120, 200), "kind": "symbol",
         "direction": "on your left", "proximity": "nearby"}
    out = priority.build_speech([], [], [], [m], keyword="washroom", match=m)
    assert "washroom is on your left" in out["speech"]
    # matched symbol must not be duplicated in the Signs section
    assert out["speech"].count("washroom") == 1


def test_priority_match_label_differs_from_keyword():
    m = {"label": "FIRE EXIT", "box": box(120, 200), "kind": "text",
         "direction": "on your right", "proximity": "nearby"}
    out = priority.build_speech([], [], [m], [], keyword="exit", match=m)
    assert "exit found: FIRE EXIT is on your right" in out["speech"]


def test_steps_pinhole_monotonic():
    # taller box (closer person) => fewer steps
    near = spatial.estimate_steps("person", box(450, 300, 200, 600), W, H)
    far = spatial.estimate_steps("person", box(450, 300, 60, 150), W, H)
    assert near is not None and far is not None
    assert near < far
    assert near >= 1


def test_steps_unknown_class_is_none():
    assert spatial.estimate_steps("WASHROOM SIGN", box(450, 300), W, H) is None


def test_fmt_uses_steps_wording():
    it = {"label": "chair", "box": box(800, 300, 120, 240), "kind": "object",
          "direction": "on your right", "proximity": "nearby", "steps": 2}
    out = priority.build_speech([], [it], [], [], keyword=None, match=None)
    assert "chair, about two steps on your right" in out["speech"]


def test_steps_truncated_box_uses_ground_plane():
    # sink cut off at LEFT frame edge, bottom visible at y=560 (of 720):
    # height model would wildly overestimate; ground plane must dominate
    b = (0, 470, 180, 560)
    s = spatial.estimate_steps("sink", b, 1280, 720)
    assert s is not None and s <= 8, s   # was ~19 with height-only model


def test_steps_object_at_feet():
    # chair filling lower frame, touching bottom edge => 1 step
    b = (200, 200, 760, 718)
    assert spatial.estimate_steps("chair", b, 1280, 720) == 1


def test_steps_fusion_takes_closer():
    # person: height says ~3, ground-plane says ~5 -> conservative min = 3
    b = (300, 60, 640, 660)
    s = spatial.estimate_steps("person", b, 1280, 720)
    assert s == 3, s


def test_steps_capped():
    b = (450, 355, 470, 375)   # tiny distant object, above-horizon guard
    s = spatial.estimate_steps("bottle", b, 1280, 720)
    assert s is None or s <= 99


# ---------- AV-14 trained-model behaviour ----------
def test_stairs_down_is_critical_and_first():
    hz = [
        {"label": "person", "raw": "person", "box": box(450, 300, 200, 500),
         "direction": "straight ahead", "proximity": "nearby", "steps": 5},
        {"label": "stairs going down", "raw": "stairs_down",
         "box": box(450, 500, 400, 200), "direction": "straight ahead",
         "proximity": "very close", "steps": 3},
    ]
    out = priority.build_speech(hz, [], [], [], None, None)
    assert out["speech"].startswith("Warning!")
    assert "stairs going down" in out["speech"]
    assert "proceed carefully" in out["speech"]
    # person still announced, but after the critical alert
    assert out["speech"].index("stairs") < out["speech"].index("person")


def test_keyword_matches_trained_sign_class():
    objs = [{"label": "washroom sign", "raw": "sign_washroom",
             "box": box(800, 200), "direction": "on your right",
             "proximity": "nearby", "steps": 4}]
    m = symbols.match_keyword("washroom", [], [], objs)
    assert m is objs[0]


def test_av_heights_used_for_new_classes():
    # 'door' is not in COCO KNOWN_HEIGHTS but is in AV_HEIGHTS (2.0 m)
    s = spatial.estimate_steps("door", box(450, 300, 120, 400), W, H)
    assert s is not None and s >= 1


def test_critical_hazard_is_counted():
    # regression: a lone stairs_down reported hazard_count 0, so the client
    # skipped vibration / red flash / speech interrupt on the worst hazard
    hz = [{"label": "stairs going down", "raw": "stairs_down",
           "box": box(450, 500, 400, 200), "direction": "straight ahead",
           "proximity": "very close", "steps": 3}]
    out = priority.build_speech(hz, [], [], [], None, None)
    assert out["hazard_count"] == 1


def test_keyword_does_not_match_substring():
    # regression: 'men' is a substring of MENU/WOMEN, so a menu board was
    # announced as a washroom; 'exit' matched 'EXITED'
    menu = [{"label": "MENU", "box": box(100, 100), "kind": "text"}]
    assert symbols.symbols_from_texts(menu) == []
    assert symbols.match_keyword("washroom", menu, [], []) is None
    exited = [{"label": "EXITED STAFF ONLY", "box": box(100, 100), "kind": "text"}]
    assert symbols.match_keyword("exit", exited, [], []) is None


def test_keyword_still_matches_whole_words():
    for label, kw in [("LADIES WASHROOM", "washroom"), ("FIRE EXIT", "exit"),
                      ("WOMEN", "washroom"), ("Gents", "washroom")]:
        t = [{"label": label, "box": box(100, 100), "kind": "text"}]
        assert symbols.match_keyword(kw, t, [], []) is t[0], (label, kw)


def test_hazards_sorted_by_steps():
    hz = [
        {"label": "table", "raw": "table", "box": box(200, 300),
         "direction": "on your left", "proximity": "nearby", "steps": 7},
        {"label": "chair", "raw": "chair", "box": box(700, 300),
         "direction": "on your right", "proximity": "very close", "steps": 2},
    ]
    out = priority.build_speech(hz, [], [], [], None, None)
    assert out["speech"].index("chair") < out["speech"].index("table")


# ---------- P0 regressions ----------
def test_feet_check_uses_real_width_not_assumed_16_9():
    """A chair at the user's feet on a PORTRAIT frame must read as 1 step.

    estimate_steps derived frame area from frame_h alone assuming 16:9. On a
    720x1280 portrait frame that denominator is 216% too large, so area_ratio
    came out ~3x too small, never cleared FEET_AREA_RATIO, and the object fell
    through to the height model (~3 steps) instead of the trip warning.
    """
    b = (120, 700, 660, 1278)                    # chair filling the lower frame
    assert spatial.estimate_steps("chair", b, 720, 1280) == 1
    # the old 16:9-from-height denominator, for contrast
    old_ratio = ((660 - 120) * (1278 - 700)) / float(1280 * 1280 * 16 / 9)
    new_ratio = ((660 - 120) * (1278 - 700)) / float(720 * 1280)
    assert old_ratio < 0.15 <= new_ratio


def test_feet_check_unchanged_on_16_9():
    # the frames that already worked must not move
    assert spatial.estimate_steps("chair", (200, 200, 760, 718), 1280, 720) == 1


def test_annotate_passes_width_through():
    items = [{"label": "chair", "box": (120, 700, 660, 1278)}]
    spatial.annotate(items, 720, 1280)
    assert items[0]["steps"] == 1


# ---------- detector schema validation ----------
def test_coco_model_at_custom_path_is_not_treated_as_av14():
    """The bug: 'custom' was inferred from the weights PATH, so a COCO model
    dropped at models/av_obstacle.pt switched hazard lookup to AV_HAZARDS,
    where no vehicle appears — cars silently stopped being hazards."""
    coco = {i: n for i, n in enumerate(sorted(config.HAZARD_CLASSES |
                                              config.OBJECT_CLASSES))}
    assert detector.detect_schema(coco) == detector.SCHEMA_COCO


def test_av14_model_detected_from_class_names():
    assert detector.detect_schema(dict(enumerate(AV_CLASSES))) == detector.SCHEMA_AV7


def test_unknown_schema_is_refused_loudly():
    with pytest.raises(ValueError, match="Unrecognised detector class schema"):
        detector.detect_schema({0: "widget", 1: "gizmo"})


def test_vehicles_stay_hazards_under_coco_schema():
    # the concrete safety property the schema check protects
    for v in ("car", "bus", "truck", "bicycle", "motorcycle"):
        assert v in config.HAZARD_CLASSES


# ---------- schema role integrity (B2) ----------
def test_every_role_names_a_real_class():
    """Roles are keyed by NAME, so a typo makes a hazard that never fires.
    classes_av checks this at import; assert the checker actually bites."""
    import importlib
    from server import classes_av as cav
    for label, names in (("AV_HAZARDS", cav.AV_HAZARDS),
                         ("AV_CRITICAL", cav.AV_CRITICAL),
                         ("AV_HEIGHTS", cav.AV_HEIGHTS),
                         ("AV_SPOKEN", cav.AV_SPOKEN),
                         ("keywords", cav.AV_KEYWORD_TO_CLASS.values())):
        assert set(names) <= set(cav.AV_ALL_CLASSES), label
    with pytest.raises(ValueError, match="never fire"):
        cav._check_names("bogus", {"not_a_class"})
    importlib.reload(cav)          # leave the module as we found it


def test_critical_is_always_a_hazard():
    """priority.build_speech splits criticals out of the hazard list it is
    given, so a critical that is not a hazard would never reach it."""
    from server import classes_av as cav
    assert cav.AV_CRITICAL <= cav.AV_HAZARDS


def test_dormant_hazards_are_derived_not_hand_listed():
    """Un-retiring a class must not also require updating a second list."""
    from server import classes_av as cav
    trained = set(cav.AV_CLASSES)
    assert cav.TRAINED_HAZARDS == cav.AV_HAZARDS & trained
    assert cav.DORMANT_HAZARDS == (cav.AV_HAZARDS | cav.AV_CRITICAL) - trained
    assert cav.CRITICAL_ACTIVE == bool(cav.AV_CRITICAL & trained)


def test_critical_path_is_dormant_and_says_so():
    """stairs_down is annotated but not trained, so the interrupt-everything
    branch cannot fire. The system must report that, not imply it works."""
    from server import classes_av as cav
    assert cav.CRITICAL_ACTIVE is False
    assert "stairs_down" in cav.DORMANT_HAZARDS
    # and the dormant branch really is unreachable for any trained class
    assert not any(c in cav.AV_CRITICAL for c in cav.AV_CLASSES)


def test_build_speech_raises_no_critical_while_dormant():
    """End-to-end: even handed a hazard for every trained class, no 'Warning!'
    is emitted, because none of them carries the critical role."""
    from server import classes_av as cav
    hazards = [{"label": c, "raw": c, "direction": "straight ahead",
                "proximity": "very close", "steps": 2}
               for c in cav.AV_CLASSES if c in cav.AV_HAZARDS]
    out = priority.build_speech(hazards, [], [], [])
    assert "Warning!" not in out["speech"], out["speech"]
    assert out["hazard_count"] == len(hazards)


def test_critical_path_reactivates_when_the_class_is_trained():
    """The branch is DORMANT, not deleted. Hand build_speech a stairs_down
    hazard directly: it must still produce the interrupt. This is what makes
    'annotate it and add the name back' a one-line change rather than a
    rewrite -- if this ever fails, the migration path is broken."""
    from server import classes_av as cav
    assert "stairs_down" in cav.AV_CRITICAL       # role still declared
    hz = [{"label": "stairs going down", "raw": "stairs_down",
           "direction": "straight ahead", "proximity": "very close",
           "steps": 2}]
    out = priority.build_speech(hz, [], [], [])
    assert out["speech"].startswith("Warning!"), out["speech"]
    assert "Stop and proceed carefully" in out["speech"]
    assert out["hazard_count"] == 1               # criticals must still count


# ---------- retired keyword routes degrade to OCR, not to nothing (B3) ----------
RETIRED_SIGN_TEXT = {
    "washroom": "GENTS WASHROOM", "restroom": "RESTROOM", "toilet": "TOILET",
    "bathroom": "BATHROOM", "exit": "EXIT", "way out": "WAY OUT",
    "emergency exit": "EMERGENCY EXIT", "lift": "LIFT", "elevator": "ELEVATOR",
    "reception": "RECEPTION", "front desk": "FRONT DESK",
    "wheelchair": "WHEELCHAIR ACCESS", "pole": "POLE",
}


def _retired_keywords():
    from server import classes_av as cav
    return {k: t for k, t in cav.AV_KEYWORD_TO_CLASS.items()
            if t not in cav.AV_CLASSES}


def _text_item(label):
    return {"label": label, "raw": None, "kind": "text",
            "direction": "on your left", "proximity": "nearby"}


def test_every_retired_keyword_still_resolves_through_ocr():
    """These routes point at classes no model can emit, so the text path is the
    ONLY way 'find washroom' can succeed. If a refactor breaks it, the feature
    disappears with nothing failing."""
    for kw in _retired_keywords():
        texts = [_text_item(RETIRED_SIGN_TEXT[kw])]
        syms = symbols.symbols_from_texts(texts)
        assert symbols.match_keyword(kw, texts, syms, []) is not None, kw


def test_retired_keyword_on_a_textless_pictogram_reports_not_found():
    """The honest failure: an ISO-7001 pictogram carries no text, so an
    OCR-only lookup cannot find it. It must say so, not invent a match."""
    assert symbols.match_keyword("washroom", [], [], []) is None
    out = priority.build_speech([], [], [], [], keyword="washroom", match=None)
    assert "not found" in out["speech"], out["speech"]


def test_trained_keyword_still_takes_the_detection_path():
    """The retired routes must not have broken the ones that do work: a trained
    class matches on `raw`, without needing any text in frame."""
    from server import classes_av as cav
    for kw, target in cav.AV_KEYWORD_TO_CLASS.items():
        if target not in cav.AV_CLASSES:
            continue
        obj = {"label": cav.AV_SPOKEN.get(target, target), "raw": target,
               "kind": "object", "direction": "straight ahead",
               "proximity": "nearby"}
        assert symbols.match_keyword(kw, [], [], [obj]) is obj, kw


def test_retired_targets_are_still_declared_for_the_migration_path():
    """Deleting these routes would make un-retiring a class a two-file change.
    They are kept for the same reason AV_CRITICAL still names stairs_down."""
    retired = _retired_keywords()
    assert "washroom" in retired and retired["washroom"] == "sign_washroom"
    from server import classes_av as cav
    assert set(retired.values()) <= set(cav.AV_RETIRED)
