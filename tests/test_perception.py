"""Multi-provider fusion: one physical object must not be announced twice.

Two detectors that both know 'person' will both report the same person. Without
fusion the user hears "person ahead" twice, from models that disagree slightly
on the box and not at all on the fact.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import perception  # noqa: E402
from server.perception import AUTHORITY, canonical, fuse, permitted  # noqa: E402


def det(provider, label, box, conf, **kw):
    return {"provider_id": provider, "model_id": provider + ".pt", "class_id": 0,
            "canonical": canonical(label), "label": label, "raw": label,
            "conf": conf, "box": box, "timestamp": 0.0, "kind": "object",
            "hazard": kw.get("hazard", False)}


def test_both_vocabularies_reach_the_same_canonical_class():
    assert canonical("dining table") == canonical("table") == "TABLE"
    assert canonical("person") == "PERSON"
    assert canonical("stairs_up") == "STAIRS_UP"
    assert canonical("traffic light") == "TRAFFIC_LIGHT"


def test_the_same_person_seen_by_two_providers_is_one_detection():
    out = fuse([det("general", "person", (100, 100, 200, 400), .82),
                det("assistive", "person", (104, 96, 196, 408), .76)])
    assert len(out) == 1, "the user would have been told about one person twice"
    # confidence is the best either model gave it; both are recorded
    assert out[0]["conf"] == pytest.approx(.82)
    assert out[0]["agreed_by"] == ["assistive", "general"]


def test_two_actual_people_stay_two_detections():
    out = fuse([det("general", "person", (100, 100, 200, 400), .82),
                det("general", "person", (600, 100, 700, 400), .80)])
    assert len(out) == 2


def test_the_class_authority_decides_the_survivor_not_the_confidence():
    # PERSON prefers the assistive provider, even when COCO is more confident
    out = fuse([det("general", "person", (100, 100, 200, 400), .95),
                det("assistive", "person", (102, 102, 198, 398), .60)])
    assert out[0]["provider_id"] == "assistive"
    assert out[0]["conf"] == pytest.approx(.95), "the better score is still carried over"


def test_each_provider_only_speaks_for_classes_it_was_trained_on():
    # the AV models have never seen a car; COCO has never seen a stairs_down
    assert not permitted(det("assistive", "car", (0, 0, 10, 10), .9))
    assert permitted(det("general", "car", (0, 0, 10, 10), .9))
    assert not permitted(det("general", "stairs_up", (0, 0, 10, 10), .9))
    assert permitted(det("assistive", "stairs_up", (0, 0, 10, 10), .9))


def test_an_unlisted_class_is_accepted_from_any_provider():
    assert "BACKPACK" not in AUTHORITY
    assert permitted(det("general", "backpack", (0, 0, 10, 10), .9))


def test_outdoor_and_indoor_hazards_survive_together():
    """The whole point: a corridor model and a street model in one stream."""
    out = fuse([det("general", "car", (0, 0, 100, 100), .9),
                det("assistive", "stairs_up", (200, 200, 300, 300), .7)])
    assert {d["canonical"] for d in out} == {"CAR", "STAIRS_UP"}


def test_a_provider_whose_weights_are_missing_does_not_break_the_others(monkeypatch):
    monkeypatch.setattr(perception.config, "PERCEPTION_PROVIDERS",
                        [{"id": "general", "path": "models/__absent__.pt"},
                         {"id": "assistive", "path": "models/__also_absent__.pt"}])
    perception.reset()
    try:
        assert perception.providers() == []
    finally:
        perception.reset()
