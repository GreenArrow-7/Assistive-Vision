"""The Colab trainer's coverage gate must refuse an incomplete dataset.

Weights trained from a 14-name data.yaml identify as 'av7' to
server/detector.detect_schema, so the server grants them the full AV_HAZARDS
set -- including stairs_down, the only AV_CRITICAL class. If some of those
classes had no training boxes, the model silently never raises them. That is
the failure detect_schema exists to prevent, arriving from the dataset side.

train_av14_colab.py is pasted into a Colab cell, so it must stay standalone and
cannot be imported. The gate cell is therefore executed straight from the file:
this exercises the shipped source, not a copy of it that could drift.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.classes_av import AV_CLASSES  # noqa: E402

TRAINER = Path(__file__).resolve().parent.parent / "scripts" / "train_av14_colab.py"


def gate_source():
    """The '# %% ... coverage gate' cell, up to the next cell marker."""
    lines = TRAINER.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if "coverage gate" in ln)
    end = next(i for i, ln in enumerate(lines[start + 1:], start + 1)
               if ln.startswith("# %%"))
    return "\n".join(lines[start:end])


def make_split(tmp_path, train_ids, val_ids):
    """Minimal split: one label file per split holding one box per class id."""
    for split, ids in (("train", train_ids), ("val", val_ids)):
        lbl = tmp_path / split / "labels"
        lbl.mkdir(parents=True)
        (lbl / "f0.txt").write_text(
            "\n".join(f"{c} 0.5 0.5 0.2 0.2" for c in ids))
    return tmp_path


def run_gate(work, allow_sparse=False):
    """Exec the real cell; returns its namespace (COMPLETE, etc.)."""
    ns = {"work": work, "cfg": {"names": dict(enumerate(AV_CLASSES))},
          "AV_NAMES": list(AV_CLASSES), "Path": Path}
    src = gate_source()
    if allow_sparse:
        src = src.replace("ALLOW_SPARSE = False", "ALLOW_SPARSE = True")
    exec(compile(src, str(TRAINER), "exec"), ns)      # noqa: S102
    return ns


def test_full_coverage_passes(tmp_path):
    every = list(range(len(AV_CLASSES)))
    ns = run_gate(make_split(tmp_path, every, every))
    assert ns["COMPLETE"] is True


def test_missing_classes_refuse_before_burning_a_gpu_session(tmp_path):
    """A GPU session is hours long; refuse before spending it on a split that
    cannot populate every class the schema declares."""
    have = list(range(len(AV_CLASSES) - 3))        # last three left empty
    missing = AV_CLASSES[len(AV_CLASSES) - 3:]
    with pytest.raises(SystemExit) as e:
        run_gate(make_split(tmp_path, have, have))
    msg = str(e.value)
    assert "3 class(es) have NO boxes at all" in msg, msg
    for gone in missing:
        assert gone in msg, msg
    assert "models/av_obstacle.pt" in msg      # names the deploy hazard


def test_class_present_only_in_val_is_refused(tmp_path):
    """The real sign_lift case that motivated this: boxes in val, 0 in train. Unlearnable, and it
    only drags the val metrics down."""
    every = list(range(len(AV_CLASSES)))
    starved = len(AV_CLASSES) - 1
    with pytest.raises(SystemExit) as e:
        run_gate(make_split(tmp_path, [c for c in every if c != starved], every))
    msg = str(e.value)
    assert "only in val, never in train" in msg, msg
    assert AV_CLASSES[starved] in msg, msg


def test_allow_sparse_trains_but_marks_the_run_incomplete(tmp_path):
    have = list(range(len(AV_CLASSES) - 3))
    ns = run_gate(make_split(tmp_path, have, have), allow_sparse=True)
    assert ns["COMPLETE"] is False            # gates the deploy message later


def test_deploy_message_is_withheld_when_incomplete():
    """COMPLETE must guard the 'install at av_obstacle.pt' instruction: a
    matching name set is what makes a partial model dangerous, not safe."""
    src = TRAINER.read_text(encoding="utf-8")
    verify = src[src.index("verify + install"):]
    assert "if COMPLETE:" in verify
    assert "DO NOT DEPLOY" in verify
    assert verify.index("if COMPLETE:") < verify.index("av_obstacle.pt")
