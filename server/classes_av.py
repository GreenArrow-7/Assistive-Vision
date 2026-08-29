"""AV-7: the class schema the fine-tuned model is actually trained on.

Design notes (defend these in the viva/paper):
  * No generic "obstacle" class — it has no consistent visual signature, so
    annotators disagree and mAP collapses. We name real objects and mark which
    ones BEHAVE as hazards (HAZARD set below). Hazard = a semantic role, not a
    visual class.
  * Signs are trained classes, not OCR keywords — so the system still works
    when a sign is a pictogram with no text (ISO 7001), which is exactly the
    case OCR-only assistive readers fail on. `signboard` carries this today;
    the five pictogram `sign_*` classes are annotated but not yet trained.

WHY 7 AND NOT 14. The schema was designed with 14 classes. Seven of them have
zero annotated boxes, and a declared-but-empty class is not free: the model
carries an output head that can never fire, reports 0 AP, and drags macro mAP
down — while `detector.detect_schema` still certifies the weights as ours,
because it matches on class NAMES. A model that claims a class it cannot
detect is exactly the silent-capability gap this codebase exists to prevent.
So the schema states only what the weights can actually do.

The seven omitted classes are not abandoned — the vocabulary is data-driven,
so annotating them and adding the name back here restores them. Their heights,
spoken names, hazard roles and keyword routes are all still defined below,
precisely so that re-adding a class is a one-line change.

  stairs_down  the critical class; needs top-of-staircase footage, which no
               public dataset supplies (Open Images photographs staircases
               from the bottom looking up). AV_CRITICAL still names it, so
               the "Warning! Stop and proceed carefully" path reactivates the
               moment the class is trained. Until then that path is DORMANT.
  pole         no boxes in the walkthrough footage or Open Images.
  sign_*       pictogram signs; they exist only in our own frames and need
               the Roboflow annotation pass.
"""

# Vocabulary retired from the trained schema, kept for the migration path.
# scripts/reindex_labels.py maps old label indices onto the list below BY NAME;
# removing a class shifts every index above it (dustbin 7 -> 5), so label files
# written against the 14-class list MUST be reindexed, never reused as-is.
AV_RETIRED = [
    "stairs_down", "pole", "sign_washroom", "sign_exit", "sign_lift",
    "sign_reception", "sign_wheelchair",
]

# --- 7 classes, index order MUST match data.yaml used for training ---
AV_CLASSES = [
    "person",          # 0
    "chair",           # 1
    "table",           # 2
    "door",            # 3
    "stairs_up",       # 4
    "dustbin",         # 5
    "signboard",       # 6   generic text sign (feeds OCR)
]

# What a HUMAN may annotate. Training narrows this to AV_CLASSES via
# scripts/reindex_labels.py, which maps by name and drops the rest. Annotation
# tools validate against this list, not AV_CLASSES: a labeller should be free to
# box a stairs_down the moment they see one, even though no model is trained on
# it yet. That box is what un-retires the class.
AV_ALL_CLASSES = AV_CLASSES + AV_RETIRED

# Objects that endanger a walking user -> announced first, with "Caution"
AV_HAZARDS = {"person", "stairs_up", "stairs_down", "pole", "chair",
              "table", "dustbin", "door"}

# The highest-urgency class: announced alone, interrupts everything
AV_CRITICAL = {"stairs_down"}

# Real-world heights (m) for the pinhole step-distance model
AV_HEIGHTS = {
    "person": 1.70, "chair": 0.90, "table": 0.75, "door": 2.00,
    "stairs_up": 1.00, "stairs_down": 1.00, "pole": 2.20, "dustbin": 0.70,
    "signboard": 0.35, "sign_washroom": 0.30, "sign_exit": 0.30,
    "sign_lift": 0.30, "sign_reception": 0.35, "sign_wheelchair": 0.30,
}

# Spoken names (sign_washroom -> "washroom sign")
AV_SPOKEN = {
    "stairs_up": "stairs going up",
    "stairs_down": "stairs going down",
    "sign_washroom": "washroom sign",
    "sign_exit": "exit sign",
    "sign_lift": "lift sign",
    "sign_reception": "reception sign",
    "sign_wheelchair": "wheelchair access sign",
    "signboard": "signboard",
    "dustbin": "dustbin",
}

# Voice keyword -> class, so "find X" can match a DETECTION rather than relying
# on reading the sign. Routes whose target is retired (every sign_*, plus pole)
# currently match nothing here and fall through to symbols.match_keyword's text
# path -- so those lookups are OCR-only today, and a pictogram with no legible
# text is honestly reported as not found. That is precisely the gap the trained
# sign classes exist to close, and re-adding a name to AV_CLASSES closes it
# without touching this table. tests/test_pipeline_logic.py pins the fallback:
# it is the ONLY path these keywords have, so it must not regress silently.
AV_KEYWORD_TO_CLASS = {
    "washroom": "sign_washroom", "restroom": "sign_washroom",
    "toilet": "sign_washroom", "bathroom": "sign_washroom",
    "exit": "sign_exit", "way out": "sign_exit", "emergency exit": "sign_exit",
    "lift": "sign_lift", "elevator": "sign_lift",
    "reception": "sign_reception", "front desk": "sign_reception",
    "wheelchair": "sign_wheelchair",
    "stairs": "stairs_up", "staircase": "stairs_up", "steps": "stairs_up",
    "door": "door", "chair": "chair", "table": "table",
    "dustbin": "dustbin", "bin": "dustbin", "pole": "pole",
}


# --- schema self-check ---------------------------------------------------
# Every table above is keyed by class NAME, so a typo or a rename produces a
# role that simply never fires: a hazard that is never announced, a height that
# never applies. That is the silent-capability failure this module argues
# against everywhere else, so it is checked at import. Refusing to start beats
# mis-announcing one class in a thousand frames.
def _check_names(label, names):
    unknown = sorted(set(names) - set(AV_ALL_CLASSES))
    if unknown:
        raise ValueError(
            f"{label} names {unknown}, which are not in AV_ALL_CLASSES. Add the "
            "class or fix the spelling: a role keyed to a name no class has can "
            "never fire, and nothing else would report it.")


_check_names("AV_HAZARDS", AV_HAZARDS)
_check_names("AV_CRITICAL", AV_CRITICAL)
_check_names("AV_HEIGHTS", AV_HEIGHTS)
_check_names("AV_SPOKEN", AV_SPOKEN)
_check_names("AV_KEYWORD_TO_CLASS targets", AV_KEYWORD_TO_CLASS.values())

if not AV_CRITICAL <= AV_HAZARDS:
    raise ValueError(
        f"AV_CRITICAL {sorted(AV_CRITICAL - AV_HAZARDS)} is not in AV_HAZARDS. "
        "priority.build_speech splits criticals out of the hazard list it is "
        "handed, so a critical that is not a hazard never reaches it.")

# Roles are declared over the ANNOTATION vocabulary on purpose: "stairs_down is
# dangerous" is a fact about the world, true whether or not a model can see one
# yet. What the DEPLOYED system can act on is the intersection with AV_CLASSES.
# Derived, never hand-maintained -- un-retiring a class must not also require
# remembering to update a second list.
TRAINED_HAZARDS = AV_HAZARDS & set(AV_CLASSES)

# Hazard roles no trained class can carry. The docstring calls this path
# DORMANT; this is that claim as a value the server can actually report.
DORMANT_HAZARDS = (AV_HAZARDS | AV_CRITICAL) - set(AV_CLASSES)

# False => priority.build_speech's "Warning! ... Stop and proceed carefully"
# branch is unreachable, because no trained class carries the critical role.
# Read by /health so the gap is observable at runtime, not just in this comment.
CRITICAL_ACTIVE = bool(AV_CRITICAL & set(AV_CLASSES))


def data_yaml(path: str = "../datasets/av7") -> str:
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(AV_CLASSES))
    return (f"path: {path}\ntrain: train/images\nval: valid/images\n"
            f"test: test/images\n\nnames:\n{names}\n")


if __name__ == "__main__":
    print(data_yaml())
