"""Object + obstacle detection using YOLOv8.

Output items: {label, box, conf, kind='object'|'hazard'}
A detection is a hazard if its class is in the active schema's hazard set AND
it is close (proximity computed later); the raw split here is by class only —
the priority engine re-ranks by proximity.
"""
from . import config
from .classes_av import AV_ALL_CLASSES, AV_CLASSES, AV_HAZARDS, AV_SPOKEN

# Which class vocabulary a loaded model speaks. Everything downstream —
# hazard set, spoken names, class filtering — branches on this.
# Derived from the vocabulary, never written out as a literal: a hand-typed
# "av7" survived the schema shrinking to six classes and would have told
# /health, the web app and every report a class count the weights no longer
# have. The id cannot outlive the list it names.
SCHEMA_AV = f"av{len(AV_CLASSES)}"
SCHEMA_COCO = "coco"


def detect_schema(names) -> str:
    """Identify a model by its CLASS NAMES, never by its filename.

    This used to be inferred from the weights path: a file present at
    OBJECT_MODEL_CUSTOM was assumed to be the AV-6 fine-tune. A COCO-class
    model dropped there therefore switched hazard lookup to AV_HAZARDS, which
    contains no vehicles — 'car', 'bus', 'truck', 'bicycle' and 'motorcycle'
    all silently stopped being hazards, with no error and no log line.

    Schemas are now derived from what the model actually predicts, so the two
    vocabularies can never be crossed. An unrecognised schema is refused
    outright rather than guessed at: for a blind user, a detector whose hazard
    set we cannot establish is worse than no detector at all.
    """
    labels = {str(v) for v in names.values()}
    # Any model over our own vocabulary, as long as it carries the trained core.
    # Pinning this to exactly AV_CLASSES refused the AV-7 weights -- which are
    # the six plus stairs_down -- so un-retiring a class could not be deployed
    # without editing this function. The id still counts the model's OWN labels,
    # so it can never claim a class the weights do not have.
    if labels <= set(AV_ALL_CLASSES) and set(AV_CLASSES) <= labels:
        return f"av{len(labels)}"
    if config.HAZARD_CLASSES <= labels:      # COCO, or any superset of it
        return SCHEMA_COCO
    missing = sorted(config.HAZARD_CLASSES - labels)
    raise ValueError(
        f"Unrecognised detector class schema ({len(labels)} classes). A model "
        f"must expose either the {len(AV_CLASSES)} AV classes or all COCO "
        f"hazard classes; this one is missing {missing[:6]}"
        f"{' …' if len(missing) > 6 else ''}. Refusing to load — an unknown "
        "schema would silently disable hazard alerts."
    )


def active_schema():
    """Schema in force for hazard lookup: the assistive provider's if loaded.

    With several providers there is no single schema any more, so this reports
    the one whose vocabulary names the indoor hazards; /health lists them all.
    """
    from . import perception
    loaded = [p for p in perception.providers() if p.schema]
    for p in loaded:
        if p.schema != SCHEMA_COCO:
            return p.schema
    return loaded[0].schema if loaded else None


def detect_objects(frame_bgr):
    """One fused detection stream from every configured provider.

    The per-model work -- class filtering, confidence gates, spoken names --
    moved to perception.Provider; what is left here is the split the rest of the
    server expects, hazards from everything else.
    """
    from . import perception
    objects, hazards = [], []
    for det in perception.observe(frame_bgr):
        item = {k: det[k] for k in ("label", "raw", "box", "conf")}
        item.update(kind="object", provider=det["provider_id"], model=det["model_id"],
                    canonical=det["canonical"])
        if det.get("agreed_by"):
            item["agreed_by"] = det["agreed_by"]
        if det["hazard"]:
            hazards.append({**item, "kind": "hazard"})
        else:
            objects.append(item)
    return objects, hazards
