"""Add a weight file to the model registry, identified by what is inside it.

Twice now a model has been misidentified by its filename: two files both named
`av_obstacle_candidate.pt` differed by one class and 1,088 bytes, and cost a day
of searching (ARCHITECTURE_RULINGS R-002). So the registry key is derived from
the class names and the training date read out of the checkpoint, and the file
is copied in under a fixed name. Nothing here trusts the path it was given.

    python scripts/register_model.py models/av7_candidate.pt --status production

Writes models/registry/<model_id>/{model.pt,metadata.json,README.md}.
Metrics are not invented: they are pulled from an evaluation JSON if one is
passed with --metrics, and otherwise left null with a note saying so.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

REGISTRY = os.path.join("models", "registry")


def inspect(path):
    """Read identity out of the checkpoint. Never out of the filename."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ck.get("model")
    names = getattr(model, "names", None)
    if not names:
        sys.exit(f"{path}: no class names in the checkpoint; refusing to register it")
    args = ck.get("train_args") or {}
    blob = open(path, "rb").read()
    return {
        "class_names": [names[i] for i in sorted(names)],
        "trained_at": str(ck.get("date")),
        "trainer_version": ck.get("version"),
        "train_config": {k: args.get(k) for k in
                         ("model", "data", "epochs", "patience", "batch",
                          "imgsz", "optimizer", "lr0", "seed")},
        "sha256": hashlib.sha256(blob).hexdigest(),
        "md5": hashlib.md5(blob).hexdigest(),
        "size_bytes": len(blob),
    }


def model_id(info, family):
    day = (info["trained_at"] or "unknown")[:10].replace("-", "")
    return f"{family}-av{len(info['class_names'])}-{day}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--family", default="assistive")
    ap.add_argument("--status", default="candidate",
                    choices=["candidate", "production", "superseded", "experimental"])
    ap.add_argument("--dataset-id", default=None)
    ap.add_argument("--metrics", default=None, help="evaluation JSON to copy in")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    info = inspect(a.weights)
    mid = model_id(info, a.family)
    out = os.path.join(REGISTRY, mid)
    os.makedirs(os.path.join(out, "evaluation"), exist_ok=True)

    dest = os.path.join(out, "model.pt")
    if not os.path.exists(dest) or hashlib.sha256(open(dest, "rb").read()).hexdigest() != info["sha256"]:
        shutil.copy2(a.weights, dest)

    metrics = None
    if a.metrics:
        metrics = json.load(open(a.metrics))
        json.dump(metrics, open(os.path.join(out, "evaluation", "metrics.json"), "w"), indent=1)

    meta = {
        "model_id": mid,
        "status": a.status,
        "registered_from": a.weights,
        "dataset_id": a.dataset_id,
        # Ultralytics YOLOv8 derivative: AGPL-3.0 applies to anything shipped
        # from it. Unresolved release blocker, see ARCHITECTURE_RULINGS R-006.
        "licensing": {"base": "Ultralytics YOLOv8", "license": "AGPL-3.0",
                      "redistribution_cleared": False},
        "metrics": metrics,
        "notes": a.notes,
        **info,
    }
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"), indent=1)

    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(f"""# {mid}

**Status** {a.status} | **sha256** `{info['sha256']}` | **{info['size_bytes']:,} bytes**

Classes ({len(info['class_names'])}): {', '.join(info['class_names'])}

Trained {info['trained_at']} on `{meta['dataset_id']}`,
{info['train_config'].get('epochs')} epochs at imgsz {info['train_config'].get('imgsz')},
Ultralytics {info['trainer_version']}.

Full provenance, hashes and metrics: [metadata.json](metadata.json).
Measured comparison against the other providers:
[docs/MODEL_EVALUATION.md](../../../docs/MODEL_EVALUATION.md).

Licensing: AGPL-3.0 via Ultralytics YOLOv8. Redistribution **not** cleared.

{a.notes}
""")
    print(mid, info["sha256"][:12], ",".join(info["class_names"]))


if __name__ == "__main__":
    main()
