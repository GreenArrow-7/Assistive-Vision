"""Short evidence-based context hints, never a declaration of location."""
from .symbols import mentions, normalize

def context_hint(texts):
    words = normalize(" ".join(t["label"] for t in texts))
    clues = {
        "a healthcare setting": {"pharmacy", "emergency", "radiology", "patient"},
        "an educational setting": {"classroom", "library", "lecture", "laboratory"},
        "a shopping area": {"checkout", "food court", "shop", "sale"},
    }
    for context, keys in clues.items():
        if sum(mentions(words, k) for k in keys) >= 2:
            return f"The signs suggest {context}."
    return ""

def select_items(items, limit):
    out, seen = [], set()
    for item in sorted(items, key=lambda x: (x.get("steps") or 999, -x.get("conf", 0))):
        key = (normalize(item["label"]), item.get("direction"))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out
