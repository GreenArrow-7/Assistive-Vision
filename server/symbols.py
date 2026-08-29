"""Symbol recognition + keyword utilities.

Symbols are resolved from (a) detected object classes (e.g. COCO 'toilet'),
and (b) OCR keywords found on signboards. A custom-trained YOLO symbol model
can be dropped in later without changing this interface.
"""
import re
from . import config
from .classes_av import AV_KEYWORD_TO_CLASS


def mentions(haystack: str, needle: str) -> bool:
    """Whole-word containment.

    Plain substring matching sent 'find washroom' to a MENU board ('men' is a
    substring of 'MENU'/'WOMEN') and matched 'exit' against 'EXITED'.
    """
    if not haystack or not needle:
        return False
    return re.search(rf"\b{re.escape(needle)}\b", haystack, re.IGNORECASE) is not None


def clean_query(raw: str) -> str:
    """'where is the exit' -> 'exit'."""
    words = re.findall(r"[a-z0-9]+", (raw or "").lower())
    kept = [w for w in words if w not in config.QUERY_STOPWORDS]
    return " ".join(kept).strip()


def symbols_from_objects(objects):
    out = []
    for o in objects:
        sym = config.SYMBOL_FROM_CLASS.get(o["label"])
        if sym:
            out.append({**o, "label": sym, "kind": "symbol"})
    return out


def symbols_from_texts(texts):
    out = []
    for t in texts:
        low = t["label"].lower()
        for sym, keys in config.SYMBOL_KEYWORDS.items():
            if any(mentions(low, k) for k in keys):
                out.append({**t, "label": sym, "kind": "symbol"})
                break
    return out


def match_keyword(keyword: str, texts, symbols, objects):
    """Find best item matching the user's keyword.

    Priority: trained sign class (most reliable) > text > symbol > object.
    """
    if not keyword:
        return None
    k = keyword.lower()

    # trained-model path: "washroom" -> class sign_washroom
    target = AV_KEYWORD_TO_CLASS.get(k)
    if target:
        for pool in (objects, symbols, texts):
            for it in pool:
                if it.get("raw") == target:
                    return it

    keys = config.SYMBOL_KEYWORDS.get(k, []) + [k]
    for pool in (texts, symbols, objects):
        for it in pool:
            low = it["label"].lower()
            if any(mentions(low, key) or mentions(key, low) for key in keys):
                return it
    return None
