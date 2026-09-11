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


def normalize(text):
    import unicodedata
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def safe_match(text, query):
    """Whole-token phrase matching, one OCR edit only on long nonnumeric words."""
    a, b = normalize(text).split(), normalize(query).split()
    if not b:
        return False
    def token(x, y):
        if x == y:
            return True
        if min(len(x), len(y)) < 5 or any(c.isdigit() for c in x+y):
            return False
        # One insertion, deletion or substitution; never truncate whole phrases.
        if abs(len(x)-len(y)) > 1:
            return False
        if len(x) == len(y):
            return sum(i != j for i, j in zip(x, y)) <= 1
        if len(x) > len(y):
            x, y = y, x
        return any(y[:i] + y[i+1:] == x for i in range(len(y)))
    return any(all(token(x, y) for x, y in zip(a[i:i+len(b)], b))
               for i in range(len(a)-len(b)+1))


def match_keyword(keyword: str, texts, symbols, objects):
    if not keyword:
        return None
    k = normalize(keyword)
    target = AV_KEYWORD_TO_CLASS.get(k)
    keys = config.SYMBOL_KEYWORDS.get(k, []) + [k]
    candidates = []
    for pool in (texts, symbols, objects):
        for it in pool:
            if (target and it.get("raw") == target) or any(safe_match(it["label"], key) for key in keys):
                candidates.append(it)
    # Nearest relevant result, then confidence. Unknown distances sort last.
    return min(candidates, key=lambda i: (i.get("steps") or 999, -i.get("conf", 0))) if candidates else None
