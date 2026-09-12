"""Priority engine.

Builds the final spoken message in strict priority order per the project spec:
  1. Immediate hazards (obstacles, vehicles, people very close)
  2. Keyword-match result (the thing the user asked for)
  3. Recognized navigation symbols
  4. Environment summary (texts + objects)
"""
from . import config
from .environment import context_hint, select_items
from .classes_av import AV_CRITICAL


def is_critical(item):
    # Approximate monocular proximity is an alert cue, never proof of collision.
    if item.get("conf", 0) < .8:
        return False
    return (item.get("raw") in {"stairs_up", "stairs_down"} or
            (item.get("steps") is not None and item["steps"] <= 2))


def _fmt(item) -> str:
    label = item.get("label", "unknown")
    d = item.get("direction", "ahead")
    steps = item.get("steps")
    if steps:
        if steps > config.MAX_STEPS_ANNOUNCE:
            return f"{label} far away {d}"
        from .spatial import steps_phrase
        urgency = ", very close" if steps <= 2 else ""
        return f"{label}, about {steps_phrase(steps)} step{'s' if steps != 1 else ''} {d}{urgency}"
    prox = item.get("proximity", "")
    prox = f", {prox}" if prox and prox != "at a distance" else ""
    return f"{label} {d}{prox}"


def build_speech(hazards, objects, texts, symbols, keyword=None, match=None) -> dict:
    """Return dict with ordered speech string + structured sections."""
    parts = []

    # 0 — CRITICAL (descending stairs): announced alone, first, emphatic
    critical = [h for h in hazards if h.get("raw") in AV_CRITICAL or is_critical(h)]
    if critical:
        c = min(critical, key=lambda x: x.get("steps") or 99)
        parts.append("Warning! " + _fmt(c) + ". Stop and proceed carefully.")

    # 1 — hazards, closest first (steps when known, else proximity)
    rest_hz = [h for h in hazards if h not in critical]
    hz = sorted(rest_hz, key=lambda h: (h.get("steps") or 99))
    hz = hz[: config.MAX_ANNOUNCE]
    if hz:
        parts.append("Caution. " + ". ".join(_fmt(h) for h in hz) + ".")

    # 2 — keyword result
    if keyword:
        if match:
            prox = match.get("proximity", "")
            prox = f", {prox}" if prox else ""
            label = str(match.get("label", keyword))
            if label.lower().strip() == keyword.lower().strip():
                parts.append(f"{keyword} is {match['direction']}{prox}.")
            else:
                parts.append(f"{keyword} found: {label} is {match['direction']}{prox}.")
        else:
            parts.append(f"{keyword} not found in the current view.")

    # 3 — symbols (skip any that duplicate the match)
    def _dup(s):
        return match is not None and (
            s is match or (s.get("box") is not None and s.get("box") == match.get("box")) or str(s.get("label", "")).lower() == str(match.get("label", "")).lower()
        )
    sym = select_items([s for s in symbols if not _dup(s)], config.MAX_ANNOUNCE)
    if sym:
        parts.append("Signs: " + ", ".join(_fmt(s) for s in sym) + ".")

    # 4 — environment summary
    if not keyword:
        rest = [t for t in texts if not (match and t is match)]
        rest += objects   # objects and hazards are disjoint by construction
        rest = [r for r in rest if not any(r.get("box") == s.get("box") for s in sym)]
        rest = select_items(rest, config.MAX_ANNOUNCE)
        hint = context_hint(texts)
        if hint: parts.append(hint)
        if rest:
            parts.append("I can see: " + ", ".join(_fmt(r) for r in rest) + ".")

    if not parts:
        parts.append("No reliable text or objects identified in this view.")

    # critical hazards MUST be counted: the client uses hazard_count > 0 to
    # vibrate, flash and interrupt speech — a lone stairs_down reported 0.
    return {"speech": " ".join(parts), "hazard_count": len(hazards)}
