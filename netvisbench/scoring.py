"""Answer extraction and exact-match scoring.

Models are asked to finish with a line ``ANSWER: <value>``. We take the
last such line; if a model forgets it, we fall back to the last value of
the right type in its reply (and record that a fallback was used).
"""
from __future__ import annotations

import re

_IP = r"\d{1,3}(?:\.\d{1,3}){3}"


def _clean(s: str) -> str:
    s = s.strip().strip("*`_ ").strip()
    s = re.sub(r"^[\(\[]|[\)\]]$", "", s)
    return s.rstrip(".,;:!").strip()


def _typed(text: str, atype: str) -> str | None:
    if atype == "int":
        m = re.findall(r"-?\d+", text.replace(",", ""))
        return str(int(m[-1])) if m else None
    if atype == "choice":
        m = re.findall(r"\b([AB])\b", text)
        return m[-1] if m else None
    # label: host names (H3), IP addresses, or a single router letter
    m = re.findall(rf"({_IP}|\bH\d+\b|\b[A-Za-z]\b)", text, flags=re.IGNORECASE)
    return m[-1].upper() if m else None


def extract_answer(raw: str | None, atype: str) -> str | None:
    if not raw:
        return None
    tagged = re.findall(r"answer\s*[:：]\s*(.+)", raw, flags=re.IGNORECASE)
    if tagged:
        val = _typed(_clean(tagged[-1]), atype)
        if val is not None:
            return val
    return _typed(raw, atype)


def used_tag(raw: str | None) -> bool:
    return bool(raw) and bool(re.search(r"answer\s*[:：]", raw, flags=re.IGNORECASE))


def normalise(value: str, atype: str) -> str:
    value = value.strip()
    if atype == "int":
        try:
            return str(int(value))
        except ValueError:
            return value
    return value.upper()


def is_correct(pred: str | None, gold: str, atype: str) -> bool:
    return pred is not None and normalise(pred, atype) == normalise(gold, atype)
