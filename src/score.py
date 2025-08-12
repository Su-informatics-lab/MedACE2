from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

# simple phrase lists for high‑confidence cues (expandable)
HIGH_POSITIVE = [
    "ici-related nephritis",
    "immune-related nephritis",
    "aki due to",
    "drug-induced interstitial nephritis",
    "checkpoint inhibitor nephritis",
    "immune-mediated aki",
    "iraki",
]
HIGH_NEGATED = [
    "not immune related",
    "not irae",
    "not due to ici",
    "unlikely immune-mediated",
]
SUPPORTIVE = [
    "concern for",
    "suspect",
    "possible",
    "likely",
    "consider",
    "consistent with",
]

# label a single note using two‑level evidence rules


def label_note_evidence(
    note_row: Dict[str, Any],
    attributions: List[Dict[str, Any]],
    drug_dates: List[pd.Timestamp],
    aki_onset: Optional[pd.Timestamp],
    window_days: int = 60,
    has_alt_cause: bool = False,
) -> Dict[str, Any]:
    for att in attributions or []:
        rel = (att.get("relation") or "").lower()
        level = (att.get("evidence_level") or "").lower()
        if rel in {"caused_by", "ruled_out"}:
            return {"best_relation": rel, "evidence_level": "high"}
    text = (note_row.get("note_text") or "").lower()
    if any(p in text for p in HIGH_POSITIVE):
        return {"best_relation": "caused_by", "evidence_level": "high"}
    if any(n in text for n in HIGH_NEGATED):
        return {"best_relation": "ruled_out", "evidence_level": "high"}
    if aki_onset is None or not drug_dates:
        return {"best_relation": "none", "evidence_level": "none"}
    if any(abs((aki_onset - d).days) <= window_days for d in drug_dates):
        if not has_alt_cause and any(s in text for s in SUPPORTIVE):
            return {"best_relation": "possible", "evidence_level": "prob"}
    return {"best_relation": "none", "evidence_level": "none"}
