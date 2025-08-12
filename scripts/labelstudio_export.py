from __future__ import annotations

import json
import os

import pandas as pd

# convert probabilistic cases to labelstudio import json


def to_labelstudio(df_prob: pd.DataFrame, out_path: str) -> None:
    tasks = []
    for _, r in df_prob.iterrows():
        data = {
            "text": r.get("evidence_sentences", ""),
            "meta": {
                "patient_id": r.get("patient_id"),
                "note_id": r.get("note_id"),
                "note_date": str(r.get("note_date")),
                "drug_name": r.get("drug_name"),
                "best_relation": r.get("best_relation"),
                "evidence_level": r.get("evidence_level"),
                "rationale": r.get("rationale"),
            },
        }
        tasks.append({"data": data})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(tasks, f, indent=2)
