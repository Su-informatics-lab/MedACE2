from __future__ import annotations

import os

import pandas as pd

# io helpers for notes


def load_notes(path: str, file_format: str) -> pd.DataFrame:
    if file_format == "parquet":
        df = pd.read_parquet(path)
    elif file_format == "csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("file_format must be 'parquet' or 'csv'")
    req = ["patient_id", "note_id", "note_datetime", "note_text"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"missing required column: {c}")
    if "encounter_id" not in df.columns:
        df["encounter_id"] = None
    if "note_type" not in df.columns:
        df["note_type"] = None
    df["note_datetime"] = pd.to_datetime(df["note_datetime"], errors="coerce")
    return df


def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
