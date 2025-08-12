#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedACE – Minimal Concept Extraction (no SaT, no chunking)

Purpose
-------
Extract three concept families from DelPHEA v3 parquet clinical notes:
  1) ICI Drug Exposure
  2) AKI Mention
  3) AKI Attribution (clinician reasoning/stance)

Design
------
- YAGNI: no sentence segmentation / SaT, no chunking
- Soft-fail per note; continue the job
- Stable Parquet schemas; periodic checkpoints
- Quiet logs via ContextGem's logger env var
- Built-in `--sanity-check` runs on two mock notes to verify end-to-end

Environment / Logging
---------------------
Set CONTEXTGEM_LOGGER_LEVEL to control verbosity. Examples:
  export CONTEXTGEM_LOGGER_LEVEL=ERROR  # default, quiet
  export CONTEXTGEM_LOGGER_LEVEL=OFF    # fully silent from ContextGem

Per-note flow
-------------
  ┌───────────────────────────────────────────────────────────┐
  │ Load single note text                                     │
  ├───────────────────────────────────────────────────────────┤
  │ Build ContextGem Document (whole note, add_references=False)
  │   ├─ ICI Drug Exposure
  │   ├─ AKI Mention
  │   └─ AKI Attribution (stance + brief rationale)
  ├───────────────────────────────────────────────────────────┤
  │ LLM extract_all (single pass)                             │
  │   └─ on failure → record empty concepts (soft fail)       │
  ├───────────────────────────────────────────────────────────┤
  │ Post-process                                              │
  │   ├─ Normalize drug names; optional vocab filter (ICI)    │
  │   └─ Optional RxNorm lookup for missing rxcui             │
  ├───────────────────────────────────────────────────────────┤
  │ Append rows; checkpoint every N notes                     │
  └───────────────────────────────────────────────────────────┘

Outputs (Parquet)
-----------------
- out/drug_mentions.parquet
    patient_id, encounter_id, note_id, note_date, note_type,
    drug_text, normalized_name, canonical_generic, class,
    rxcui, dose_text, route_text, when_text

- out/note_concepts.parquet
    patient_id, encounter_id, note_id, note_date, note_type,
    n_drug_exposures, n_aki_mentions, n_attributions, concepts_json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

# quiet ContextGem before importing it
os.environ.setdefault("CONTEXTGEM_LOGGER_LEVEL", "ERROR")

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# import ContextGem AFTER setting logging env
from contextgem import (
    Document,
    DocumentLLM,
    JsonObjectConcept,
    JsonObjectExample,
    reload_logger_settings,
)
from tqdm import tqdm

from src.prompts import SYSTEM_IRAKI
from src.rxnorm import resolve_rxcui
from src.vocab import load_vocab, normalize_name

# ----------------------------- logging (quiet) ------------------------------


def _setup_logging(level: str = "ERROR") -> None:
    """Set Python logging level for local modules; ContextGem uses its own env."""
    lvl = getattr(logging, level.upper(), logging.ERROR)
    logging.basicConfig(level=lvl)
    for name in ("httpx", "urllib3"):
        logging.getLogger(name).setLevel(lvl)


# --------------------------- parquet / io helpers ---------------------------


def _normalize_model_id(m: str) -> str:
    """Ensure 'org/model' form to align with vLLM OpenAI router."""
    m = (m or "").strip()
    return m if "/" in m else f"openai/{m}"


def _read_v3_parquet_coerced(path: str) -> pd.DataFrame:
    """Read DelPHEA v3 parquet with schema coercion before pandas."""
    t = pq.read_table(path)
    target = pa.schema(
        [
            pa.field("SERVICE_NAME", pa.string()),
            pa.field("PHYSIOLOGIC_TIME", pa.timestamp("s")),
            pa.field("OBFUSCATED_GLOBAL_PERSON_ID", pa.string()),
            pa.field("ENCOUNTER_ID", pa.string()),
            pa.field("REPORT_TEXT", pa.string()),
        ]
    )
    try:
        t = t.cast(target, safe=False)
    except Exception:
        pass
    return t.to_pandas(types_mapper=pd.ArrowDtype)


def _to_int64_from_str(series: pd.Series) -> pd.Series:
    """Convert string/float-ish IDs to pandas Int64 safely."""

    def conv(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return pd.NA
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return pd.NA
        if "." in s and set(s.split(".", 1)[1]) <= {"0"}:
            s = s.split(".", 1)[0]
        try:
            if "e" in s.lower():
                return int(Decimal(s))
        except DecimalException:  # type: ignore[name-defined]
            pass
        except Exception:
            pass
        try:
            return int(s)
        except Exception:
            try:
                return int(float(s))
            except Exception:
                return pd.NA

    # handle DecimalException import lazily to avoid hard dependency
    from decimal import DecimalException  # noqa: E402

    return series.map(conv).astype("Int64")


def _clean_str(x: Any) -> str:
    """Normalize None/NaN/Nullish to empty string; strip whitespace."""
    s = "" if x is None else str(x)
    s = s.strip()
    if s.lower() in {"none", "nan", "null"}:
        return ""
    return s


def _load_notes_delphea_v3(path: str) -> pd.DataFrame:
    """Load and normalize DelPHEA v3 parquet for extraction."""
    base = os.path.basename(path)
    df = _read_v3_parquet_coerced(path).copy()

    df["note_type"] = df.get("SERVICE_NAME", "").astype(str).str.strip()
    df["note_datetime"] = pd.to_datetime(df.get("PHYSIOLOGIC_TIME"), errors="coerce")
    df["patient_id"] = _to_int64_from_str(df.get("OBFUSCATED_GLOBAL_PERSON_ID"))
    df["encounter_id"] = _to_int64_from_str(df.get("ENCOUNTER_ID"))

    txt = df.get("REPORT_TEXT", "")
    df["note_text"] = (
        pd.Series(txt, dtype="string")
        .fillna("")
        .apply(
            lambda s: ""
            if str(s).strip().lower() in {"none", "nan", "null"}
            else str(s)
        )
    )

    df = df.reset_index(drop=True)
    df["note_id"] = [f"{base}:{i:07d}" for i in range(len(df))]
    df["source_file"] = base

    keep = [
        "patient_id",
        "note_id",
        "encounter_id",
        "note_datetime",
        "note_type",
        "note_text",
        "source_file",
    ]
    return df[keep]


# --------------------------- sanity-check fixtures --------------------------


def _make_sanity_df() -> pd.DataFrame:
    """Create two mock notes: one positive (ICI+AKI+attribution), one negative."""
    data = [
        {
            "patient_id": 1,
            "note_id": "sanity:0000001",
            "encounter_id": 101,
            "note_datetime": pd.Timestamp("2025-07-24"),
            "note_type": "Oncology Progress Note",
            "note_text": (
                "Metastatic melanoma. Receiving pembrolizumab 200 mg IV every 3 weeks; "
                "last dose 2025-07-10. Creatinine rose from 1.0 to 2.1 on 2025-07-24 — "
                "acute kidney injury likely ICI-related given bland urinalysis and eosinophilia. "
                "Plan: hold pembrolizumab; start prednisone 1 mg/kg."
            ),
            "source_file": "sanity.parquet",
        },
        {
            "patient_id": 2,
            "note_id": "sanity:0000002",
            "encounter_id": 202,
            "note_datetime": pd.Timestamp("2025-07-24"),
            "note_type": "Hospitalist Progress Note",
            "note_text": (
                "Admitted for COPD exacerbation. Home meds include lisinopril. "
                "Creatinine stable at 0.9 mg/dL; no evidence of kidney injury. "
                "No immunotherapy or checkpoint inhibitor exposure documented."
            ),
            "source_file": "sanity.parquet",
        },
    ]
    df = pd.DataFrame(data)
    # enforce pandas nullable int for ids
    df["patient_id"] = df["patient_id"].astype("Int64")
    df["encounter_id"] = df["encounter_id"].astype("Int64")
    return df


# --------------------------- concepts (no references) -----------------------


def _examples_drug() -> List[JsonObjectExample]:
    return [
        JsonObjectExample(
            content={
                "drug_text": "Pembrolizumab 200 mg IV q3w given on 2025-07-10.",
                "normalized_name": "pembrolizumab",
                "dose": "200 mg",
                "route": "IV",
                "date": "2025-07-10",
                "relative_time": None,
                "rxcui": "1607602",
            }
        ),
        JsonObjectExample(
            content={
                "drug_text": "Started Opdivo last month",
                "normalized_name": "nivolumab",
                "dose": None,
                "route": None,
                "date": None,
                "relative_time": "last month",
                "rxcui": "1658841",
            }
        ),
    ]


def _examples_aki() -> List[JsonObjectExample]:
    return [
        JsonObjectExample(
            content={
                "sentence": "Acute kidney injury noted on 07/24.",
                "onset_date": "2025-07-24",
            }
        ),
        JsonObjectExample(content={"sentence": "AKI improving.", "onset_date": None}),
    ]


def _examples_attr() -> List[JsonObjectExample]:
    return [
        JsonObjectExample(
            content={
                "stance": "likely ICI-related",
                "drug_name": "pembrolizumab",
                "rationale": "Temporal association; bland UA; eosinophilia.",
            }
        ),
        JsonObjectExample(
            content={
                "stance": "unlikely ICI-related",
                "drug_name": None,
                "rationale": "Sepsis/ATN pattern without immune features.",
            }
        ),
        JsonObjectExample(
            content={
                "stance": "uncertain",
                "drug_name": "nivolumab",
                "rationale": "Plausible timing but concomitant nephrotoxins.",
            }
        ),
    ]


def _concepts_for_note(
    allowed_names_for_hint: Optional[List[str]] = None,
) -> List[JsonObjectConcept]:
    """Define minimal concepts without sentence references (no SaT)."""
    hint_list = (allowed_names_for_hint or [])[:30]
    allow_hint = (
        f" Only keep if the drug is an ICI (examples: {', '.join(hint_list)})."
        if hint_list
        else " Only keep if the drug is an ICI."
    )

    drug_concept = JsonObjectConcept(
        name="ICI Drug Exposure",
        description=(
            "Extract immune checkpoint inhibitor exposures with attributes. "
            "Return an empty list if none." + allow_hint
        ),
        structure={
            "drug_text": str,
            "normalized_name": str
            | None,  # <- relaxed to avoid hard validation failures
            "dose": str | None,
            "route": str | None,
            "date": str | None,
            "relative_time": str | None,
            "rxcui": str | None,
        },
        add_references=False,
        examples=_examples_drug(),
    )
    aki_concept = JsonObjectConcept(
        name="AKI Mention",
        description=(
            "Extract mentions of acute kidney injury and onset date/timeframe if present. "
            "Return an empty list if none."
        ),
        structure={"sentence": str, "onset_date": str | None},
        add_references=False,
        examples=_examples_aki(),
    )
    attrib_concept = JsonObjectConcept(
        name="AKI Attribution",
        description=(
            "Clinician stance whether AKI is ICI-related, ruled out, or uncertain, "
            "plus a brief rationale. Return an empty list if none."
        ),
        structure={"stance": str, "drug_name": str | None, "rationale": str | None},
        add_references=False,
        examples=_examples_attr(),
    )
    return [drug_concept, aki_concept, attrib_concept]


# ------------------------------- save helpers -------------------------------

DRUG_COLS = [
    "patient_id",
    "encounter_id",
    "note_id",
    "note_date",
    "note_type",
    "drug_text",
    "normalized_name",
    "canonical_generic",
    "class",
    "rxcui",
    "dose_text",
    "route_text",
    "when_text",
]
NOTE_COLS = [
    "patient_id",
    "encounter_id",
    "note_id",
    "note_date",
    "note_type",
    "n_drug_exposures",
    "n_aki_mentions",
    "n_attributions",
    "concepts_json",
]


def _atomic_write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write Parquet atomically (tmp→replace)."""
    tmp = f"{path}.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------- main ------------------------------------


def _parse_range(s: Optional[str], n: int) -> Tuple[int, int]:
    """Parse 'start:end' (end exclusive) with clamping to [0, n]."""
    if not s:
        return 0, n
    parts = s.split(":", 1)
    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if len(parts) > 1 and parts[1] else n
    start = max(0, min(start, n))
    end = max(start, min(end, n))
    return start, end


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(
        description="MedACE minimal extractor (no SaT, no chunking)"
    )
    ap.add_argument("--notes", help="path to DelPHEA v3 Parquet notes")
    ap.add_argument(
        "--sanity-check",
        action="store_true",
        help="run on two built-in dummy notes instead of parquet",
    )
    ap.add_argument("--outdir", default="out", help="output directory")
    ap.add_argument("--debug", action="store_true", help="process first 10 notes only")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N notes")
    ap.add_argument(
        "--range", type=str, default=None, help="slice 'START:END' (END exclusive)"
    )
    ap.add_argument("--backend-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--drug-vocab", default="resources/ici_vocab.csv")
    ap.add_argument("--filter-ici", action="store_true", help="keep only ICI exposures")
    ap.add_argument(
        "--rxnorm-online",
        action="store_true",
        help="resolve RxCUI via RxNav if missing",
    )
    ap.add_argument(
        "--checkpoint-every", type=int, default=50, help="write partials every N notes"
    )
    ap.add_argument(
        "--log-level",
        type=str,
        default="ERROR",
        help="python logging level for our code",
    )
    ap.add_argument(
        "--cg-log-level",
        type=str,
        default=None,
        help="ContextGem logger level: TRACE|DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL|OFF",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--max-items-per-call", type=int, default=1)
    args = ap.parse_args()

    if not args.sanity_check and not args.notes:
        raise SystemExit("--notes is required unless --sanity-check is used")

    # logging
    _setup_logging(args.log_level)
    if args.cg_log_level:
        os.environ["CONTEXTGEM_LOGGER_LEVEL"] = args.cg_log_level
        reload_logger_settings()

    os.makedirs(args.outdir, exist_ok=True)

    # source dataframe
    if args.sanity_check:
        df = _make_sanity_df()
    else:
        df = _load_notes_delphea_v3(args.notes)  # type: ignore[arg-type]

    if args.offset > 0:
        df = df.iloc[args.offset :].reset_index(drop=True)
    if args.debug:
        df = df.head(10).copy()
    start, end = _parse_range(args.range, len(df))
    if start or end != len(df):
        df = df.iloc[start:end].reset_index(drop=True)

    # vocab (optional)
    vocab_path = (
        args.drug_vocab
        if (args.drug_vocab and os.path.exists(args.drug_vocab))
        else None
    )
    vocab = load_vocab(vocab_path) if vocab_path else None
    allowed_keys = sorted(set(vocab["name_to_canonical"].keys())) if vocab else []

    # LLM
    llm = DocumentLLM(
        model=_normalize_model_id(args.model),
        api_base=args.backend_url,
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        system_message=SYSTEM_IRAKI,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=90,
        max_tokens=args.max_tokens,
        num_retries_failed_request=3,
        max_retries_invalid_data=2,
        output_language="en",
    )

    drug_rows: List[Dict[str, Any]] = []
    note_rows: List[Dict[str, Any]] = []

    pbar = tqdm(df.iterrows(), total=len(df), desc="notes")
    for i, (_, row) in enumerate(pbar, start=1):
        text = _clean_str(row.get("note_text"))

        # empty-note fast path
        if not text:
            concept_payload = {
                "drug_exposures": [],
                "aki_mentions": [],
                "attributions": [],
            }
            note_rows.append(
                {
                    "patient_id": row.get("patient_id"),
                    "encounter_id": row.get("encounter_id"),
                    "note_id": row.get("note_id"),
                    "note_date": row.get("note_datetime"),
                    "note_type": row.get("note_type"),
                    "n_drug_exposures": 0,
                    "n_aki_mentions": 0,
                    "n_attributions": 0,
                    "concepts_json": json.dumps(concept_payload, ensure_ascii=False),
                }
            )
            if args.checkpoint_every and (i % args.checkpoint_every == 0):
                _atomic_write_parquet(
                    pd.DataFrame(drug_rows, columns=DRUG_COLS),
                    os.path.join(args.outdir, "drug_mentions.partial.parquet"),
                )
                _atomic_write_parquet(
                    pd.DataFrame(note_rows, columns=NOTE_COLS),
                    os.path.join(args.outdir, "note_concepts.partial.parquet"),
                )
                pbar.set_postfix_str(f"checkpoint@{i}")
            continue

        # concepts (no references)
        c_drug, c_aki, c_attr = _concepts_for_note(allowed_names_for_hint=allowed_keys)
        doc = Document(raw_text=text)
        doc.add_concepts([c_drug, c_aki, c_attr])

        # extract; soft-fail per note to keep pipeline running
        try:
            doc = llm.extract_all(
                doc,
                overwrite_existing=True,
                max_items_per_call=args.max_items_per_call,
                raise_exception_on_extraction_error=True,
            )
        except Exception:
            c_drug.extracted_items = []
            c_aki.extracted_items = []
            c_attr.extracted_items = []

        # flatten concept items
        def _flatten(concept: JsonObjectConcept) -> List[dict]:
            out: List[dict] = []
            for it in getattr(concept, "extracted_items", []) or []:
                val = getattr(it, "value", None)
                if isinstance(val, dict):
                    out.append(val)
            return out

        drug_items = _flatten(c_drug)
        aki_items = _flatten(c_aki)
        attr_items = _flatten(c_attr)

        # post-process drugs (normalize, filter, optional RxNorm)
        cleaned_drugs: List[Dict[str, Any]] = []
        for de in drug_items:
            dn_raw = _clean_str(de.get("normalized_name") or de.get("drug_text"))
            key = normalize_name(dn_raw)
            canon = vocab["name_to_canonical"].get(key) if vocab else None
            if (
                args.filter_ici
                and vocab
                and (key not in vocab["name_to_canonical"])
                and not canon
            ):
                continue

            rxcui = _clean_str(de.get("rxcui"))
            if not rxcui and args.rxnorm_online and dn_raw:
                try:
                    rxcui = resolve_rxcui(dn_raw) or ""
                except Exception:
                    rxcui = ""

            cleaned_drugs.append(
                {
                    "patient_id": row.get("patient_id"),
                    "encounter_id": row.get("encounter_id"),
                    "note_id": row.get("note_id"),
                    "note_date": row.get("note_datetime"),
                    "note_type": row.get("note_type"),
                    "drug_text": de.get("drug_text"),
                    "normalized_name": dn_raw or None,
                    "canonical_generic": canon,
                    "class": (
                        (vocab["by_name"].get(key, {}) or {}).get("class")
                        if vocab
                        else None
                    ),
                    "rxcui": rxcui or None,
                    "dose_text": _clean_str(de.get("dose")) or None,
                    "route_text": _clean_str(de.get("route")) or None,
                    "when_text": _clean_str(de.get("date") or de.get("relative_time"))
                    or None,
                }
            )

        drug_rows.extend(cleaned_drugs)
        concept_payload = {
            "drug_exposures": cleaned_drugs,
            "aki_mentions": aki_items,
            "attributions": attr_items,
        }
        note_rows.append(
            {
                "patient_id": row.get("patient_id"),
                "encounter_id": row.get("encounter_id"),
                "note_id": row.get("note_id"),
                "note_date": row.get("note_datetime"),
                "note_type": row.get("note_type"),
                "n_drug_exposures": len(cleaned_drugs),
                "n_aki_mentions": len(aki_items),
                "n_attributions": len(attr_items),
                "concepts_json": json.dumps(concept_payload, ensure_ascii=False),
            }
        )

        # checkpoints
        if args.checkpoint_every and (i % args.checkpoint_every == 0):
            _atomic_write_parquet(
                pd.DataFrame(drug_rows, columns=DRUG_COLS),
                os.path.join(args.outdir, "drug_mentions.partial.parquet"),
            )
            _atomic_write_parquet(
                pd.DataFrame(note_rows, columns=NOTE_COLS),
                os.path.join(args.outdir, "note_concepts.partial.parquet"),
            )
            pbar.set_postfix_str(f"checkpoint@{i}")

    # final write
    _atomic_write_parquet(
        pd.DataFrame(drug_rows, columns=DRUG_COLS),
        os.path.join(args.outdir, "drug_mentions.parquet"),
    )
    _atomic_write_parquet(
        pd.DataFrame(note_rows, columns=NOTE_COLS),
        os.path.join(args.outdir, "note_concepts.parquet"),
    )
    print(
        f"[OK] wrote {len(drug_rows)} drug rows and {len(note_rows)} note rows to {args.outdir}"
    )


if __name__ == "__main__":
    main()
