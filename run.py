#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedACE | Minimal concept extraction over clinical notes using ContextGem.

What it does (per note):
    raw note text
        ↓
    ContextGem.Document + three JsonObjectConcepts
        ↓
    LLM extraction (strict JSON validation; fail fast on invalid)
        ↓
    flatten concept items → rows
        ↓
    write Parquet tables

ASCII flow (per note)
---------------------
+-------------------+
| Raw note (string) |
+---------+---------+
          |
          v
+------------------------+
| ContextGem.Document    |
|  + ICI Drug Exposure   |
|  + AKI Mention         |
|  + AKI Attribution     |
+-----------+------------+
            |
            v
+----------------------------+
| LLM extract_all()          |
|  (strict JSON; retries)    |
+-----------+----------------+
            |
            v
+----------------------------+
| Flatten to rows            |
|  drug_exposures[]          |
|  aki_mentions[]            |
|  attributions[]            |
+-----------+----------------+
            |
            v
+----------------------------+
| Save Parquet:              |
|  out/drug_mentions.parquet |
|  out/note_concepts.parquet |
+----------------------------+

Notes:
- By default we DO NOT override ContextGem’s built-in system prompt. This usually
  yields better JSON adherence for open-source models served via vLLM.
- Use `--use-system-steering` to enable your long clinical system message if you
  truly need it.
- Use env var `CONTEXTGEM_LOGGER_LEVEL=INFO|WARNING|ERROR|OFF` to control library logs.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from contextgem import Document, DocumentLLM, JsonObjectConcept

# Optional clinical steering (off by default)
try:
    from src.prompts import SYSTEM_IRAKI
except Exception:  # pragma: no cover
    SYSTEM_IRAKI = None


# --------------------------- IO / parquet helpers ---------------------------


def _read_delphea_v3(path: str) -> pd.DataFrame:
    """Read DelPHEA v3 parquet (schema: SERVICE_NAME, PHYSIOLOGIC_TIME,
    OBFUSCATED_GLOBAL_PERSON_ID, ENCOUNTER_ID, REPORT_TEXT). Cast to sensible types.
    """
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
    df = t.to_pandas(types_mapper=pd.ArrowDtype)

    base = os.path.basename(path)
    df = df.rename(
        columns={
            "SERVICE_NAME": "note_type",
            "PHYSIOLOGIC_TIME": "note_datetime",
            "OBFUSCATED_GLOBAL_PERSON_ID": "patient_id",
            "ENCOUNTER_ID": "encounter_id",
            "REPORT_TEXT": "note_text",
        }
    )
    df["note_text"] = (
        pd.Series(df.get("note_text", ""), dtype="string")
        .fillna("")
        .map(
            lambda s: ""
            if str(s).strip().lower() in {"none", "nan", "null"}
            else str(s)
        )
    )
    df = df.reset_index(drop=True)
    df["note_id"] = [f"{base}:{i:07d}" for i in range(len(df))]
    keep = [
        "patient_id",
        "encounter_id",
        "note_id",
        "note_datetime",
        "note_type",
        "note_text",
    ]
    return df[keep]


def _atomic_write_parquet(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ------------------------- Concepts (short + literal) ------------------------

ICI_WHITELIST = (
    "pembrolizumab, nivolumab, ipilimumab, atezolizumab, durvalumab, "
    "avelumab, cemiplimab, dostarlimab, tremelimumab"
)


def _concepts() -> List[JsonObjectConcept]:
    """Define three concepts with concise instructions + strict structures."""
    drug_concept = JsonObjectConcept(
        name="ICI Drug Exposure",
        description=(
            "Extract mentions of immune checkpoint inhibitors (PD-1, PD-L1, CTLA-4). "
            "Only return items if the drug is an ICI. Examples include: "
            f"{ICI_WHITELIST}. "
            "Capture the raw text and any dose/route/when if stated."
        ),
        structure={
            "drug_text": str,
            "normalized_name": str | None,
            "dose": str | None,
            "route": str | None,
            "date": str | None,
            "relative_time": str | None,
            "sentence": str | None,
            "rxcui": str | None,
        },
        add_references=False,
    )

    aki_concept = JsonObjectConcept(
        name="AKI Mention",
        description=(
            "Extract sentences that indicate acute kidney injury (AKI). "
            "Keywords may include 'AKI', 'acute kidney injury', 'rising creatinine'. "
            "If an explicit onset date/timeframe is stated, include it."
        ),
        structure={"sentence": str, "onset_date": str | None},
        add_references=False,
    )

    attr_concept = JsonObjectConcept(
        name="AKI Attribution",
        description=(
            "Extract explicit clinician statements about whether AKI is caused by ICI, "
            "ruled out, or uncertain; include the cited drug name if mentioned and a brief rationale."
        ),
        structure={
            "stance": str,  # 'caused' | 'ruled_out' | 'uncertain'
            "drug_name": str | None,
            "sentences": list[str],
            "rationale": str | None,
        },
        add_references=False,
    )
    return [drug_concept, aki_concept, attr_concept]


def _flatten_items(concept: JsonObjectConcept) -> List[dict]:
    """Extract clean dicts from ContextGem items (no references used)."""
    out: List[dict] = []
    for it in getattr(concept, "extracted_items", []) or []:
        val = getattr(it, "value", None)
        if isinstance(val, dict):
            out.append(val)
        elif isinstance(val, str):
            out.append({"sentence": val})
    return out


# ------------------------------ LLM utilities --------------------------------


def _make_llm(args: argparse.Namespace) -> DocumentLLM:
    """Create a DocumentLLM; use ContextGem default system prompt by default."""
    sys_msg = None
    if getattr(args, "use_system_steering", False) and SYSTEM_IRAKI:
        # keep it small if present
        sys_msg = SYSTEM_IRAKI[:2000]

    return DocumentLLM(
        model=args.model,
        api_base=args.backend_url,
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        system_message=sys_msg,  # None => use ContextGem’s built-in extractor prompt
        temperature=0.0,
        top_p=0.2,
        timeout=120,
        max_retries_invalid_data=2,  # small retry for malformed JSON
        num_retries_failed_request=2,
        output_language="en",
    )


# --------------------------------- main --------------------------------------


def _sanity_df() -> pd.DataFrame:
    """Two small mock notes (1 positive with ICI/AKI/attribution, 1 negative)."""
    rows = [
        dict(
            patient_id="demo1",
            encounter_id="e1",
            note_id="sanity:0000001",
            note_datetime=pd.Timestamp("2024-01-01"),
            note_type="Oncology Progress",
            note_text=(
                "Started pembrolizumab 200 mg IV q3w on 2024-03-15. "
                "Creatinine rose from 0.9 to 1.6 mg/dL last week, concerning for AKI. "
                "Given timing after PD-1 therapy and lack of infection, AKI is likely immune-related."
            ),
        ),
        dict(
            patient_id="demo2",
            encounter_id="e2",
            note_id="sanity:0000002",
            note_datetime=pd.Timestamp("2024-01-02"),
            note_type="Nephrology Consult",
            note_text=(
                "Contrast given for CT yesterday. Creatinine 1.1→1.6 mg/dL; "
                "AKI most consistent with contrast nephropathy. ICI therapy held previously; "
                "no evidence that checkpoint inhibitor caused AKI."
            ),
        ),
    ]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="MedACE concept extraction (minimal)")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--notes", help="Path to DelPHEA v3 parquet file")
    src.add_argument(
        "--sanity-check", action="store_true", help="Run on two mock notes"
    )

    ap.add_argument("--outdir", default="out", help="Output directory (parquet)")
    ap.add_argument(
        "--backend-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible base URL",
    )
    ap.add_argument(
        "--model", default="openai/gpt-oss-120b", help="Model id for the vLLM server"
    )
    ap.add_argument(
        "--max-items-per-call",
        type=int,
        default=1,
        help="Extraction batch size per concept",
    )
    ap.add_argument(
        "--use-system-steering",
        action="store_true",
        help="Enable long clinical system prompt",
    )
    ap.add_argument(
        "--save-usage", action="store_true", help="Write API usage to out/usage.jsonl"
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Write partial parquet every N notes (0=disabled)",
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        default=True,
        help="Raise on invalid extraction (default True)",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.sanity_check:
        df = _sanity_df()
    elif args.notes:
        df = _read_delphea_v3(args.notes)
    else:
        raise SystemExit("Provide --notes <parquet> or --sanity-check")

    llm = _make_llm(args)

    drug_rows: List[Dict[str, Any]] = []
    note_rows: List[Dict[str, Any]] = []

    from tqdm import tqdm

    pbar = tqdm(df.itertuples(index=False), total=len(df), desc="notes")
    for i, row in enumerate(pbar, start=1):
        text = str(getattr(row, "note_text", "") or "").strip()
        # Prepare empty default output dicts
        drugs: List[dict] = []
        akis: List[dict] = []
        attrs: List[dict] = []

        if text:
            c_drug, c_aki, c_attr = _concepts()
            doc = Document(raw_text=text)
            doc.add_concepts([c_drug, c_aki, c_attr])

            # Fail-fast by default so we don’t silently get zero items on invalid JSON
            doc = llm.extract_all(
                doc,
                overwrite_existing=True,
                max_items_per_call=args.max_items_per_call,
                raise_exception_on_extraction_error=bool(args.fail_fast),
            )

            drugs = _flatten_items(c_drug)
            akis = _flatten_items(c_aki)
            attrs = _flatten_items(c_attr)

        # Per-note summary row
        note_rows.append(
            {
                "patient_id": getattr(row, "patient_id"),
                "encounter_id": getattr(row, "encounter_id"),
                "note_id": getattr(row, "note_id"),
                "note_date": getattr(row, "note_datetime"),
                "note_type": getattr(row, "note_type"),
                "n_drug_exposures": len(drugs),
                "n_aki_mentions": len(akis),
                "n_attributions": len(attrs),
                "concepts_json": json.dumps(
                    {
                        "drug_exposures": drugs,
                        "aki_mentions": akis,
                        "attributions": attrs,
                    },
                    ensure_ascii=False,
                ),
            }
        )

        # Drug-level rows
        for d in drugs:
            drug_rows.append(
                {
                    "patient_id": getattr(row, "patient_id"),
                    "encounter_id": getattr(row, "encounter_id"),
                    "note_id": getattr(row, "note_id"),
                    "note_date": getattr(row, "note_datetime"),
                    "note_type": getattr(row, "note_type"),
                    "drug_text": d.get("drug_text"),
                    "normalized_name": d.get("normalized_name"),
                    "canonical_generic": None,  # YAGNI: no vocab mapping here
                    "class": None,
                    "rxcui": d.get("rxcui"),
                    "dose_text": d.get("dose"),
                    "route_text": d.get("route"),
                    "when_text": d.get("date") or d.get("relative_time"),
                }
            )

        # checkpoint (optional)
        if args.checkpoint_every and (i % args.checkpoint_every == 0):
            _atomic_write_parquet(
                pd.DataFrame(drug_rows),
                os.path.join(args.outdir, "drug_mentions.partial.parquet"),
            )
            _atomic_write_parquet(
                pd.DataFrame(note_rows),
                os.path.join(args.outdir, "note_concepts.partial.parquet"),
            )
            pbar.set_postfix_str(f"checkpoint@{i}")

    # Final writes
    _atomic_write_parquet(
        pd.DataFrame(drug_rows), os.path.join(args.outdir, "drug_mentions.parquet")
    )
    _atomic_write_parquet(
        pd.DataFrame(note_rows), os.path.join(args.outdir, "note_concepts.parquet")
    )
    print(
        f"[OK] wrote {len(drug_rows)} drug rows and {len(note_rows)} note rows to {args.outdir}"
    )

    # Optional: usage trace (serialize safely)
    if args.save_usage:
        try:
            usage = llm.get_usage()
            out_path = os.path.join(args.outdir, "usage.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for call in getattr(usage, "calls", []) or []:
                    rec = {
                        "model": getattr(call, "model", None),
                        "prompt_tokens": getattr(call, "prompt_tokens", None),
                        "completion_tokens": getattr(call, "completion_tokens", None),
                        "latency_ms": getattr(call, "latency_ms", None),
                        "route": getattr(call, "route", None),
                        # Avoid dumping huge text; keep first call small if needed
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
