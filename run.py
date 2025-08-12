"""
MedACE | ContextGem note-level extraction for ICI drugs & irAKI context (Parquet-only).

Input parquet schema (DelPHEA v3):
  SERVICE_NAME, PHYSIOLOGIC_TIME, OBFUSCATED_GLOBAL_PERSON_ID, ENCOUNTER_ID, REPORT_TEXT

todos:
  - change online rxnorm api to a local one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from contextgem import Document, DocumentLLM, JsonObjectConcept
from tqdm import tqdm

from src.prompts import SYSTEM_IRAKI  # clinical steering text
from src.rxnorm import resolve_rxcui
from src.vocab import load_vocab, normalize_name


def _normalize_model_id(m: str) -> str:
    m = (m or "").strip()
    return m if "/" in m else f"openai/{m}"


def _read_v3_parquet_coerced(path: str) -> pd.DataFrame:
    """
    read DelPHEA v3 parquet and coerce types BEFORE pandas sees them:
      - SERVICE_NAME, REPORT_TEXT -> string
      - PHYSIOLOGIC_TIME -> timestamp (seconds)
      - OBFUSCATED_GLOBAL_PERSON_ID, ENCOUNTER_ID -> string (avoid float)
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
    return t.to_pandas(types_mapper=pd.ArrowDtype)


def _to_int64_from_str(series: pd.Series) -> pd.Series:
    """convert string-ish ids like '1168...000000000000' or '1.168e+15' to pandas Int64 safely."""

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
        except InvalidOperation:
            pass
        try:
            return int(s)
        except Exception:
            try:
                return int(float(s))
            except Exception:
                return pd.NA

    return series.map(conv).astype("Int64")


def _clean_str(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    if s.lower() in {"none", "nan", "null"}:
        return ""
    return s


def _load_notes_delphea_v3(path: str) -> pd.DataFrame:
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


def _concepts_for_note(
    allowed_names_for_hint: Optional[List[str]] = None,
) -> List[JsonObjectConcept]:
    # show a small, mixed list of synonyms/generics in the hint to bias extraction
    hint_list = (allowed_names_for_hint or [])[:30]
    allow_hint = (
        f" only keep if the drug is an ICI (examples: {', '.join(hint_list)})."
        if hint_list
        else " only keep if the drug is an ICI."
    )
    drug_concept = JsonObjectConcept(
        name="ICI Drug Exposure",
        description="extract immune checkpoint inhibitor exposures with attributes."
        + allow_hint,
        structure={
            "drug_text": str,
            "normalized_name": str,
            "dose": str | None,
            "route": str | None,
            "date": str | None,
            "relative_time": str | None,
            "sentence": str,
            "rxcui": str | None,
        },
        add_references=True,
        reference_depth="sentences",
    )
    aki_concept = JsonObjectConcept(
        name="AKI Mention",
        description="sentences mentioning acute kidney injury (AKI) and any onset date/timeframe.",
        structure={"sentence": str, "onset_date": str | None},
        add_references=True,
        reference_depth="sentences",
    )
    attrib_concept = JsonObjectConcept(
        name="AKI Attribution",
        description="clinician stance that AKI is caused by ICI, ruled out, or uncertain; include brief rationale and citations.",
        structure={
            "stance": str,
            "drug_name": str | None,
            "sentences": list[str],
            "rationale": str | None,
        },
        add_references=True,
        reference_depth="sentences",
    )
    alt_cause_concept = JsonObjectConcept(
        name="Alternative AKI Causes",
        description="non-ICI AKI causes (e.g., sepsis, contrast, obstruction, nephrotoxins) with evidence sentences.",
        structure={"cause": str, "sentences": list[str]},
        add_references=True,
        reference_depth="sentences",
    )
    return [drug_concept, aki_concept, attrib_concept, alt_cause_concept]


def _flatten_items(concept: JsonObjectConcept) -> List[dict]:
    out: List[dict] = []
    for it in getattr(concept, "extracted_items", []) or []:
        val = getattr(it, "value", None)
        refs = getattr(it, "reference_sentences", None)
        ref_sents = [s.raw_text for s in refs] if refs else []
        if isinstance(val, dict):
            if ref_sents and "source_sentences" not in val:
                val["source_sentences"] = ref_sents
            out.append(val)
        elif isinstance(val, str):
            out.append({"sentence": val, "source_sentences": ref_sents})
    return out


# ------------------------- SaT safety: pre-chunking helpers -------------------------

_BOUNDARIES_RE = re.compile(r"(\n{2,}|(?<=[.!?])\s+)")


def _safe_chunk_text(text: str, *, max_chars: int = 30000) -> List[str]:
    """split text into chunks ≤ max_chars on friendly boundaries to avoid SaT chunk asserts.

    strategy:
      1) normalize whitespace
      2) if already small, return as-is
      3) otherwise, accumulate pieces split on double newlines or sentence-like whitespace,
         flushing to a new chunk when adding the next piece would exceed max_chars
      4) if any single piece itself exceeds max_chars, hard-split it
    """
    if not text:
        return []
    s = re.sub(r"[ \t]+", " ", text).strip()
    if len(s) <= max_chars:
        return [s]

    chunks: List[str] = []
    cur = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len, chunks
        if cur:
            chunks.append("".join(cur).strip())
            cur = []
            cur_len = 0

    for piece in _BOUNDARIES_RE.split(s):
        if not piece:
            continue
        if len(piece) > max_chars:
            # hard-split oversized piece
            flush()
            for i in range(0, len(piece), max_chars):
                chunks.append(piece[i : i + max_chars].strip())
            continue
        if cur_len + len(piece) > max_chars:
            flush()
        cur.append(piece)
        cur_len += len(piece)
    flush()
    return [c for c in chunks if c]


def _extract_from_text_chunks(
    text: str,
    llm: DocumentLLM,
    allowed_keys: List[str],
    *,
    max_chars: int,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """run extraction over pre-chunked text and merge results."""
    all_drug: List[dict] = []
    all_aki: List[dict] = []
    all_attr: List[dict] = []
    all_alt: List[dict] = []

    # pre-chunk to keep SaT happy
    chunks = _safe_chunk_text(text, max_chars=max_chars) or []
    if not chunks:
        return all_drug, all_aki, all_attr, all_alt

    # prepare concepts once per chunk
    for chunk in chunks:
        c_drug, c_aki, c_attr, c_alt = _concepts_for_note(
            allowed_names_for_hint=allowed_keys
        )
        doc = Document(raw_text=chunk)
        doc.add_concepts([c_drug, c_aki, c_attr, c_alt])

        try:
            llm.extract_all(
                doc, overwrite_existing=True, raise_exception_on_extraction_error=False
            )
        except AssertionError:
            # rare SaT internal mismatch; retry with smaller chunks
            smaller = _safe_chunk_text(chunk, max_chars=max(8000, max_chars // 2))
            for sub in smaller:
                sub_c_drug, sub_c_aki, sub_c_attr, sub_c_alt = _concepts_for_note(
                    allowed_names_for_hint=allowed_keys
                )
                sub_doc = Document(raw_text=sub)
                sub_doc.add_concepts([sub_c_drug, sub_c_aki, sub_c_attr, sub_c_alt])
                llm.extract_all(
                    sub_doc,
                    overwrite_existing=True,
                    raise_exception_on_extraction_error=False,
                )
                all_drug.extend(_flatten_items(sub_c_drug))
                all_aki.extend(_flatten_items(sub_c_aki))
                all_attr.extend(_flatten_items(sub_c_attr))
                all_alt.extend(_flatten_items(sub_c_alt))
            continue

        all_drug.extend(_flatten_items(c_drug))
        all_aki.extend(_flatten_items(c_aki))
        all_attr.extend(_flatten_items(c_attr))
        all_alt.extend(_flatten_items(c_alt))

    return all_drug, all_aki, all_attr, all_alt


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MedACE irAKI concept extraction (ContextGem, parquet-only)"
    )
    ap.add_argument("--notes", required=True, help="path to DelPHEA v3 Parquet notes")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--debug", action="store_true", help="process first 10 notes only")
    ap.add_argument(
        "--offset", type=int, default=0, help="skip the first N notes before processing"
    )
    ap.add_argument("--backend-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument(
        "--drug-vocab",
        default="resources/ici_vocab.csv",
        help="csv from vocab builder (defaults to resources/ici_vocab.csv if present)",
    )
    ap.add_argument(
        "--filter-ici",
        action="store_true",
        help="keep only drug exposures that map to vocab",
    )
    ap.add_argument(
        "--rxnorm-online",
        action="store_true",
        help="resolve RxCUI via RxNav if missing",
    )
    ap.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=30000,
        help="max characters per chunk to avoid SaT chunking asserts",
    )
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = _load_notes_delphea_v3(args.notes)
    if args.offset > 0:
        df = df.iloc[args.offset :].reset_index(drop=True)
    if args.debug:
        df = df.head(10).copy()

    # vocab: use full synonym set for hinting; keys are normalized terms (brands + generics)
    vocab_path = args.drug_vocab or (
        os.path.join("resources", "ici_vocab.csv")
        if os.path.exists(os.path.join("resources", "ici_vocab.csv"))
        else None
    )
    vocab = load_vocab(vocab_path) if vocab_path else None
    allowed_keys = sorted(set(vocab["name_to_canonical"].keys())) if vocab else []

    llm = DocumentLLM(
        model=_normalize_model_id(args.model),
        api_base=args.backend_url,
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        system_message=SYSTEM_IRAKI,
        temperature=0.1,
        top_p=1.0,
        timeout=120,
        max_retries_invalid_data=3,  # contextgem validates and retries malformed json
        output_language="en",
    )

    drug_rows: List[Dict[str, Any]] = []
    note_rows: List[Dict[str, Any]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="notes"):
        text = _clean_str(row.get("note_text"))
        if not text:
            note_rows.append(
                {
                    "patient_id": row.get("patient_id"),
                    "encounter_id": row.get("encounter_id"),
                    "note_id": row.get("note_id"),
                    "note_date": row.get("note_datetime"),
                    "note_type": row.get("note_type"),
                    "has_alt_cause": False,
                    "n_drug_exposures_raw": 0,
                    "n_drug_exposures": 0,
                    "n_aki_mentions": 0,
                    "concepts_json": json.dumps(
                        {
                            "drug_exposures": [],
                            "aki_mentions": [],
                            "attributions": [],
                            "alt_causes": [],
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            continue

        # run extraction across safe-sized chunks and merge
        drug_items, aki_items, attr_items, alt_items = _extract_from_text_chunks(
            text,
            llm,
            allowed_keys,
            max_chars=args.max_chars_per_chunk,
        )

        raw_drug_count = len(drug_items)

        # canonicalize + optional filtering
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
                    "sentence": de.get("sentence")
                    or (de.get("source_sentences") or [None])[0],
                }
            )

        drug_rows.extend(cleaned_drugs)
        concept_payload = {
            "drug_exposures": cleaned_drugs,
            "aki_mentions": aki_items,
            "attributions": attr_items,
            "alt_causes": alt_items,
        }
        note_rows.append(
            {
                "patient_id": row.get("patient_id"),
                "encounter_id": row.get("encounter_id"),
                "note_id": row.get("note_id"),
                "note_date": row.get("note_datetime"),
                "note_type": row.get("note_type"),
                "has_alt_cause": bool(alt_items),
                "n_drug_exposures_raw": raw_drug_count,
                "n_drug_exposures": len(cleaned_drugs),
                "n_aki_mentions": len(aki_items),
                "concepts_json": json.dumps(concept_payload, ensure_ascii=False),
            }
        )

    pd.DataFrame(drug_rows).to_parquet(
        os.path.join(args.outdir, "drug_mentions.parquet"), index=False
    )
    pd.DataFrame(note_rows).to_parquet(
        os.path.join(args.outdir, "note_concepts.parquet"), index=False
    )
    print(
        f"[ok] wrote {len(drug_rows)} drug rows and {len(note_rows)} note rows to {args.outdir}"
    )


if __name__ == "__main__":
    main()
