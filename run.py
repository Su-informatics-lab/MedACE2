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
import logging
import os
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from contextgem import Document, DocumentLLM, JsonObjectConcept, JsonObjectExample
from tqdm import tqdm

from src.prompts import SYSTEM_IRAKI  # clinical steering text
from src.rxnorm import resolve_rxcui
from src.vocab import load_vocab, normalize_name

# ----------------------------- logging controls -----------------------------


def _setup_logging(level: str = "ERROR") -> None:
    # quiet noisy libs; keep our own prints/tqdm
    lvl = getattr(logging, level.upper(), logging.ERROR)
    logging.basicConfig(level=lvl)
    for name in ("contextgem", "wtpsplit_lite", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(lvl)


# --------------------------- parquet / io helpers ---------------------------


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


# --------------------------- concept + flattening ---------------------------


def _make_examples_for_drug() -> List[JsonObjectExample]:
    return [
        JsonObjectExample(
            content={
                "drug_text": "Pembrolizumab 200 mg IV q3w given on 2024-06-15.",
                "normalized_name": "pembrolizumab",
                "dose": "200 mg",
                "route": "IV",
                "date": "2024-06-15",
                "relative_time": None,
                "sentence": "Pembrolizumab 200 mg IV q3w given on 2024-06-15.",
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
                "sentence": "Started Opdivo last month",
                "rxcui": "1658841",
            }
        ),
    ]


def _make_examples_for_aki_mention() -> List[JsonObjectExample]:
    return [
        JsonObjectExample(
            content={
                "sentence": "Acute kidney injury noted on 03/21.",
                "onset_date": "03/21",
            }
        ),
        JsonObjectExample(content={"sentence": "AKI improving.", "onset_date": None}),
    ]


def _make_examples_for_attr() -> List[JsonObjectExample]:
    return [
        JsonObjectExample(
            content={
                "stance": "likely ICI-related",
                "drug_name": "pembrolizumab",
                "sentences": [
                    "Temporal association with pembrolizumab initiation.",
                    "Urinalysis bland; eosinophilia present.",
                ],
                "rationale": "Timing and labs consistent with AIN from PD-1 inhibitor.",
            }
        ),
        JsonObjectExample(
            content={
                "stance": "unlikely ICI-related",
                "drug_name": None,
                "sentences": ["Sepsis and hypotension at AKI onset."],
                "rationale": "Prerenal/ATN pattern more likely.",
            }
        ),
    ]


def _concepts_for_note(
    allowed_names_for_hint: Optional[List[str]] = None,
) -> List[JsonObjectConcept]:
    # brief but explicit guidance + examples; always return [] if none
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
        examples=_make_examples_for_drug(),
    )
    aki_concept = JsonObjectConcept(
        name="AKI Mention",
        description=(
            "Sentences mentioning acute kidney injury (AKI) and any onset date/timeframe. "
            "Return an empty list if none."
        ),
        structure={"sentence": str, "onset_date": str | None},
        add_references=True,
        reference_depth="sentences",
        examples=_make_examples_for_aki_mention(),
    )
    attrib_concept = JsonObjectConcept(
        name="AKI Attribution",
        description=(
            "Clinician stance that AKI is caused by ICI, ruled out, or uncertain; include brief rationale and citations. "
            "Return an empty list if none."
        ),
        structure={
            "stance": str,
            "drug_name": str | None,
            "sentences": list[str],
            "rationale": str | None,
        },
        add_references=True,
        reference_depth="sentences",
        examples=_make_examples_for_attr(),
    )
    alt_cause_concept = JsonObjectConcept(
        name="Alternative AKI Causes",
        description=(
            "Non-ICI AKI causes (e.g., sepsis, contrast, obstruction, nephrotoxins) with evidence sentences. "
            "Return an empty list if none."
        ),
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


# ------------------------- SaT safety: pre-chunking -------------------------

_BOUNDARIES_RE = re.compile(r"(\n{2,}|(?<=[.!?])\s+)")


def _safe_chunk_text(text: str, *, max_chars: int) -> List[str]:
    """split text into chunks ≤ max_chars on friendly boundaries to avoid SaT chunk asserts."""
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
            cur, cur_len = [], 0

    for piece in _BOUNDARIES_RE.split(s):
        if not piece:
            continue
        if len(piece) > max_chars:
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
    max_items_per_call: int,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """run extraction over pre-chunked text and merge results. robust to SaT asserts."""
    all_drug: List[dict] = []
    all_aki: List[dict] = []
    all_attr: List[dict] = []
    all_alt: List[dict] = []

    # tiered chunk sizes: primary -> retry smaller -> micro-fallback
    tiers = [max_chars, max(max_chars // 2, 4000), 2000]

    last_err: Optional[Exception] = None
    for tier in tiers:
        try:
            chunks = _safe_chunk_text(text, max_chars=tier)
            for chunk in chunks:
                c_drug, c_aki, c_attr, c_alt = _concepts_for_note(
                    allowed_names_for_hint=allowed_keys
                )
                doc = Document(raw_text=chunk)
                doc.add_concepts([c_drug, c_aki, c_attr, c_alt])
                # IMPORTANT: raise on invalid JSON so we don't silently drop
                llm.extract_all(
                    doc,
                    overwrite_existing=True,
                    max_items_per_call=max_items_per_call,
                    raise_exception_on_extraction_error=True,
                )
                all_drug.extend(_flatten_items(c_drug))
                all_aki.extend(_flatten_items(c_aki))
                all_attr.extend(_flatten_items(c_attr))
                all_alt.extend(_flatten_items(c_alt))
            # success at this tier
            return all_drug, all_aki, all_attr, all_alt
        except Exception as e:  # capture JSON validation or API errors
            last_err = e
            all_drug.clear()
            all_aki.clear()
            all_attr.clear()
            all_alt.clear()
            continue

    # all tiers failed
    if last_err:
        raise last_err
    raise AssertionError("SaT/LLM extraction failed after multi-tier chunking")


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
    "sentence",
]

NOTE_COLS = [
    "patient_id",
    "encounter_id",
    "note_id",
    "note_date",
    "note_type",
    "has_alt_cause",
    "n_drug_exposures_raw",
    "n_drug_exposures",
    "n_aki_mentions",
    "concepts_json",
]


def _atomic_write_parquet(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# ---------------------------------- main ------------------------------------


def _parse_range(s: Optional[str], n: int) -> Tuple[int, int]:
    """'start:end' -> (start, end) with end exclusive; handles blanks."""
    if not s:
        return 0, n
    parts = s.split(":", 1)
    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if len(parts) > 1 and parts[1] else n
    start = max(0, min(start, n))
    end = max(start, min(end, n))
    return start, end


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MedACE irAKI concept extraction (ContextGem, parquet-only)"
    )
    ap.add_argument("--notes", required=True, help="path to DelPHEA v3 Parquet notes")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--debug", action="store_true", help="process first 10 notes only")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N notes")
    ap.add_argument(
        "--range",
        type=str,
        default=None,
        help="slice as 'START:END' (END exclusive), applied after --offset",
    )
    ap.add_argument("--backend-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument(
        "--drug-vocab",
        default="resources/ici_vocab.csv",
        help="csv from vocab builder (defaults to resources/ici_vocab.csv if present)",
    )
    ap.add_argument("--filter-ici", action="store_true", help="keep only ICI exposures")
    ap.add_argument(
        "--rxnorm-online",
        action="store_true",
        help="resolve RxCUI via RxNav if missing",
    )
    ap.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=8000,
        help="max characters per chunk for SaT",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="write partial results every N notes (atomic overwrite)",
    )
    ap.add_argument(
        "--log-level",
        type=str,
        default="ERROR",
        help="python logging level for libraries (default ERROR)",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=1200,
        help="max completion tokens for extraction responses",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature for extraction",
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=0.2,
        help="top-p for extraction",
    )
    ap.add_argument(
        "--max-items-per-call",
        type=int,
        default=1,
        help="batch size of concepts per LLM call (1=safer)",
    )
    ap.add_argument(
        "--save-usage",
        action="store_true",
        help="dump LLM usage/calls into out/usage.jsonl at checkpoints",
    )
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
        help="abort on the first LLM/segmentation error instead of continuing",
    )
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    _setup_logging(args.log_level)

    df = _load_notes_delphea_v3(args.notes)

    # selection
    if args.offset > 0:
        df = df.iloc[args.offset :].reset_index(drop=True)
    if args.debug:
        df = df.head(10).copy()

    n = len(df)
    start, end = _parse_range(args.range, n)
    if start or end != n:
        df = df.iloc[start:end].reset_index(drop=True)

    # vocab
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
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=120,
        max_tokens=args.max_tokens,
        num_retries_failed_request=3,
        max_retries_invalid_data=3,
        output_language="en",
    )

    drug_rows: List[Dict[str, Any]] = []
    note_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    pbar = tqdm(df.iterrows(), total=len(df), desc="notes")
    for i, (_, row) in enumerate(pbar, start=1):
        text = _clean_str(row.get("note_text"))
        placeholder_concepts = {
            "drug_exposures": [],
            "aki_mentions": [],
            "attributions": [],
            "alt_causes": [],
        }

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
                        placeholder_concepts, ensure_ascii=False
                    ),
                }
            )
        else:
            try:
                (
                    drug_items,
                    aki_items,
                    attr_items,
                    alt_items,
                ) = _extract_from_text_chunks(
                    text,
                    llm,
                    allowed_keys,
                    max_chars=args.max_chars_per_chunk,
                    max_items_per_call=args.max_items_per_call,
                )
            except Exception as e:
                # record error + placeholder; optionally stop
                errors.append(
                    {
                        "note_id": row.get("note_id"),
                        "len": len(text),
                        "error": repr(e),
                    }
                )
                if args.stop_on_error:
                    raise
                drug_items, aki_items, attr_items, alt_items = [], [], [], []

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
                        "when_text": _clean_str(
                            de.get("date") or de.get("relative_time")
                        )
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

        # checkpoint
        if args.checkpoint_every and (i % args.checkpoint_every == 0):
            df_d = pd.DataFrame(drug_rows, columns=DRUG_COLS)
            df_n = pd.DataFrame(note_rows, columns=NOTE_COLS)
            _atomic_write_parquet(
                df_d, os.path.join(args.outdir, "drug_mentions.partial.parquet")
            )
            _atomic_write_parquet(
                df_n, os.path.join(args.outdir, "note_concepts.partial.parquet")
            )
            if args.save_usage:
                usage = llm.get_usage()
                with open(os.path.join(args.outdir, "usage.jsonl"), "a") as f:
                    f.write(json.dumps(usage, ensure_ascii=False) + "\n")
            if errors:
                with open(os.path.join(args.outdir, "errors.jsonl"), "a") as f:
                    for rec in errors:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                errors.clear()
            pbar.set_postfix_str(f"checkpoint@{i}")

    # final write (ensure stable schemas even if empty)
    _atomic_write_parquet(
        pd.DataFrame(drug_rows, columns=DRUG_COLS),
        os.path.join(args.outdir, "drug_mentions.parquet"),
    )
    _atomic_write_parquet(
        pd.DataFrame(note_rows, columns=NOTE_COLS),
        os.path.join(args.outdir, "note_concepts.parquet"),
    )
    if errors:
        with open(os.path.join(args.outdir, "errors.jsonl"), "a") as f:
            for rec in errors:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(
        f"[OK] wrote {len(drug_rows)} drug rows and {len(note_rows)} note rows to {args.outdir}"
    )


if __name__ == "__main__":
    main()
