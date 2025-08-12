# run.py
# medace iraki prototype: note-level extraction + evidence labeling
# supports your clinical_notes_version3 parquet schema and the standard schema.

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.backend import LLMClient
from src.chunker import chunk_text
from src.extractors import LLMExtractor
from src.io_utils import save_parquet  # reuse writer
from src.rxnorm import resolve_rxcui
from src.score import label_note_evidence
from src.vocab import load_vocab, normalize_name

# ------------------------------- helpers ---------------------------------


def _approx_date_from_text(s: str) -> Optional[pd.Timestamp]:
    import re

    if not s:
        return None
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        try:
            return pd.to_datetime(m.group(1))
        except Exception:
            pass
    m = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", s)
    if m:
        try:
            return pd.to_datetime(m.group(1))
        except Exception:
            pass
    return None


def _clean_str(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    if s.lower() in {"none", "nan", "null"}:
        return ""
    return s


def _strip_decimal_id(s: str) -> str:
    # turn "1168000077278488.000000000000000000" into "1168000077278488"
    if "." in s:
        left, right = s.split(".", 1)
        if set(right) <= {"0"}:
            return left
    return s


def _stable_note_id(person: str, enc: str, dt: str, idx: int) -> str:
    base = f"{person}|{enc}|{dt}|{idx}"
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"iraki_note_{h}"


def _detect_schema(cols: List[str]) -> str:
    lc = {c.lower() for c in cols}
    if {
        "service_name",
        "physiologic_time",
        "obfuscated_global_person_id",
        "report_text",
    }.issubset(lc):
        return "delphea_v3"
    if {"patient_id", "note_id", "note_datetime", "note_text"}.issubset(lc):
        return "standard"
    return "unknown"


def _load_notes_any(path: str, file_format: str, dataset_schema: str) -> pd.DataFrame:
    # read raw
    if file_format == "parquet":
        df = pd.read_parquet(path)
    elif file_format == "csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("format must be parquet or csv")

    # decode schema
    schema = dataset_schema
    if schema == "auto":
        schema = _detect_schema(list(df.columns))

    if schema == "standard":
        # require columns already present; ensure optional ones exist
        need = ["patient_id", "note_id", "note_datetime", "note_text"]
        for c in need:
            if c not in df.columns:
                raise ValueError(f"missing required column: {c}")
        if "encounter_id" not in df.columns:
            df["encounter_id"] = None
        if "note_type" not in df.columns:
            df["note_type"] = None
        df["note_datetime"] = pd.to_datetime(df["note_datetime"], errors="coerce")
        df["note_text"] = df["note_text"].fillna("")
        return df

    if schema == "delphea_v3":
        # map:
        # SERVICE_NAME -> note_type
        # PHYSIOLOGIC_TIME -> note_datetime
        # OBFUSCATED_GLOBAL_PERSON_ID -> patient_id
        # ENCOUNTER_ID -> encounter_id
        # REPORT_TEXT -> note_text
        df = df.copy()
        df["note_type"] = (
            df["SERVICE_NAME"].apply(_clean_str) if "SERVICE_NAME" in df.columns else ""
        )
        df["note_datetime"] = (
            pd.to_datetime(df["PHYSIOLOGIC_TIME"], errors="coerce")
            if "PHYSIOLOGIC_TIME" in df.columns
            else pd.NaT
        )

        pid = (
            df["OBFUSCATED_GLOBAL_PERSON_ID"].apply(_clean_str)
            if "OBFUSCATED_GLOBAL_PERSON_ID" in df.columns
            else ""
        )
        df["patient_id"] = pid.apply(_strip_decimal_id)

        enc = (
            df["ENCOUNTER_ID"].apply(_clean_str) if "ENCOUNTER_ID" in df.columns else ""
        )
        df["encounter_id"] = enc.apply(_strip_decimal_id)

        txt = df["REPORT_TEXT"].astype(str) if "REPORT_TEXT" in df.columns else ""
        # normalize None/placeholder
        df["note_text"] = txt.apply(
            lambda s: "" if s.strip().lower() in {"none", "nan", "null"} else s
        )

        # synthesize note_id deterministically
        ids = []
        for i, r in df.reset_index(drop=True).iterrows():
            ids.append(
                _stable_note_id(
                    _clean_str(r.get("patient_id")),
                    _clean_str(r.get("encounter_id")),
                    str(r.get("note_datetime")),
                    i,
                )
            )
        df["note_id"] = ids

        # keep only the unified columns + a few pass-throughs
        keep = [
            "patient_id",
            "note_id",
            "encounter_id",
            "note_datetime",
            "note_type",
            "note_text",
        ]
        # preserve original service and time for debugging if needed
        for extra in ["SERVICE_NAME", "PHYSIOLOGIC_TIME"]:
            if extra in df.columns and extra not in keep:
                keep.append(extra)
        return df[keep]

    raise ValueError(
        "unknown dataset schema. pass --dataset-schema delphea_v3 or standard, "
        "or use --dataset-schema auto and ensure columns are detectable."
    )


# ------------------------------- main -------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MedACE irAKI prototype: note-level extraction + evidence labeling"
    )

    # data
    ap.add_argument("--notes", required=True, help="path to notes parquet/csv")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument(
        "--dataset-schema", choices=["auto", "standard", "delphea_v3"], default="auto"
    )
    ap.add_argument("--sample-n", type=int, default=50)
    ap.add_argument("--outdir", default="outputs")

    # llm backend
    ap.add_argument("--backend-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="openai/gpt-oss-120b")

    # chunking
    ap.add_argument("--chunk-chars", type=int, default=3500)
    ap.add_argument("--max-chunks", type=int, default=6)

    # vocab + filtering
    ap.add_argument(
        "--drug-vocab",
        default=None,
        help="CSV from your vocab builder; if omitted, tries resources/ici_vocab.csv",
    )
    ap.add_argument(
        "--filter-ici",
        action="store_true",
        help="keep only drug exposures that match vocab",
    )

    # rxnorm
    ap.add_argument(
        "--rxnorm-online",
        action="store_true",
        help="resolve RxCUI via RxNav (prefer local RxNav-in-a-Box)",
    )

    # scoring
    ap.add_argument("--window-days", type=int, default=60)

    # outputs
    ap.add_argument("--save-csv", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # notes loader supports your parquet schema
    df = _load_notes_any(args.notes, args.format, args.dataset_schema)
    if args.sample_n and args.sample_n > 0 and len(df) > args.sample_n:
        df = df.sample(n=args.sample_n, random_state=42).copy()

    # auto default vocab if present
    vocab_path = args.drug_vocab
    if not vocab_path:
        candidate = os.path.join("resources", "ici_vocab.csv")
        if os.path.exists(candidate):
            vocab_path = candidate

    vocab = load_vocab(vocab_path) if vocab_path else None

    # wire llm
    client = LLMClient(endpoint_url=args.backend_url, model=args.model)
    extractor = LLMExtractor(client)

    drug_rows: List[Dict[str, Any]] = []
    ev_rows: List[Dict[str, Any]] = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="notes"):
        note_text = str(row.get("note_text") or "")
        chunks = chunk_text(
            note_text, max_chars=args.chunk_chars, max_chunks=args.max_chunks
        )

        all_concepts: List[Dict[str, Any]] = []
        for ch in chunks:
            try:
                concepts = extractor.concepts(ch["text"])
            except Exception:
                concepts = {
                    "drug_exposures": [],
                    "aki_mentions": [],
                    "attributions": [],
                    "alt_causes": [],
                }
            all_concepts.append(concepts)

        # collect
        drug_dates: List[pd.Timestamp] = []
        alt_causes_present = False

        # alt causes
        for c in all_concepts:
            for alt in c.get("alt_causes") or []:
                if (alt.get("cause") or "").strip():
                    alt_causes_present = True

        # drug exposures
        for c in all_concepts:
            for de in c.get("drug_exposures") or []:
                dn_raw = (
                    de.get("normalized_name") or de.get("drug_text") or ""
                ).strip()
                dn_key = normalize_name(dn_raw)

                # optional filter to ICI
                if (
                    args.filter_ici
                    and vocab is not None
                    and dn_key not in vocab["name_to_canonical"]
                ):
                    continue

                rxcui = (de.get("rxcui") or "").strip()
                if not rxcui and args.rxnorm_online and dn_raw:
                    r = resolve_rxcui(dn_raw)
                    if r:
                        rxcui = r

                sentence = de.get("sentence") or ""
                date_guess = _approx_date_from_text(sentence) or _approx_date_from_text(
                    de.get("date") or ""
                )
                if date_guess is not None:
                    drug_dates.append(pd.to_datetime(date_guess, errors="coerce"))

                vocab_row = vocab["by_name"].get(dn_key, {}) if vocab else {}

                drug_rows.append(
                    {
                        "patient_id": row.get("patient_id"),
                        "encounter_id": row.get("encounter_id"),
                        "note_id": row.get("note_id"),
                        "note_date": row.get("note_datetime"),
                        "note_type": row.get("note_type"),
                        "drug_text": de.get("drug_text"),
                        "normalized_name": dn_raw,
                        "canonical_generic": vocab_row.get("canonical_generic"),
                        "class": vocab_row.get("class"),
                        "rxcui": rxcui,
                        "dose_text": de.get("dose"),
                        "route_text": de.get("route"),
                        "when_text": de.get("date") or de.get("relative_time"),
                        "sentence": sentence,
                    }
                )

        # aki + attribution
        aki_onset: Optional[pd.Timestamp] = None
        evidence_sentences: List[str] = []
        rationale = ""
        attributions: List[Dict[str, Any]] = []
        drug_name_for_att: Optional[str] = None

        for c in all_concepts:
            for a in c.get("attributions") or []:
                attributions.append(a)
                if a.get("sentences"):
                    evidence_sentences.extend([s for s in a.get("sentences") if s])
                if a.get("rationale"):
                    rationale = a.get("rationale")
                if a.get("drug_name"):
                    drug_name_for_att = a.get("drug_name")

            for am in c.get("aki_mentions") or []:
                if am.get("onset_date"):
                    try:
                        aki_onset = pd.to_datetime(am["onset_date"])
                    except Exception:
                        pass
                if am.get("sentence"):
                    evidence_sentences.append(am["sentence"])

        label = label_note_evidence(
            note_row=row.to_dict(),
            attributions=attributions,
            drug_dates=[d for d in drug_dates if pd.notnull(d)],
            aki_onset=aki_onset,
            window_days=args.window_days,
            has_alt_cause=alt_causes_present,
        )

        ev_rows.append(
            {
                "patient_id": row.get("patient_id"),
                "encounter_id": row.get("encounter_id"),
                "note_id": row.get("note_id"),
                "note_date": row.get("note_datetime"),
                "best_relation": label["best_relation"],
                "evidence_level": label["evidence_level"],
                "drug_name": drug_name_for_att,
                "has_alt_cause": alt_causes_present,
                "evidence_sentences": "\n".join(evidence_sentences[:5]),
                "rationale": rationale,
                "note_text": (row.get("note_text") or "")[:1200],
            }
        )

    df_drugs = pd.DataFrame(drug_rows)
    df_evid = pd.DataFrame(ev_rows)

    save_parquet(df_drugs, os.path.join(args.outdir, "drug_mentions.parquet"))
    save_parquet(df_evid, os.path.join(args.outdir, "iraki_evidence.parquet"))

    if args.save_csv:
        df_drugs.to_csv(os.path.join(args.outdir, "drug_mentions.csv"), index=False)
        df_evid.to_csv(os.path.join(args.outdir, "iraki_evidence.csv"), index=False)

    try:
        from scripts.labelstudio_export import to_labelstudio

        df_prob = df_evid[df_evid["evidence_level"] == "prob"].copy()
        to_labelstudio(df_prob, os.path.join(args.outdir, "labelstudio_prob.json"))
    except Exception as e:
        print(f"[warn] labelstudio export skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
