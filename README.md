# MedACE: ICI and irAKI concept extraction

ContextGem-based extractor that pulls immune checkpoint inhibitor (ICI) drug exposures and irAKI context from clinical 
notes. Works with a local vLLM-hosted OSS-120B.

## Quick start

```bash
# install
pip install -U contextgem pandas pyarrow tqdm requests
export OPENAI_API_KEY=not-needed
```

```bash
# init oss endpoint
sbatch serve.sbatch
```

```bash
# run pipeline
python run.py \
  --notes `the parquet file hosting notes` \
  --outdir outputs/run1 \
  --backend-url http://127.0.0.1:8000/v1 \
  --model openai/gpt-oss-120b \
  --drug-vocab "resources/ici_vocab.csv" \
  --filter-ici
```

## Inputs

Parquet columns: SERVICE_NAME, PHYSIOLOGIC_TIME, OBFUSCATED_GLOBAL_PERSON_ID, ENCOUNTER_ID, REPORT_TEXT

## Outputs

**outputs/.../drug_mentions.parquet** (one row per ICI exposure)

- patient_id(Int64), encounter_id(Int64), note_id(str), note_date(ts), note_type(str)
- drug_text(str), normalized_name(str), canonical_generic(str|null), class(str|null)
- rxcui(str|null), dose_text(str|null), route_text(str|null), when_text(str|null)
- sentence(str)

**outputs/.../note_concepts.parquet** (one row per note)

- patient_id(Int64), encounter_id(Int64), note_id(str), note_date(ts), note_type(str)
- has_alt_cause(bool), n_drug_exposures_raw(int), n_drug_exposures(int), n_aki_mentions(int)
- concepts_json(str) with keys: drug_exposures, aki_mentions, attributions, alt_causes

## Useful flags

- --debug          process first 10 notes
- --offset N       skip first N notes then process
- --filter-ici     keep only drug exposures that match the ICI vocab

