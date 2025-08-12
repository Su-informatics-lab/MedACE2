IRAKI_KNOWLEDGE = """
you are extracting evidence for immune‑checkpoint‑inhibitor‑associated acute kidney injury (iraki).
key points:
- iraki has no single definitive biomarker; diagnosis relies on exclusion and clinical judgment.
- explicit clinician attribution (or denial) outranks heuristic signals.
- supportive signals include temporal proximity to ici exposure, steroid response, and absence of alternative causes.
- outputs must be strict json. all fields in the schema must be present even if null.
    """.strip()

ASPECT_TASK = """
task: identify sentences about these aspects:
(A) ici exposure/administration
(B) kidney injury mentions (aki, interstitial nephritis, kdigo staging, creatinine rises)
(C) clinician attribution of kidney injury to ici (positive / ruled‑out / uncertain)
(D) alternative causes (contrast, sepsis, obstruction, dehydration, nephrotoxins)
return json list of objects: {"aspect": "A|B|C|D", "sentence": "...", "start": int, "end": int}.
    """.strip()

CONCEPT_TASK_SCHEMA = """
extract structured concepts in strict json with this schema:
{
  "drug_exposures": [{
    "drug_text": "", "normalized_name": "", "rxcui": "",
    "date": "", "relative_time": "", "dose": "", "route": "",
    "sentence": "", "span": [0,0]
  }],
  "aki_mentions": [{
    "assertion": "positive|negated|uncertain",
    "stage": "KDIGO1|KDIGO2|KDIGO3|null",
    "scr_value": null, "scr_unit": null,
    "onset_date": "", "sentence": "", "span": [0,0]
  }],
  "attributions": [{
    "relation": "caused_by|ruled_out|possible",
    "drug_rxcui": "", "drug_name": "",
    "evidence_level": "high|prob",
    "sentences": [""], "rationale": ""
  }],
  "alt_causes": [{
    "cause": "contrast|sepsis|obstruction|dehydration|nephrotoxin|other",
    "sentence": "", "span": [0,0]
  }]
}
    """.strip()
