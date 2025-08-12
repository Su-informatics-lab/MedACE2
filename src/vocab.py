import re
from typing import Optional

import pandas as pd


def normalize_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def load_vocab(path: Optional[str]):
    """
    Robust loader for the canonical ICI vocab CSV.

    - Treats empty cells as empty strings (not NaN)
    - Ignores literal 'nan' strings
    - Builds lookup dicts:
        - by_name[normalized_name] -> {"canonical_generic": ..., "class": ...}
        - name_to_canonical[normalized_name] -> canonical_generic
    """
    if not path:
        return None

    # Avoid NaN: keep_default_na=False keeps blanks as "", and fillna('') guards any leftovers.
    df = pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")

    def _split(colval: str) -> list[str]:
        s = (colval or "").strip()
        if not s or s.lower() == "nan":
            return []
        return [x.strip() for x in s.split("|") if x and x.strip().lower() != "nan"]

    by_name = {}
    name_to_canonical = {}

    for _, r in df.iterrows():
        canon = (r.get("canonical_generic") or "").strip()
        cls = (r.get("class") or "").strip()

        brands = _split(r.get("brand_names", ""))
        syns = _split(r.get("synonyms", ""))
        # include the canonical generic itself as a synonym
        names = [canon] + brands + syns

        for n in names:
            key = normalize_name(n)
            if not key:
                continue
            name_to_canonical[key] = canon
            by_name[key] = {"canonical_generic": canon, "class": cls}

    return {"by_name": by_name, "name_to_canonical": name_to_canonical}
