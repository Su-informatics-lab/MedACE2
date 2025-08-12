from __future__ import annotations

import argparse
import re
from typing import Optional

import pandas as pd


def normalize_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def load_vocab(path: Optional[str]):
    if not path:
        return None
    df = pd.read_csv(path, dtype=str)
    by_name = {}
    name_to_canonical = {}
    for _, r in df.iterrows():
        canon = (r.get("canonical_generic") or "").strip()
        cls = (r.get("class") or "").strip()
        names = [canon] + [
            x.strip() for x in (r.get("brand_names") or "").split("|") if x
        ]
        names += [x.strip() for x in (r.get("synonyms") or "").split("|") if x]
        for n in names:
            key = normalize_name(n)
            if not key:
                continue
            name_to_canonical[key] = canon
            by_name[key] = {"canonical_generic": canon, "class": cls}
    return {"by_name": by_name, "name_to_canonical": name_to_canonical}


def build_vocab(hcpcs_csv: str, ndc_csv: str, regimens_xlsx: str) -> pd.DataFrame:
    hc = pd.read_csv(hcpcs_csv, dtype=str, low_memory=False)
    nd = pd.read_csv(ndc_csv, dtype=str, low_memory=False)
    xls = pd.ExcelFile(regimens_xlsx)

    hc["generic"] = hc.get("Generic Name")
    hc["brand"] = hc.get("Brand Name")
    hc["major_class"] = hc.get("Major Drug Class")
    hc["hcpcs_code"] = hc.get("HCPCS")

    nd["generic"] = nd.get("Generic Name")
    nd["brand"] = nd.get("Brand Name")
    nd["ndc11"] = nd.get("NDC-11 (Package)")
    nd["ndc9"] = nd.get("NDC-9 (Product)")

    sheets = [s for s in xls.sheet_names if s not in {"Cover Page"}]
    rows = []
    for s in sheets:
        df = pd.read_excel(regimens_xlsx, sheet_name=s, dtype=str)
        cand_name_cols = [
            c
            for c in df.columns
            if any(k in c.lower() for k in ["generic", "drug", "name"])
        ]
        name_cols = cand_name_cols or list(df.columns)
        for _, r in df.iterrows():
            names = []
            for c in name_cols:
                v = (str(r.get(c)) if c in r else "").strip()
                if v and v.lower() != "nan":
                    names.append(v)
            generic = names[0] if names else ""
            rows.append({"canonical_generic": generic, "class": s.replace("_", "-")})

    reg = pd.DataFrame(rows)

    def agg_names(df, generic_col, brand_col):
        tmp = df[[generic_col, brand_col]].dropna()
        tmp[generic_col] = tmp[generic_col].fillna("").astype(str)
        tmp[brand_col] = tmp[brand_col].fillna("").astype(str)
        g = tmp.groupby(generic_col)[brand_col].apply(
            lambda x: sorted(set([v for v in x if v and v.lower() != "nan"]))
        )
        return g

    brands_hc = agg_names(hc, "generic", "brand")
    brands_nd = agg_names(nd, "generic", "brand")

    def agg_codes(df, generic_col, code_col):
        tmp = df[[generic_col, code_col]].dropna()
        tmp = tmp[tmp[generic_col].notna() & tmp[code_col].notna()]
        g = tmp.groupby(generic_col)[code_col].apply(
            lambda x: sorted(set([v for v in x if v and v.lower() != "nan"]))
        )
        return g

    hcpcs_by_generic = agg_codes(hc, "generic", "hcpcs_code")
    ndc11_by_generic = agg_codes(nd, "generic", "ndc11")
    ndc9_by_generic = agg_codes(nd, "generic", "ndc9")

    reg["brand_names"] = reg["canonical_generic"].map(
        lambda g: sorted(
            set((brands_hc.get(g, []) or []) + (brands_nd.get(g, []) or []))
        )
    )
    reg["hcpcs_codes"] = reg["canonical_generic"].map(
        lambda g: hcpcs_by_generic.get(g, [])
    )
    reg["ndc11_list"] = reg["canonical_generic"].map(
        lambda g: ndc11_by_generic.get(g, [])
    )
    reg["ndc9_list"] = reg["canonical_generic"].map(
        lambda g: ndc9_by_generic.get(g, [])
    )
    reg["synonyms"] = reg.apply(
        lambda r: sorted(set([r["canonical_generic"], *r["brand_names"]])), axis=1
    )
    reg["rxcui"] = ""

    for c in ["brand_names", "hcpcs_codes", "ndc11_list", "ndc9_list", "synonyms"]:
        reg[c] = reg[c].apply(
            lambda x: "|".join(x) if isinstance(x, list) else (x or "")
        )

    reg = reg.dropna(subset=["canonical_generic"]).drop_duplicates(
        subset=["canonical_generic", "class"]
    )
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hcpcs",
        default="resources/Hcpcs Medication List-07-14-2025_07_52.csv",
        required=True,
    )
    ap.add_argument(
        "--ndc",
        default="resources/Published Ndc Package List-07-14-2025_07_49.csv",
        required=True,
    )
    ap.add_argument(
        "--regimens", default="resources/ICI_Regimens_PK_06_02_2025.xlsx", required=True
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reg = build_vocab(args.hcpcs, args.ndc, args.regimens)
    reg.to_csv(args.out, index=False)
    print(f"wrote {args.out} with {len(reg)} rows")


if __name__ == "__main__":
    main()
