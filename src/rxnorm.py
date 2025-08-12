from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

# optional online rxcui resolver; off by default


def resolve_rxcui(drug_name: str, base_url: Optional[str] = None) -> Optional[str]:
    base = base_url or os.environ.get(
        "RXNAV_BASE_URL", "https://rxnav.nlm.nih.gov/REST"
    )
    try:
        r = requests.get(f"{base}/rxcui.json", params={"name": drug_name}, timeout=20)
        r.raise_for_status()
        data: Dict[str, Any] = r.json()
        rxcui = (data.get("idGroup") or {}).get("rxnormId", [None])[0]
        return rxcui
    except Exception:
        return None
