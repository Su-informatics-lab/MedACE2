from __future__ import annotations

import re
from typing import Any, Dict, List

# section‑aware chunking: lightweight heuristics


def simple_sections(text: str) -> List[str]:
    parts = re.split(
        r"\n\s*(?:ASSESSMENT|PLAN|HPI|HISTORY|MEDICATIONS|IMPRESSION|DIAGNOSIS|ROS|PE)\s*:|\n\s*\n",
        text,
        flags=re.IGNORECASE,
    )
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts


def chunk_text(
    text: str, max_chars: int = 3500, max_chunks: int = 6
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for sec in simple_sections(text) or [text]:
        s = sec
        while s and len(chunks) < max_chunks:
            chunk = s[:max_chars]
            chunks.append({"text": chunk})
            s = s[max_chars:]
        if len(chunks) >= max_chunks:
            break
    if not chunks:
        chunks = [{"text": text[:max_chars]}]
    return chunks[:max_chunks]
