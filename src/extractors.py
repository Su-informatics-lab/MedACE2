from __future__ import annotations

import json
from typing import Any, Dict, List

from .backend import LLMClient
from .prompts import ASPECT_TASK, CONCEPT_TASK_SCHEMA, IRAKI_KNOWLEDGE

# llm‑only extractor for prototype (contextgem adapter can be added later)


class LLMExtractor:
    def __init__(self, client: LLMClient):
        self.client = client

    def aspects(self, chunk: str) -> List[Dict[str, Any]]:
        messages = [
            {"role": "system", "content": IRAKI_KNOWLEDGE},
            {"role": "user", "content": f"{ASPECT_TASK}\n\nTEXT:\n{chunk}"},
        ]
        out = self.client.chat(
            messages,
            temperature=0.0,
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(out)
            if isinstance(data, dict) and "items" in data:
                items = data.get("items", [])
            elif isinstance(data, list):
                items = data
            else:
                items = []
        except Exception:
            items = []
        return items

    def concepts(self, chunk: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": IRAKI_KNOWLEDGE},
            {"role": "user", "content": f"{CONCEPT_TASK_SCHEMA}\n\nTEXT:\n{chunk}"},
        ]
        out = self.client.chat(
            messages,
            temperature=0.0,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(out)
        except Exception:
            data = {
                "drug_exposures": [],
                "aki_mentions": [],
                "attributions": [],
                "alt_causes": [],
            }
        return data


class ContextGemExtractor:
    def __init__(self):
        raise NotImplementedError(
            "contextgem extractor not wired yet; use LLMExtractor for prototype."
        )
