"""Explicit test doubles: never selected by runtime configuration or production code."""

import hashlib
import json
import re

import numpy as np
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from rfp.agents.agent import Agent


class TestEmbeddings:
    fingerprint = "synthetic-test-v1"

    def __init__(self):
        self.encoded_documents = 0

    def encode(self, texts, *, query=False):
        if not query:
            self.encoded_documents += len(texts)
        matrix = np.zeros((len(texts), 256), dtype="float32")
        for i, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                matrix[i, int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 256] += 1
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)


class TestReranker:
    def score(self, query, passages):
        tokens = set(re.findall(r"\w+", query.lower()))
        return [len(tokens & set(re.findall(r"\w+", p.lower()))) for p in passages]


def field_from_hit(hit):
    chunk = hit["chunk"]
    return {
        "value": chunk["text"],
        "sources": [
            {
                "chunk_id": chunk["chunk_id"],
                "bid_id": chunk["bid_id"],
                "file": chunk["file_name"],
                "page": chunk["page_number"],
                "section": chunk["section"],
                "quote": chunk["text"],
            }
        ],
        "confidence": 0.8,
        "notes": "Synthetic test response",
    }


class ScriptedLlm(BaseLlm):
    """Exercise the real ADK Runner/session/schema path without calling a paid provider."""

    model: str = "scripted-test"

    @classmethod
    def supported_models(cls):
        return ["scripted-test"]

    async def generate_content_async(self, llm_request, stream=False):
        payload = json.loads(llm_request.contents[-1].parts[0].text)
        if "groups" in payload:
            result = {"tasks": ["retrieve", "extract", "reconcile", "validate"]}
        elif "evidence_by_field" in payload:
            fields = {}
            for name, hits in payload["evidence_by_field"].items():
                if name == "Due Date" or not hits:
                    fields[name] = {
                        "value": None,
                        "sources": [],
                        "confidence": 0,
                        "notes": "Not found in documents",
                    }
                else:
                    fields[name] = field_from_hit(hits[0])
                    if name == "Bid Summary":
                        fields[name]["value"] = (
                            "Test devices are requested. Test equipment is specified. Evidence is cited."
                        )
            result = {"fields": fields}
        elif "addendum_passages" in payload:
            result = {"fields": payload["fields"], "addendum_changes": []}
        elif "question" in payload:
            item = field_from_hit(payload["evidence"][0])
            result = {
                "answer": item["value"],
                "sources": item["sources"],
                "confidence": 0.8,
                "notes": "Test double",
            }
        else:
            result = {"rejected": {}}
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=json.dumps(result))]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10, candidates_token_count=10
            ),
        )


class TestAgent(Agent):
    def _build_model(self):
        return ScriptedLlm()
