"""Typed contracts shared by the API, search tools and cooperating agents."""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FIELDS = (
    "Bid Number",
    "Title",
    "Due Date",
    "Bid Submission Type",
    "Term of Bid",
    "Pre Bid Meeting",
    "Installation",
    "Bid Bond Requirement",
    "Delivery Date",
    "Payment Terms",
    "Any Additional Documentation Required",
    "MFG for Registration",
    "Contract or Cooperative to use",
    "Model_no",
    "Part_no",
    "Product",
    "contact_info",
    "company_name",
    "Bid Summary",
    "Product Specification",
)
DocType = Literal["bid_page", "rfp", "addendum", "specs", "affidavit"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Chunk(Contract):
    chunk_id: str
    bid_id: str
    file_name: str
    doc_type: DocType
    addendum_number: int | None = None
    page_number: int | None = None
    printed_page: str | None = None
    document_date: str | None = None
    section: str = ""
    method: str = "native"
    text: str
    content_hash: str


class Evidence(Contract):
    """A citation points to a real stored chunk, with an exact supporting quote."""

    chunk_id: str
    bid_id: str
    file: str
    page: int | None
    section: str = ""
    quote: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def decode_provider_object(cls, value):
        """Normalize a JSON-encoded citation object; still require every citation field."""
        if isinstance(value, str) and len(value) <= 50_000:
            decoded = json.loads(value)
            if isinstance(decoded, dict):
                return decoded
        return value


class SearchHit(Contract):
    chunk: Chunk
    score: float
    dense_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None

    def citation(self) -> Evidence:
        """Build a traceable source reference for this search result."""
        c = self.chunk
        return Evidence(
            chunk_id=c.chunk_id,
            bid_id=c.bid_id,
            file=c.file_name,
            page=c.page_number,
            section=c.section,
            quote=c.text,
        )


class FieldValue(Contract):
    value: Any = None
    sources: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(0, ge=0, le=1)
    notes: str = "Not found in documents"

    @model_validator(mode="after")
    def require_sources(self):
        """Make unsupported non-null values impossible at contract boundaries."""
        if self.value is not None and not self.sources:
            raise ValueError("Every non-null value must have a citation")
        if self.value is not None and self.notes == "Not found in documents":
            self.notes = ""
        if self.value is None:
            self.confidence = 0
            if not self.notes.strip():
                self.notes = "Not found in documents"
        return self


class FieldBatch(Contract):
    fields: dict[str, FieldValue] = Field(default_factory=dict)


class Change(Contract):
    field: str
    previous_value: Any = None
    new_value: Any = None
    addendum_number: int
    sources: list[Evidence]
    explanation: str


class Reconciliation(FieldBatch):
    addendum_changes: list[Change] = Field(default_factory=list)


class ValidationReport(Contract):
    passed: int = 0
    failed: int = 0
    not_found: int = 0
    rejected: dict[str, str] = Field(default_factory=dict)


class Critique(Contract):
    rejected: dict[str, str] = Field(default_factory=dict)


class Extraction(Contract):
    bid_id: str
    fields: dict[str, FieldValue]
    addendum_changes: list[Change] = Field(default_factory=list)
    validation: ValidationReport
    run_id: str
    status: Literal["completed", "partial"]
    errors: list[str] = Field(default_factory=list)


class Answer(Contract):
    answer: str
    sources: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(0, ge=0, le=1)
    notes: str = ""


class Plan(Contract):
    tasks: list[str]


class RunState(Contract):
    run_id: str
    bid_ids: list[str]
    plan: list[str] = Field(default_factory=list)
    evidence: dict[str, Chunk] = Field(default_factory=dict)
    drafts: dict[str, FieldValue] = Field(default_factory=dict)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    retries: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    final_output: dict[str, Any] | None = None
