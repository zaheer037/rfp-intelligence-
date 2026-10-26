"""Deterministic evidence gates complement the semantic validator agent."""

import re

from rfp.schemas import FIELDS, Answer, Chunk, Evidence, FieldValue, ValidationReport


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def citation_error(source: Evidence, evidence: dict[str, Chunk]) -> str | None:
    """Verify a quote against the retrieved evidence, including file/page/bid identity."""
    chunk = evidence.get(source.chunk_id)
    if chunk is None:
        return "Citation was not retrieved in this run"
    if (source.bid_id, source.file, source.page) != (chunk.bid_id, chunk.file_name, chunk.page_number):
        return "Citation metadata does not match stored source"
    if normalized(source.quote) not in normalized(chunk.text) or len(source.quote.strip()) < 8:
        return "Citation quote is missing, too short or not verbatim"
    return None


def validate_fields(fields: dict[str, FieldValue], evidence: dict[str, Chunk]) -> ValidationReport:
    """Check field coverage, verifiable citations and key date/summary format rules."""
    report = ValidationReport()
    for name in FIELDS:
        item = fields.get(name)
        error = None
        if item is None:
            error = "Required output field missing"
        elif item.value is None:
            if item.notes.startswith("Extraction failed"):
                error = "Agent omitted this field or extraction failed"
            else:
                error = next((e for source in item.sources if (e := citation_error(source, evidence))), None)
                if not error:
                    report.not_found += 1
                    continue
        else:
            error = next((e for source in item.sources if (e := citation_error(source, evidence))), None)
            if not error and name == "Due Date":
                text = str(item.value)
                if not re.search(r"\d{4}", text) or not re.search(r"\d{1,2}:\d{2}", text):
                    error = "Deadline must include date and time, or be null with reason"
                elif not re.search(
                    r"\b(?:UTC|GMT|CST|CDT|EST|EDT|PST|PDT|MST|MDT|IST|BST|CET|CEST|[A-Za-z]+/[A-Za-z_]+)\b|[+-]\d{2}:\d{2}|T\d{2}:\d{2}.*Z",
                    text,
                ):
                    error = "Deadline timezone missing"
            if not error and name == "Bid Summary":
                count = len(re.findall(r"[.!?](?:\s|$)", str(item.value)))
                if not 3 <= count <= 6:
                    error = "Summary must have 3–6 sentences"
        if error:
            report.rejected[name] = error
        else:
            report.passed += 1
    report.failed = len(report.rejected)
    return report


def validate_answer(answer: Answer, evidence: dict[str, Chunk]) -> Answer:
    """Abstain if a generated answer lacks real, in-scope citations."""
    if not answer.sources or any(citation_error(source, evidence) for source in answer.sources):
        return Answer(
            answer="Not found in documents", confidence=0, notes="No valid supporting citations were returned"
        )
    return answer


def bind_sources(fields: dict[str, FieldValue], evidence: dict[str, Chunk]) -> dict[str, FieldValue]:
    """Use stored passage text for citations with valid identities; never trust model-generated quotes.

    Field entailment still goes through the semantic critic. Unknown IDs and incorrect metadata
    are deliberately left invalid so the deterministic citation gate rejects them.
    """
    for field in fields.values():
        field.sources = bind_citations(field.sources, evidence)
    return fields


def bind_citations(sources: list[Evidence], evidence: dict[str, Chunk]) -> list[Evidence]:
    """Resolve identity-checked citations to stored text before checking semantic entailment."""
    for source in sources:
        chunk = evidence.get(source.chunk_id)
        if chunk and (source.bid_id, source.file, source.page) == (
            chunk.bid_id,
            chunk.file_name,
            chunk.page_number,
        ):
            source.quote = chunk.text
            source.section = chunk.section
    return sources
