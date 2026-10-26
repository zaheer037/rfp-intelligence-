"""Validated REST request bodies; domain/agent output contracts live in rfp.schemas."""

from pydantic import Field, field_validator

from rfp.schemas import Contract
from rfp.utils.identifiers import validate_bid_id


class IndexRequest(Contract):
    folder: str = Field(min_length=1, max_length=500)
    bid_id: str | None = None

    @field_validator("bid_id")
    @classmethod
    def valid_id(cls, value):
        return validate_bid_id(value) if value else value


class ExtractRequest(Contract):
    bid_id: str

    @field_validator("bid_id")
    @classmethod
    def valid_id(cls, value):
        return validate_bid_id(value)


class AskRequest(Contract):
    question: str = Field(min_length=1, max_length=2000)
    bid_ids: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("bid_ids")
    @classmethod
    def valid_ids(cls, values):
        return [validate_bid_id(v) for v in values]
