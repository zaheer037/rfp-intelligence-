"""Portable bid identifiers shared by ingestion, API and agent boundaries."""

import re


def validate_bid_id(value: str) -> str:
    """Allow portable folder identifiers, never paths or traversal characters."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("bid_id must be 1–80 letters, digits, underscores or hyphens")
    return value
