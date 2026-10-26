"""Process-local activity for the seven assignment roles, including concurrent specialists."""

from collections import Counter
from contextlib import contextmanager

ROLES = (
    ("planner", "Orchestrator", "Plans and coordinates extraction tasks"),
    ("ingestion", "Ingestion", "Parses documents and updates the index"),
    ("retrieval", "Retrieval", "Finds and ranks source evidence"),
    ("extraction", "Extraction", "Dates, legal and product specialists"),
    ("reconciliation", "Addendum reconciliation", "Applies valid amendments"),
    ("validation", "Validator", "Checks evidence and rejected claims"),
    ("qa", "Q&A", "Writes cited answers to your questions"),
)
STEP_ROLES = {
    "dates_logistics": "extraction",
    "commercial_legal": "extraction",
    "products_specs": "extraction",
    "field_repair": "extraction",
    "addendum_reconciliation": "reconciliation",
    "validator": "validation",
    "qa_validator": "validation",
    "qa_report": "qa",
}


class AgentActivity:
    """Count actual executing steps, never configured agents or queued work as running."""

    def __init__(self):
        self.counts = Counter()

    @contextmanager
    def track(self, step):
        """Release activity on success, exception or cancellation; preserve nested counts."""
        role = STEP_ROLES.get(step, step)
        self.counts[role] += 1
        try:
            yield
        finally:
            self.counts[role] -= 1

    def snapshot(self):
        """Return public role descriptions and current concurrent executions."""
        return [
            {"id": role, "name": name, "description": description, "active_steps": self.counts[role]}
            for role, name, description in ROLES
        ]
