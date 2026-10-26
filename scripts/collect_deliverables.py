"""Run real extraction and 10 cited questions, preserving actual outputs and traces."""

import asyncio
import json
from pathlib import Path

from rfp.agents.agent import Agent
from rfp.core.config import Settings
from rfp.core.observability import configure_logging
from rfp.ingestion.indexing import Indexing
from rfp.search.models import LocalEmbeddings, LocalReranker
from rfp.search.retrieval import Retrieval


async def main():
    settings = Settings()
    if not settings.groq_api_key.get_secret_value():
        raise SystemExit("Set GROQ_API_KEY in .env before collecting live deliverables")
    configure_logging(settings.log_level)
    embedding = LocalEmbeddings(settings)
    indexing = Indexing(settings, embedding)
    retrieval = Retrieval(indexing, embedding, LocalReranker(settings))
    agent = Agent(settings, indexing, retrieval)
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    for bid in ("Bid1", "Bid2"):
        await agent.ingest(bid)
        result = await agent.extract(bid)
        (output / f"{bid}.json").write_text(result.model_dump_json(indent=2))
    questions = json.loads(Path("eval/questions.json").read_text())[:10]
    answers = []
    for row in questions:
        result = await agent.ask(row["question"], [row["bid_id"]])
        answers.append({"question": row["question"], **result.model_dump()})
        # Preserve completed real answers even if a later provider request fails.
        (output / "qa-log.json").write_text(json.dumps(answers, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
