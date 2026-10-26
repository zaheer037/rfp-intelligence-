"""Google ADK orchestration with scoped evidence, parallel specialists and bounded feedback."""

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.events import Event, EventActions
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import Field, ValidationError

from rfp.agents.activity import AgentActivity
from rfp.agents.prompts import GROUPS, GUARDRAILS, QUERIES, ROLE_INSTRUCTIONS
from rfp.agents.validation import (
    bind_citations,
    bind_sources,
    citation_error,
    validate_answer,
    validate_fields,
)
from rfp.core.config import Settings
from rfp.core.observability import LLM_CALLS, RETRIES, TOKENS, VALIDATION, Trace, stage
from rfp.schemas import (
    FIELDS,
    Answer,
    Critique,
    Extraction,
    FieldBatch,
    FieldValue,
    Plan,
    Reconciliation,
    RunState,
)
from rfp.utils.identifiers import validate_bid_id


class Agent:
    """Application facade: each run owns its state, evidence, retries and trace."""

    def __init__(self, settings: Settings, indexing, retrieval):
        self.settings = settings
        self.indexing = indexing
        self.retrieval = retrieval
        self.activity = AgentActivity()
        self._search_slots = asyncio.Semaphore(settings.max_search_concurrency)
        self._llm_slots = asyncio.Semaphore(3)

    def _trace(self, state: RunState) -> Trace:
        return Trace(self.settings.storage_dir / "traces", state.run_id, self.settings.trace_content)

    def _build_model(self):
        """Keep the provider boundary replaceable without changing orchestration."""
        return LiteLlm(
            model=self.settings.llm_model,
            api_key=self.settings.groq_api_key.get_secret_value(),
            timeout=self.settings.llm_timeout_seconds,
            num_retries=0,
            reasoning_effort="medium",
            response_format={"type": "json_object"},
        )

    async def _call(self, role: str, payload: dict, schema, trace: Trace):
        """Run one typed ADK LLM step, logging actual usage and retrying bad responses once."""
        if not self.settings.groq_api_key.get_secret_value():
            raise RuntimeError("GROQ_API_KEY is required for agent tasks")
        payload = {**payload, "response_contract": schema.model_json_schema()}
        for attempt in range(2):
            started = time.perf_counter()
            trace.write(role, input=payload, attempt=attempt, event="started")
            try:
                async with self._llm_slots, asyncio.timeout(self.settings.llm_timeout_seconds):
                    with self.activity.track(role):
                        model = self._build_model()
                        agent = LlmAgent(
                            name=role,
                            model=model,
                            instruction=GUARDRAILS + "\n" + ROLE_INSTRUCTIONS[role],
                            generate_content_config=types.GenerateContentConfig(
                                temperature=0, max_output_tokens=12000
                            ),
                        )
                        sessions = InMemorySessionService()
                        session = await sessions.create_session(app_name="rfp", user_id=trace.run_id)
                        runner = Runner(agent=agent, app_name="rfp", session_service=sessions)
                        output = ""
                        with stage("llm"):
                            async for response in runner.run_async(
                                user_id=trace.run_id,
                                session_id=session.id,
                                new_message=types.Content(
                                    role="user", parts=[types.Part(text=json.dumps(payload))]
                                ),
                            ):
                                usage = response.usage_metadata
                                if usage:
                                    counts = {
                                        "input": usage.prompt_token_count or 0,
                                        "output": usage.candidates_token_count or 0,
                                        "cached": usage.cached_content_token_count or 0,
                                    }
                                    for direction, count in counts.items():
                                        TOKENS.labels(direction).inc(count)
                                    trace.write(role, event="usage", tokens=counts)
                                if response.is_final_response() and response.content:
                                    output = "".join(
                                        p.text or "" for p in response.content.parts or [] if not p.thought
                                    )
                        parsed = schema.model_validate_json(output)
                LLM_CALLS.labels(role, "success").inc()
                trace.write(
                    role,
                    output=parsed.model_dump(),
                    event="completed",
                    latency_seconds=time.perf_counter() - started,
                )
                return parsed
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LLM_CALLS.labels(role, "error").inc()
                trace.write(
                    role,
                    event="failed",
                    error_type=type(exc).__name__,
                    latency_seconds=time.perf_counter() - started,
                )
                if attempt:
                    raise RuntimeError(f"{role} failed: {type(exc).__name__}") from exc
                if isinstance(exc, ValidationError):
                    payload = {
                        **payload,
                        "response_validation_errors": exc.errors(include_input=False, include_url=False),
                        "repair_instruction": "Correct these contract errors. Sources must be JSON objects, never JSON-encoded strings. Return only the corrected JSON object.",
                    }
                RETRIES.labels(role).inc()
                await asyncio.sleep(0.5)
        raise AssertionError("unreachable")

    async def _search(self, state: RunState, query: str, bid_id: str, trace: Trace, **filters) -> list[dict]:
        """Retrieval agent tool: search only the requested bid and register returned evidence."""
        started = time.perf_counter()
        async with self._search_slots:
            with self.activity.track("retrieval"):
                hits = await asyncio.to_thread(self.retrieval.search, query, bid_id=bid_id, **filters)
        for hit in hits:
            state.evidence[hit.chunk.chunk_id] = hit.chunk
        trace.write(
            "retrieval_agent",
            tool_arguments={"query": query, "bid_id": bid_id, **filters},
            tool_result=[h.model_dump() for h in hits],
            hits=len(hits),
            latency_seconds=time.perf_counter() - started,
        )
        return [h.model_dump() for h in hits]

    async def ingest(self, folder: str, bid_id: str | None = None) -> dict:
        """Ingestion agent delegates its complete pipeline to Indexing, then reloads search."""
        identifier = validate_bid_id(bid_id or Path(folder).name)
        trace = Trace(self.settings.storage_dir / "traces", uuid.uuid4().hex, self.settings.trace_content)
        trace.write("ingestion_agent", event="started", input={"folder": folder, "bid_id": identifier})
        with self.activity.track("ingestion"):
            report = await asyncio.to_thread(self.indexing.index_folder, folder, identifier)
            await asyncio.to_thread(self.retrieval.reload, identifier)
        trace.write("ingestion_agent", event="completed", output=report)
        return report

    async def process_folder(self, folder: str, bid_id: str | None = None) -> Extraction:
        """Run ingestion through final validated extraction for a new or existing bid folder."""
        if not self.settings.groq_api_key.get_secret_value():
            raise RuntimeError("Configure GROQ_API_KEY before processing a bid")
        identifier = validate_bid_id(bid_id or Path(folder).name)
        await self.ingest(folder, identifier)
        return await self.extract(identifier)

    async def _extract_fields(self, state, bid_id, names, trace, feedback=None):
        """Search per field, then ask the appropriate specialist for only those fields."""
        passages = {}
        for name in names:
            hits = await self._search(state, QUERIES[name], bid_id, trace, top_k=5)
            if name == "Due Date":
                hits.extend(
                    await self._search(
                        state,
                        "submission closing date time timezone",
                        bid_id,
                        trace,
                        doc_type="bid_page",
                        top_k=2,
                    )
                )
            if name == "contact_info":
                hits.extend(
                    await self._search(
                        state, "procurement contact email phone", bid_id, trace, doc_type="bid_page", top_k=2
                    )
                )
            passages[name] = hits
        role = next((key for key, fields in GROUPS.items() if set(names) <= set(fields)), "field_repair")
        result = await self._call(
            role,
            {
                "task": "Extract exactly these fields",
                "fields": names,
                "evidence_by_field": passages,
                "validation_feedback": feedback or {},
                "summary_rule": "Bid Summary must have 3–6 cited sentences",
            },
            FieldBatch,
            trace,
        )
        return bind_sources(
            {
                name: result.fields.get(
                    name, FieldValue(notes="Extraction failed: agent omitted required field")
                )
                for name in names
            },
            state.evidence,
        )

    async def extract(self, bid_id: str, run_id: str | None = None) -> Extraction:
        """Produce every required field, reconcile addendums and repair rejected fields only."""
        validate_bid_id(bid_id)
        if not self.settings.groq_api_key.get_secret_value():
            raise RuntimeError("GROQ_API_KEY is required for extraction")
        if bid_id not in self.retrieval.bid_ids():
            raise ValueError("Index the bid before extraction")
        state = RunState(run_id=run_id or uuid.uuid4().hex, bid_ids=[bid_id])
        trace = self._trace(state)
        snapshot = await asyncio.to_thread(self.indexing.load, bid_id)
        if snapshot:
            state.errors.extend(
                "Ingestion warning: " + warning for warning in snapshot[2].get("warnings", [])
            )
        with stage("extraction"):
            try:
                plan = await self._call(
                    "planner",
                    {
                        "task": "Plan complete 20-field extraction",
                        "bid_id": bid_id,
                        "groups": GROUPS,
                        "required_steps": [
                            "retrieve",
                            "extract",
                            "reconcile",
                            "validate",
                            "repair",
                            "report",
                        ],
                    },
                    Plan,
                    trace,
                )
                state.plan = plan.tasks
            except RuntimeError as exc:
                state.errors.append(str(exc))
                state.plan = ["retrieve", "parallel_extract", "reconcile", "validate", "repair", "report"]
            results = await asyncio.gather(
                *[self._extract_fields(state, bid_id, names, trace) for names in GROUPS.values()],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    state.errors.append(str(result))
                else:
                    state.drafts.update(result)
            for name in FIELDS:
                state.drafts.setdefault(name, FieldValue(notes="Extraction failed; review run errors"))
            changes = []
            try:
                # Retrieve both base and addendum evidence: changes must have before/after support.
                addenda = await self._search(
                    state,
                    "addendum changes revised requirements questions answers deadline",
                    bid_id,
                    trace,
                    doc_type="addendum",
                    top_k=30,
                )
                if addenda:
                    reconciled = await self._call(
                        "addendum_reconciliation",
                        {
                            "task": "Apply only explicit field changes in valid addendum order. Preserve unchanged fields. "
                            "Return all 20 fields. Cite both old and new evidence for every change. "
                            "Flag contradictory timezones without silently converting CST to CDT.",
                            "fields": {k: v.model_dump() for k, v in state.drafts.items()},
                            "addendum_passages": addenda,
                            "base_evidence": [
                                c.model_dump() for c in state.evidence.values() if c.doc_type != "addendum"
                            ],
                        },
                        Reconciliation,
                        trace,
                    )
                    state.drafts.update(
                        bind_sources(
                            {k: v for k, v in reconciled.fields.items() if k in FIELDS}, state.evidence
                        )
                    )
                    changes = reconciled.addendum_changes
            except RuntimeError as exc:
                state.errors.append(str(exc))
            for attempt in range(self.settings.max_field_retries + 1):
                state.validation = validate_fields(state.drafts, state.evidence)
                try:
                    critique = await self._call(
                        "validator",
                        {
                            "task": "Check all 20 fields, including null fields for overlooked evidence. Reject false Not found. "
                            "Reject any populated field not entailed by evidence, missing required detail, "
                            "invalid dates/numbers, unresolved contradictions or stale addendum values. "
                            "Consider each field and supporting quote. Return rejected field names and reasons.",
                            "fields": {k: v.model_dump() for k, v in state.drafts.items()},
                            "evidence": [c.model_dump() for c in state.evidence.values()],
                        },
                        Critique,
                        trace,
                    )
                    state.validation.rejected.update(
                        {k: v for k, v in critique.rejected.items() if k in FIELDS}
                    )
                except RuntimeError as exc:
                    state.errors.append(str(exc))
                rejected = state.validation.rejected
                trace.write("validator", output=state.validation.model_dump(), attempt=attempt)
                if not rejected or attempt == self.settings.max_field_retries:
                    break
                names = list(rejected)
                for name in names:
                    state.retries[name] = attempt + 1
                try:
                    state.drafts.update(await self._extract_fields(state, bid_id, names, trace, rejected))
                except RuntimeError as exc:
                    state.errors.append(str(exc))
            rejected = dict(state.validation.rejected)
            for name, reason in rejected.items():
                state.drafts[name] = FieldValue(notes=f"Validation failed after retries: {reason}")
            # A change log must never retain fabricated citations or describe a rejected final value.
            changes = [
                change
                for change in changes
                if change.field in FIELDS
                and change.field not in rejected
                and change.sources
                and change.new_value == state.drafts[change.field].value
                and all(not citation_error(s, state.evidence) for s in change.sources)
                and any(
                    state.evidence[s.chunk_id].doc_type == "addendum"
                    and state.evidence[s.chunk_id].addendum_number == change.addendum_number
                    for s in change.sources
                )
            ]
            report = validate_fields(state.drafts, state.evidence)
            report.rejected = rejected
            report.failed = len(rejected)
            report.not_found -= len(rejected)
            for outcome in ("passed", "failed", "not_found"):
                VALIDATION.labels(outcome).inc(getattr(report, outcome))
            result = Extraction(
                bid_id=bid_id,
                fields=state.drafts,
                addendum_changes=changes,
                validation=report,
                run_id=state.run_id,
                status="partial" if rejected or state.errors else "completed",
                errors=state.errors,
            )
            state.final_output = result.model_dump()
            trace.write("report_agent", output=state.final_output, event="completed", status=result.status)
            return result

    async def ask(self, question: str, bid_ids: list[str] | None = None, run_id: str | None = None) -> Answer:
        """Retrieve evidence per bid so comparison questions cannot lose one bid's context."""
        ids = bid_ids or self.retrieval.bid_ids()
        if not ids or any(bid not in self.retrieval.bid_ids() for bid in ids):
            raise ValueError("Requested bids must be indexed")
        state = RunState(run_id=run_id or uuid.uuid4().hex, bid_ids=ids)
        trace = self._trace(state)
        evidence = []
        for bid in ids:
            evidence.extend(await self._search(state, question, bid, trace, top_k=8))
            evidence.extend(await self._search(state, question, bid, trace, doc_type="addendum", top_k=5))
        if not evidence:
            return Answer(answer="Not found in documents")
        answer = await self._call(
            "qa_report",
            {
                "question": question,
                "evidence": evidence,
                "task": "Answer with citations, using applicable addendum changes",
            },
            Answer,
            trace,
        )
        answer.sources = bind_citations(answer.sources, state.evidence)
        answer = validate_answer(answer, state.evidence)
        if answer.sources:
            critique = await self._call(
                "qa_validator",
                {
                    "task": "Reject answer if any claim is unsupported or conflicts with evidence. "
                    "Use key 'answer' in rejected for a failed answer.",
                    "answer": answer.model_dump(),
                    "evidence": evidence,
                },
                Critique,
                trace,
            )
            if critique.rejected:
                answer = Answer(
                    answer="Not found in documents", notes="Answer rejected by evidence validator"
                )
        trace.write("qa_report", event="completed", output=answer.model_dump())
        return answer


class RfpWorkflow(BaseAgent):
    """ADK root agent used by the A2A adapter; delegates to the same tested backend facade."""

    service: Any = Field(exclude=True)

    async def _run_async_impl(self, ctx):
        """Accept natural-language questions or typed JSON extraction/index task messages."""
        content = ctx.user_content
        text = "".join(p.text or "" for p in content.parts or []) if content else ""
        try:
            task = json.loads(text) if text.strip().startswith("{") else {"task": "ask", "question": text}
            if task.get("task") == "extract":
                result = (await self.service.extract(validate_bid_id(task["bid_id"]))).model_dump()
            elif task.get("task") == "index":
                result = await self.service.ingest(task["folder"], task.get("bid_id"))
            elif task.get("task") == "ask":
                question = task["question"]
                if not isinstance(question, str) or not 1 <= len(question) <= 2000:
                    raise ValueError("Question must be 1–2000 characters")
                result = (await self.service.ask(question, task.get("bid_ids"))).model_dump()
            else:
                raise ValueError("Task must be ask, index or extract")
        except Exception as exc:
            result = {"error": type(exc).__name__, "message": "Task failed; inspect the run trace"}
        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=json.dumps(result))]),
            actions=EventActions(state_delta={"last_result": result}),
        )
