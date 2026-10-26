"""Runnable checks for parsing, persistence, search, API trust boundaries and real ADK execution."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient

from rfp.agents.validation import citation_error
from rfp.api.jobs import JobManager
from rfp.core.config import Settings
from rfp.ingestion.indexing import Indexing
from rfp.main import create_app
from rfp.schemas import FIELDS, FieldValue
from rfp.search.retrieval import Retrieval
from tests.support import TestAgent, TestEmbeddings, TestReranker


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = Settings(
            _env_file=None,
            environment="development",
            input_root=self.root,
            storage_dir=self.root / "store",
            api_key="test-key",
            groq_api_key="synthetic-test-key",
            ocr_enabled=False,
            chunk_words=100,
            chunk_overlap=15,
            log_level="ERROR",
        )
        self.embeddings, self.reranker = TestEmbeddings(), TestReranker()
        self.indexing = Indexing(self.settings, self.embeddings)
        self.retrieval = Retrieval(self.indexing, self.embeddings, self.reranker)
        for bid, content in (
            ("NewBid", "Dell ABC-123 warranty three years 16 GB memory"),
            ("OtherBid", "Apple XYZ-999 warranty one year 8 GB memory"),
        ):
            folder = self.root / bid
            folder.mkdir()
            (folder / "specs.html").write_text(
                f"<html><nav>REMOVE ME</nav><main><h1>SPECIFICATIONS</h1>"
                f"<table><tr><th>Product</th><th>SKU</th></tr>"
                f"<tr><td>{content}</td><td>SKU-777</td></tr></table></main></html>"
            )

    def tearDown(self):
        self.temp.cleanup()

    def seed(self):
        for bid in ("NewBid", "OtherBid"):
            self.indexing.index_folder(bid)
        self.retrieval.reload()

    def test_html_tables_and_metadata(self):
        self.seed()
        chunks = list(self.retrieval.chunks(["NewBid"]).values())
        self.assertIn("|", chunks[0].text)
        self.assertNotIn("REMOVE ME", chunks[0].text)
        self.assertIsNone(chunks[0].page_number)
        self.assertEqual(chunks[0].doc_type, "bid_page")

    def test_incremental_update_delete_and_corrupt_file(self):
        self.seed()
        count = self.embeddings.encoded_documents
        self.assertEqual(self.indexing.index_folder("NewBid")["status"], "unchanged")
        self.assertEqual(self.embeddings.encoded_documents, count)
        folder = self.root / "NewBid"
        pdf = folder / "Addendum 2.pdf"
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((50, 50), "ADDENDUM No. 2\nNew due date July 9, 2024 at 2:00 PM CST.")
            doc.save(pdf)
        report = self.indexing.index_folder("NewBid")
        self.assertEqual(report["status"], "indexed")
        self.retrieval.reload("NewBid")
        hit = self.retrieval.search("new due date", bid_id="NewBid", doc_type="addendum", addendum_number=2)[
            0
        ]
        self.assertEqual(hit.chunk.page_number, 1)
        self.assertEqual(hit.chunk.addendum_number, 2)
        pdf.write_bytes(b"not a PDF")
        self.assertEqual(self.indexing.index_folder("NewBid")["status"], "partial")
        self.retrieval.reload("NewBid")
        self.assertTrue(self.retrieval.search("due date", bid_id="NewBid", doc_type="addendum"))
        pdf.unlink()
        self.indexing.index_folder("NewBid")
        self.retrieval.reload("NewBid")
        self.assertFalse(self.retrieval.search("date", bid_id="NewBid", doc_type="addendum"))
        self.assertIn("OtherBid", self.retrieval.bid_ids())

    def test_incompatible_snapshot_can_be_rebuilt_without_blocking_startup(self):
        self.seed()
        self.embeddings.fingerprint = "changed-test-model"
        self.retrieval.reload()
        self.assertFalse(self.retrieval.bid_ids())
        self.assertIn("NewBid", self.retrieval.unavailable_bids)
        self.indexing.index_folder("NewBid")
        self.retrieval.reload("NewBid")
        self.assertIn("NewBid", self.retrieval.bid_ids())
        self.assertNotIn("NewBid", self.retrieval.unavailable_bids)
        kind, _ = self.indexing._classify(
            Path("solicitation.pdf"), "REQUEST FOR PROPOSALS\nInclude a Mercury Affidavit with your offer."
        )
        self.assertEqual(kind, "rfp")

    def test_filters_and_exact_identifiers(self):
        self.seed()
        for mode in ("dense", "hybrid", "hybrid_rerank"):
            hits = self.retrieval.search("ABC-123 SKU-777", bid_id="NewBid", mode=mode)
            self.assertTrue(hits)
            self.assertTrue(all(h.chunk.bid_id == "NewBid" for h in hits))
        self.assertFalse(self.retrieval.search("warranty", bid_id="NewBid", doc_type="affidavit"))
        with self.assertRaises(ValueError):
            self.retrieval.search("x", bid_id="../NewBid")

    def test_path_boundaries_and_missing_ocr(self):
        with self.assertRaises(ValueError):
            self.indexing.resolve_folder("../")
        link = self.root / "NewBid" / "outside.html"
        link.symlink_to(self.root / "OtherBid" / "specs.html")
        with self.assertRaises(ValueError):
            self.indexing.index_folder("NewBid")
        link.unlink()
        from PIL import Image

        image = self.root / "NewBid" / "scan.png"
        Image.new("RGB", (100, 100), "white").save(image)
        report = self.indexing.index_folder("NewBid")
        self.assertEqual(report["status"], "partial")
        self.assertTrue(any("OCR disabled" in w for w in report["warnings"]))
        result = asyncio.run(TestAgent(self.settings, self.indexing, self.retrieval).process_folder("NewBid"))
        self.assertEqual(result.status, "partial")
        self.assertTrue(any("Ingestion warning" in error for error in result.errors))

    def test_citations_reject_forgery_and_unsupported_values(self):
        self.seed()
        hit = self.retrieval.search("ABC-123", bid_id="NewBid")[0]
        evidence = {hit.chunk.chunk_id: hit.chunk}
        source = hit.citation()
        self.assertIsNone(citation_error(source, evidence))
        source.quote = "forged quote that does not exist"
        self.assertTrue(citation_error(source, evidence))
        source = hit.citation()
        source.page = 999
        self.assertTrue(citation_error(source, evidence))
        with self.assertRaises(ValueError):
            FieldValue(value="invented", sources=[])

    def test_failed_publish_keeps_previous_generation(self):
        self.seed()
        pointer = (self.indexing.root / "NewBid" / "current.json").read_text()
        (self.root / "NewBid" / "specs.html").write_text("<main>changed warranty</main>")
        with patch("rfp.ingestion.indexing.faiss.write_index", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.indexing.index_folder("NewBid")
        self.assertEqual(pointer, (self.indexing.root / "NewBid" / "current.json").read_text())
        self.assertTrue(self.indexing.load("NewBid"))

    def test_thought_parts_do_not_break_structured_outputs(self):
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types

        from rfp.schemas import Plan, RunState
        from tests.support import ScriptedLlm

        class ThinkingLlm(ScriptedLlm):
            async def generate_content_async(self, llm_request, stream=False):
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(text="Private reasoning before JSON", thought=True),
                            types.Part(text='{"tasks":["retrieve"]}'),
                        ],
                    )
                )

        class ThinkingAgent(TestAgent):
            def _build_model(self):
                return ThinkingLlm()

        agent = ThinkingAgent(self.settings, self.indexing, self.retrieval)
        state = RunState(run_id="thought-regression", bid_ids=[])
        result = asyncio.run(agent._call("planner", {}, Plan, agent._trace(state)))
        self.assertEqual(result.tasks, ["retrieve"])

    def test_live_agent_counts_release_after_failure(self):
        from rfp.agents.activity import AgentActivity

        activity = AgentActivity()
        self.assertEqual(len(activity.snapshot()), 7)
        with self.assertRaises(RuntimeError):
            with activity.track("dates_logistics"), activity.track("products_specs"):
                roles = {role["id"]: role["active_steps"] for role in activity.snapshot()}
                self.assertEqual(roles["extraction"], 2)
                with activity.track("qa_validator"):
                    self.assertEqual(sum(role["active_steps"] for role in activity.snapshot()), 3)
                raise RuntimeError("provider failure")
        self.assertEqual(sum(role["active_steps"] for role in activity.snapshot()), 0)

    def test_citation_binding_preserves_identity_gate(self):
        from rfp.agents.validation import bind_citations

        self.seed()
        chunk = self.retrieval.search("warranty", bid_id="NewBid")[0].chunk
        from rfp.schemas import Evidence

        source = Evidence(
            chunk_id=chunk.chunk_id,
            bid_id=chunk.bid_id,
            file=chunk.file_name,
            page=chunk.page_number,
            quote="model paraphrase",
        )
        self.assertIsNotNone(citation_error(source, {chunk.chunk_id: chunk}))
        bind_citations([source], {chunk.chunk_id: chunk})
        self.assertIsNone(citation_error(source, {chunk.chunk_id: chunk}))
        source.file = "forged.pdf"
        source.quote = "model paraphrase"
        bind_citations([source], {chunk.chunk_id: chunk})
        self.assertIsNotNone(citation_error(source, {chunk.chunk_id: chunk}))

    def test_json_encoded_citations_keep_strict_evidence_validation(self):
        import json

        from pydantic import ValidationError

        from rfp.schemas import Evidence

        citation = {
            "chunk_id": "abc",
            "bid_id": "NewBid",
            "file": "specs.html",
            "page": None,
            "quote": "A source passage",
        }
        self.assertEqual(Evidence.model_validate(json.dumps(citation)).chunk_id, "abc")
        for invalid in ('"abc"', '{"chunk_id":"abc"}', "not-json"):
            with self.assertRaises(ValidationError):
                Evidence.model_validate(invalid)

    def test_schema_retry_receives_actionable_feedback(self):
        import json

        from google.adk.models.llm_response import LlmResponse
        from google.genai import types

        from rfp.schemas import Plan, RunState
        from tests.support import ScriptedLlm

        seen = []

        class RepairLlm(ScriptedLlm):
            async def generate_content_async(self, llm_request, stream=False):
                payload = json.loads(llm_request.contents[-1].parts[0].text)
                seen.append(payload)
                text = '{"tasks":"wrong type"}' if len(seen) == 1 else '{"tasks":["retrieve"]}'
                yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))

        class RepairAgent(TestAgent):
            def _build_model(self):
                return RepairLlm()

        agent = RepairAgent(self.settings, self.indexing, self.retrieval)
        state = RunState(run_id="schema-repair-regression", bid_ids=[])
        result = asyncio.run(agent._call("planner", {}, Plan, agent._trace(state)))
        self.assertEqual(result.tasks, ["retrieve"])
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1]["response_validation_errors"][0]["loc"], ["tasks"])
        self.assertNotIn("input", seen[1]["response_validation_errors"][0])

    def test_real_adk_runner_and_full_workflow(self):
        self.seed()
        agent = TestAgent(self.settings, self.indexing, self.retrieval)
        answer = asyncio.run(agent.ask("Which SKU?", ["NewBid"]))
        self.assertTrue(answer.sources)
        result = asyncio.run(agent.process_folder("NewBid"))
        self.assertEqual(set(result.fields), set(FIELDS))
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.validation.passed + result.validation.not_found, 20)
        self.assertEqual(result.validation.failed, 0)
        traces = list((self.settings.storage_dir / "traces").glob("*.jsonl"))
        self.assertTrue(any('"tokens"' in p.read_text() for p in traces))

    def test_validator_feedback_has_a_hard_retry_bound(self):
        from rfp.schemas import Critique

        class RejectingAgent(TestAgent):
            validation_calls = 0

            async def _call(self, role, payload, schema, trace):
                if role == "validator":
                    self.validation_calls += 1
                    return Critique(rejected={"Product": "Synthetic unsupported product"})
                return await super()._call(role, payload, schema, trace)

        self.seed()
        agent = RejectingAgent(self.settings, self.indexing, self.retrieval)
        result = asyncio.run(agent.extract("NewBid"))
        self.assertEqual(agent.validation_calls, self.settings.max_field_retries + 1)
        self.assertIsNone(result.fields["Product"].value)
        self.assertEqual(result.validation.failed, 1)
        self.assertEqual(result.status, "partial")
        self.assertEqual(
            result.validation.passed + result.validation.failed + result.validation.not_found, 20
        )

    def test_a2a_discovery_and_rpc_share_rest_auth(self):
        app = create_app(self.settings, embeddings=self.embeddings, reranker=self.reranker)
        with TestClient(app) as client:
            path = "/a2a/.well-known/agent-card.json"
            self.assertEqual(client.get(path).status_code, 401)
            headers = {"X-API-Key": "test-key"}
            card = client.get(path, headers=headers)
            self.assertEqual(card.status_code, 200)
            self.assertIn("/a2a/", card.text)
            response = client.post(
                "/a2a/",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": "test-rpc",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "messageId": "test-message",
                            "parts": [{"kind": "text", "text": "What is the deadline?"}],
                        }
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("result", response.json())

    def test_workspace_upload_metadata_and_document_boundaries(self):
        import base64
        import time

        app = create_app(
            self.settings, embeddings=self.embeddings, reranker=self.reranker, agent_factory=TestAgent
        )
        headers = {"X-API-Key": "test-key"}
        with TestClient(app) as client:
            encoded = base64.b64encode(b"<main>Public laptop bid. Delivery within 45 days.</main>").decode()
            bad = client.post(
                "/upload",
                headers=headers,
                json={"bid_id": "UploadQA", "files": [{"name": "../escape.html", "data": encoded}]},
            )
            self.assertEqual(bad.status_code, 400)
            result = client.post(
                "/upload",
                headers=headers,
                json={"bid_id": "UploadQA", "files": [{"name": "notice.html", "data": encoded}]},
            )
            self.assertEqual(result.status_code, 202)
            for _ in range(100):
                job = client.get(result.json()["status_url"], headers=headers).json()
                if job["status"] != "running":
                    break
                time.sleep(0.02)
            self.assertEqual(job["status"], "completed")
            self.assertIn(
                "UploadQA", [b["bid_id"] for b in client.get("/bids", headers=headers).json()["bids"]]
            )
            detail = client.get("/bids/UploadQA", headers=headers).json()
            self.assertEqual(detail["documents"][0]["name"], "notice.html")
            source = client.get("/bids/UploadQA/documents/notice.html", headers=headers)
            self.assertEqual(source.status_code, 200)
            self.assertIn("attachment", source.headers["content-disposition"])
            self.assertEqual(
                client.get("/bids/UploadQA/documents/secret.env", headers=headers).status_code, 404
            )
            self.assertTrue(client.get("/jobs", headers=headers).json()["jobs"])

    def test_api_auth_limits_jobs_and_metrics(self):
        app = create_app(
            self.settings, embeddings=self.embeddings, reranker=self.reranker, agent_factory=TestAgent
        )
        with TestClient(app) as client:
            self.assertEqual(client.get("/health/ready").status_code, 200)
            self.assertEqual(client.get("/status").status_code, 401)
            headers = {"X-API-Key": "test-key"}
            self.assertEqual(client.post("/index", headers=headers, json={"folder": "../"}).status_code, 400)
            response = client.post("/index", headers=headers, json={"folder": "NewBid"})
            self.assertEqual(response.status_code, 202)
            identifier = response.json()["job_id"]
            for _ in range(100):
                record = client.get(f"/jobs/{identifier}", headers=headers).json()
                if record["status"] != "running":
                    break
                import time

                time.sleep(0.02)
            self.assertEqual(record["status"], "completed")
            self.assertEqual(
                client.get("/search", headers=headers, params={"q": "SKU", "bid_id": "NewBid"}).status_code,
                200,
            )
            self.assertIn("rfp_http_requests_total", client.get("/metrics", headers=headers).text)
            self.assertEqual(client.post("/ask", headers=headers, content=b"x" * 40000).status_code, 413)
            self.assertEqual(
                client.post("/ask", headers=headers, json={"question": "x" * 2001}).status_code, 422
            )
            self.assertEqual(client.get("/jobs/nonexistent", headers=headers).status_code, 404)

    def test_job_restart_recovery_and_capacity(self):
        async def check():
            manager = JobManager(self.settings)
            gate = asyncio.Event()

            async def blocked(_):
                await gate.wait()
                return {"ok": True}

            ids = [manager.submit("ask", blocked) for _ in range(self.settings.max_active_jobs)]
            with self.assertRaises(OverflowError):
                manager.submit("ask", blocked)
            gate.set()
            await manager.close()
            self.assertTrue(all(manager.get(i)["status"] == "completed" for i in ids))
            self.assertEqual(manager.recent()[0]["result"], {"ok": True})
            with self.assertRaises(ValueError):
                manager.recent(limit=101)

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
