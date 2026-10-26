"""Read-only workspace views and bounded uploads for the separately deployed React app."""

import base64
import binascii
import json
import re
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import Field

from rfp.schemas import Contract
from rfp.utils.identifiers import validate_bid_id


class UploadedFile(Contract):
    name: str = Field(min_length=1, max_length=180)
    data: str = Field(min_length=1, max_length=2_800_000)


class Upload(Contract):
    bid_id: str = Field(min_length=1, max_length=80)
    files: list[UploadedFile] = Field(min_length=1, max_length=20)


def workspace_router(settings, indexing, retrieval, service, jobs):
    """Expose only committed metadata and validated source paths, never arbitrary filesystem reads."""
    router = APIRouter()

    def require_bid(identifier):
        validate_bid_id(identifier)
        if identifier not in retrieval.bid_ids():
            raise HTTPException(404, "Bid is not indexed")

    def latest_record(identifier):
        for row in jobs.recent():
            if row["kind"] == "extract" and row["result"]:
                result = row["result"]
                if result.get("bid_id") == identifier:
                    return result
        return None

    @router.get("/bids")
    async def bids():
        result = []
        for identifier in retrieval.bid_ids():
            chunks = list(retrieval.chunks([identifier]).values())
            files = sorted({c.file_name for c in chunks})
            record = latest_record(identifier)
            title = record.get("fields", {}).get("Title", {}).get("value") if record else None
            result.append(
                {
                    "bid_id": identifier,
                    "title": title or identifier,
                    "files": len(files),
                    "chunks": len(chunks),
                    "has_record": record is not None,
                    "addendums": sorted({c.addendum_number for c in chunks if c.addendum_number is not None}),
                }
            )
        return {"bids": result}

    @router.get("/bids/{bid_id}")
    async def bid(bid_id: str):
        require_bid(bid_id)
        grouped = {}
        for c in retrieval.chunks([bid_id]).values():
            entry = grouped.setdefault(
                c.file_name,
                {
                    "name": c.file_name,
                    "doc_type": c.doc_type,
                    "pages": 0,
                    "chunks": 0,
                    "methods": set(),
                    "addendum_number": c.addendum_number,
                },
            )
            entry["chunks"] += 1
            entry["pages"] = max(entry["pages"], c.page_number or 0)
            entry["methods"].add(c.method)
        for entry in grouped.values():
            entry["methods"] = sorted(entry["methods"])
        return {"bid_id": bid_id, "documents": list(grouped.values()), "record": latest_record(bid_id)}

    @router.get("/bids/{bid_id}/record")
    async def record(bid_id: str):
        require_bid(bid_id)
        return {"record": latest_record(bid_id)}

    @router.get("/bids/{bid_id}/documents/{file_name:path}")
    async def document(bid_id: str, file_name: str):
        require_bid(bid_id)
        allowed = {c.file_name for c in retrieval.chunks([bid_id]).values()}
        if file_name not in allowed or "/" in file_name or "\\" in file_name:
            raise HTTPException(404, "Document not found")
        upload_folder = settings.storage_dir / "uploads" / bid_id
        folder = indexing.resolve_folder(str(upload_folder) if upload_folder.exists() else bid_id)
        source = (folder / file_name).resolve()
        if not source.is_relative_to(folder) or not source.is_file():
            raise HTTPException(404, "Document not found")
        pdf = source.suffix.lower() == ".pdf"
        return FileResponse(
            source,
            media_type="application/pdf" if pdf else "application/octet-stream",
            filename=file_name,
            content_disposition_type="inline" if pdf else "attachment",
        )

    @router.get("/jobs")
    async def history():
        result = []
        for row in jobs.recent():
            payload = row.pop("result")
            row.pop("result", None)
            row["bid_id"] = payload.get("bid_id") if payload else None
            result.append(row)
        return {"jobs": result}

    @router.get("/jobs/{job_id}/trace")
    async def trace(job_id: str):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id) or not jobs.get(job_id):
            raise HTTPException(404, "Run not found")
        path = settings.storage_dir / "traces" / f"{job_id}.jsonl"
        allowed = {
            "time",
            "run_id",
            "step",
            "event",
            "latency_seconds",
            "tokens",
            "error_type",
            "attempt",
            "hits",
            "status",
        }
        events = []
        if path.exists():
            with path.open() as handle:
                for number, line in enumerate(handle):
                    if number >= 2000:
                        break
                    item = json.loads(line)
                    events.append({k: v for k, v in item.items() if k in allowed})
        return {"events": events}

    @router.get("/evaluation")
    async def evaluation():
        path = Path("artifacts/retrieval-evaluation.json")
        if not path.exists():
            raise HTTPException(404, "Evaluation report is not available")
        return json.loads(path.read_text())

    @router.post("/upload", status_code=202)
    async def upload(body: Upload):
        identifier = validate_bid_id(body.bid_id)
        if identifier in retrieval.bid_ids() or (settings.input_root / identifier).exists():
            raise HTTPException(409, "That bid identifier already exists. Use a new identifier.")
        if len(jobs.tasks) >= settings.max_active_jobs:
            raise HTTPException(429, "Job capacity reached; retry later", headers={"Retry-After": "5"})
        root = settings.storage_dir / "uploads"
        root.mkdir(parents=True, exist_ok=True)
        target = root / identifier
        if target.exists():
            raise HTTPException(409, "That upload identifier already exists")
        total, names = 0, set()
        with tempfile.TemporaryDirectory(dir=root, prefix="incoming-") as temporary:
            for file in body.files:
                if (
                    "/" in file.name
                    or "\\" in file.name
                    or "\x00" in file.name
                    or file.name in names
                    or file.name.startswith(".")
                    or Path(file.name).suffix.lower()
                    not in {".pdf", ".html", ".htm", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
                ):
                    raise HTTPException(400, "Use unique PDF, HTML or image filenames without paths")
                names.add(file.name)
                try:
                    data = base64.b64decode(file.data, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise HTTPException(400, "Invalid file encoding") from exc
                total += len(data)
                if total > 2_000_000:
                    raise HTTPException(413, "The shared demo accepts up to 2 MB per bid")
                (Path(temporary) / file.name).write_bytes(data)
            shutil.copytree(temporary, target)
        try:
            identifier_job = jobs.submit("index", lambda _: service.ingest(str(target), identifier))
        except OverflowError as exc:
            shutil.rmtree(target)
            raise HTTPException(429, "Job capacity reached; retry later") from exc
        return {"job_id": identifier_job, "status_url": f"/jobs/{identifier_job}"}

    return router
