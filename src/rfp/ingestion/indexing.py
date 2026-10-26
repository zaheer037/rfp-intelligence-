"""Whole ingestion pipeline and atomic per-bid FAISS/SQLite snapshots."""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import uuid
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import pymupdf
from bs4 import BeautifulSoup
from PIL import Image

from rfp.core.config import Settings
from rfp.core.observability import event, stage
from rfp.schemas import Chunk
from rfp.utils.identifiers import validate_bid_id

SUPPORTED = {".pdf", ".html", ".htm", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp"}


class Indexing:
    """Discover, parse, clean, chunk, embed and persist all files belonging to one bid."""

    def __init__(self, settings: Settings, embeddings):
        self.settings = settings
        self.embeddings = embeddings
        settings.storage_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        settings.storage_dir.chmod(0o700)
        self.root = settings.storage_dir / "indexes"
        self.root.mkdir(parents=True, exist_ok=True)
        # ponytail: one writer per process; deploy one worker/replica with this local snapshot store.
        self._write_lock = threading.RLock()

    def resolve_folder(self, folder: str) -> Path:
        """Limit ingestion to folders under the configured trusted input root."""
        path = (self.settings.input_root / folder).resolve()
        upload_root = (self.settings.storage_dir / "uploads").resolve()
        if (
            not (path.is_relative_to(self.settings.input_root) or path.is_relative_to(upload_root))
            or not path.is_dir()
        ):
            raise ValueError("Folder must exist under RFP_INPUT_ROOT")
        if path.is_relative_to(self.settings.storage_dir) and not path.is_relative_to(upload_root):
            raise ValueError("Cannot index internal storage")
        return path

    def load(self, bid_id: str):
        """Load one committed snapshot into memory, rejecting incompatible embeddings."""
        with self._write_lock:
            return self._load_snapshot(bid_id)

    def _load_snapshot(self, bid_id: str):
        """Read a generation while holding the writer lock so cleanup cannot race it."""
        validate_bid_id(bid_id)
        base = self.root / bid_id
        pointer = base / "current.json"
        if not pointer.exists():
            return None
        generation = json.loads(pointer.read_text())["generation"]
        if not re.fullmatch(r"[a-f0-9]{32}", generation):
            raise ValueError("Invalid snapshot pointer")
        path = base / generation
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("fingerprint") != self._fingerprint():
            raise ValueError(f"Embedding/chunk configuration changed for {bid_id}; reindex it")
        with sqlite3.connect(f"file:{path / 'chunks.sqlite'}?mode=ro", uri=True) as db:
            chunks = [
                Chunk.model_validate_json(row[0])
                for row in db.execute("SELECT payload FROM chunks ORDER BY position")
            ]
        index = faiss.read_index(str(path / "vectors.faiss"))
        if index.ntotal != len(chunks):
            raise ValueError("Index and chunk manifest disagree")
        return chunks, index, manifest

    def _fingerprint(self) -> str:
        """Invalidate snapshots when model, chunking or parser contracts change."""
        return f"{self.embeddings.fingerprint}:{self.settings.chunk_words}:{self.settings.chunk_overlap}:parser-v3"

    def index_folder(self, folder: str, bid_id: str | None = None) -> dict:
        """Index only changed files, reuse prior vectors, and atomically publish a snapshot."""
        path = self.resolve_folder(folder)
        bid_id = validate_bid_id(bid_id or path.name)
        with self._write_lock, stage("indexing"):
            files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
            if not files or len(files) > self.settings.max_files:
                raise ValueError("No supported files or too many files in bid folder")
            for p in files:
                if not p.resolve().is_relative_to(path):
                    raise ValueError("File symlinks must stay within the bid folder")
                if p.stat().st_size > self.settings.max_file_mb * 1024 * 1024:
                    raise ValueError(f"File exceeds size limit: {p.name}")
            try:
                old = self.load(bid_id)
            except (ValueError, OSError, RuntimeError, sqlite3.DatabaseError) as exc:
                event("snapshot_rebuild_required", bid_id=bid_id, error_type=type(exc).__name__)
                old = None
            previous = old[2]["files"] if old else {}
            hashes = {p.name: self._file_hash(p) for p in files}
            if old and hashes == previous and not old[2].get("warnings"):
                return {"bid_id": bid_id, "status": "unchanged", "chunks": len(old[0]), "warnings": []}
            chunks, warnings, successful = [], [], {}
            for file in files:
                if (
                    old
                    and previous.get(file.name) == hashes[file.name]
                    and not any(w.startswith(file.name + ":") for w in old[2].get("warnings", []))
                ):
                    chunks.extend(c for c in old[0] if c.file_name == file.name)
                    successful[file.name] = hashes[file.name]
                    continue
                try:
                    pages = self.parse_file(file)
                    parsed = self.chunk_pages(pages, file, bid_id)
                    if not parsed:
                        raise ValueError("No readable content")
                    chunks.extend(parsed)
                    successful[file.name] = hashes[file.name]
                    warnings.extend(f"{file.name}: {w}" for p in pages for w in p.get("warnings", []))
                except Exception as exc:
                    warnings.append(f"{file.name}: {type(exc).__name__}: {str(exc)[:160]}")
                    event("file_parse_failed", file=file.name, error_type=type(exc).__name__)
                    # Never delete previously indexed content just because its replacement is unreadable.
                    if old:
                        chunks.extend(c for c in old[0] if c.file_name == file.name)
                        if file.name in previous:
                            successful[file.name] = previous[file.name]
            if not chunks:
                raise ValueError("No readable documents; " + "; ".join(warnings))
            cache = {c.content_hash: old[1].reconstruct(i) for i, c in enumerate(old[0])} if old else {}
            missing = {c.content_hash: c.text for c in chunks if c.content_hash not in cache}
            if missing:
                with stage("index_embedding"):
                    encoded = self.embeddings.encode(list(missing.values()))
                cache.update(zip(missing, encoded, strict=True))
            vectors = np.ascontiguousarray([cache[c.content_hash] for c in chunks], dtype="float32")
            faiss.normalize_L2(vectors)
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            manifest = {
                "bid_id": bid_id,
                "fingerprint": self._fingerprint(),
                "files": successful,
                "warnings": warnings,
                "chunks": len(chunks),
            }
            self._publish(bid_id, chunks, index, manifest)
            event(
                "bid_indexed",
                bid_id=bid_id,
                chunks=len(chunks),
                embedded_chunks=len(missing),
                warnings=len(warnings),
            )
            return {
                "bid_id": bid_id,
                "status": "partial" if warnings else "indexed",
                "chunks": len(chunks),
                "warnings": warnings,
            }

    @staticmethod
    def _file_hash(path: Path) -> str:
        """Hash files without loading their complete bytes into memory."""
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    def _publish(self, bid_id, chunks, index, manifest) -> None:
        """Commit a complete generation with a single atomic pointer replacement."""
        base = self.root / bid_id
        base.mkdir(exist_ok=True)
        generation = uuid.uuid4().hex
        target = base / generation
        target.mkdir()
        try:
            with sqlite3.connect(target / "chunks.sqlite") as db:
                db.execute("CREATE TABLE chunks (position INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
                db.executemany(
                    "INSERT INTO chunks VALUES (?, ?)",
                    [(i, c.model_dump_json()) for i, c in enumerate(chunks)],
                )
            faiss.write_index(index, str(target / "vectors.faiss"))
            (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            pointer = base / "next.json"
            pointer.write_text(json.dumps({"generation": generation}))
            # fsync payloads before making the generation visible to future readers.
            for file in target.iterdir():
                with file.open("rb") as handle:
                    os.fsync(handle.fileno())
            with pointer.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(pointer, base / "current.json")
            fd = os.open(base, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            # Readers hold in-memory copies; old on-disk generations can be removed.
            for prior in base.iterdir():
                if prior.is_dir() and prior.name != generation:
                    shutil.rmtree(prior)
        except Exception:
            if not (base / "current.json").exists():
                shutil.rmtree(target, ignore_errors=True)
            raise

    def parse_file(self, path: Path) -> list[dict]:
        """Dispatch HTML, PDF and image parsing into a common page representation."""
        with stage("parsing"):
            if path.suffix.lower() in {".html", ".htm"}:
                soup = BeautifulSoup(path.read_text(errors="replace"), "html.parser")
                for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                    element.decompose()
                main = soup.find("main") or soup.find(id="main-content") or soup.body or soup
                for table in main.find_all("table"):
                    rows = [
                        " | ".join(cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"]))
                        for row in table.find_all("tr")
                    ]
                    table.replace_with(soup.new_string("\n" + "\n".join(rows) + "\n"))
                return [{"text": main.get_text("\n", strip=True), "page": None, "method": "html"}]
            if path.suffix.lower() == ".pdf":
                return self._parse_pdf(path)
            with Image.open(path) as image:
                if image.width * image.height > self.settings.max_image_pixels:
                    raise ValueError("Image pixel limit exceeded")
                pages = []
                for i in range(getattr(image, "n_frames", 1)):
                    if i >= self.settings.max_pdf_pages:
                        raise ValueError("Image frame limit exceeded")
                    image.seek(i)
                    pages.append({"text": self._ocr(image), "page": i + 1, "method": "ocr"})
                return pages

    def _parse_pdf(self, path: Path) -> list[dict]:
        """Keep physical page numbers; extract tables and OCR substantial image regions."""
        import pdfplumber

        pages = []
        with pymupdf.open(path) as doc, pdfplumber.open(path) as tables_doc:
            if len(doc) > self.settings.max_pdf_pages:
                raise ValueError("PDF page limit exceeded")
            for number, page in enumerate(doc):
                native = page.get_text("text", sort=True)
                record = {"text": native, "page": number + 1, "method": "native", "warnings": []}
                try:
                    tables = tables_doc.pages[number].extract_tables()
                    for table in tables:
                        cells = [str(cell or "").strip() for row in table for cell in row]
                        # Form/layout grids are not data tables; native text retains their content.
                        if (
                            not cells
                            or max(len(row) for row in table) > 12
                            or sum(bool(cell) for cell in cells) / len(cells) < 0.4
                        ):
                            continue
                        rows = [
                            " | ".join(str(cell or "").replace("\n", " ") for cell in row) for row in table
                        ]
                        record["text"] += "\nTABLE\n" + "\n".join(rows)
                except Exception as exc:
                    record["warnings"].append(f"page {number + 1}: table extraction {type(exc).__name__}")
                finally:
                    tables_doc.pages[number].close()
                try:
                    if len(native.strip()) < 40:
                        record["text"] = self._ocr_page(page)
                        record["method"] = "ocr"
                    elif self.settings.ocr_enabled:
                        for info in page.get_image_info():
                            rect = pymupdf.Rect(info["bbox"]) & page.rect
                            if rect.get_area() > page.rect.get_area() * 0.15:
                                ocr_text = self._ocr_page(page, rect)
                                if ocr_text.strip() and self.clean_text(ocr_text) not in self.clean_text(
                                    native
                                ):
                                    record["text"] += "\nIMAGE CONTENT\n" + ocr_text
                                    record["method"] = "native+ocr"
                except Exception as exc:
                    record["warnings"].append(f"page {number + 1}: OCR {type(exc).__name__}: {str(exc)[:80]}")
                if not record["text"].strip():
                    record["warnings"].append(f"page {number + 1}: empty/unreadable page")
                pages.append(record)
        # Only remove short recurring margins, never repeated headings or table rows in the body.
        edges = Counter()
        for p in pages:
            lines = p["text"].splitlines()
            edges.update(set(x.strip() for x in lines[:2] + lines[-2:] if 0 < len(x.strip()) < 150))
        repeated = {x for x, count in edges.items() if len(pages) >= 3 and count >= max(3, len(pages) * 0.6)}
        for p in pages:
            lines = p["text"].splitlines()
            p["text"] = "\n".join(
                x
                for i, x in enumerate(lines)
                if not ((i < 2 or i >= len(lines) - 2) and x.strip() in repeated)
            )
        return pages

    def _ocr_page(self, page, clip=None) -> str:
        """Rasterize a bounded page region before handing it to local Tesseract."""
        rect = clip or page.rect
        scale = min(2.0, (self.settings.max_image_pixels / max(1, rect.get_area())) ** 0.5)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False)
        return self._ocr(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))

    def _ocr(self, image: Image.Image) -> str:
        """Run OCR with a hard subprocess timeout; make missing dependencies explicit."""
        if not self.settings.ocr_enabled:
            raise RuntimeError("OCR disabled for image-only content")
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "page.png"
            image.convert("RGB").save(source)
            result = subprocess.run(
                ["tesseract", str(source), "stdout", "--psm", "6"],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.settings.ocr_timeout_seconds,
            )
            return result.stdout

    @staticmethod
    def clean_text(text: str) -> str:
        """Repair line-break hyphenation while preserving table and paragraph boundaries."""
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text.replace("\x00", ""))
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    @staticmethod
    def _classify(path: Path, text: str):
        """Prefer explicit addendum titles, then document-purpose clues, without bid-specific rules."""
        heading = path.stem + "\n" + text[:1500]
        addendum = re.search(r"addendum\s*(?:no\.?\s*)?(\d+)", heading, re.I)
        if addendum:
            return "addendum", int(addendum[1])
        if path.suffix.lower() in {".html", ".htm"}:
            return "bid_page", None
        titles = [line.strip() for line in text.splitlines() if line.strip()][:4]
        if "affidavit" in path.stem.lower() or any(
            len(line) < 100 and re.fullmatch(r"[A-Za-z /-]*affidavit", line, re.I) for line in titles
        ):
            return "affidavit", None
        if re.search(r"specs|specification|\bSKU\b", path.stem + "\n" + text[:200], re.I):
            return "specs", None
        return "rfp", None

    def chunk_pages(self, pages: list[dict], path: Path, bid_id: str) -> list[Chunk]:
        """Create page/section-aware chunks with overlap and repeated table headers."""
        text = "\n".join(p["text"] for p in pages)
        doc_type, addendum = self._classify(path, text)
        date_match = re.search(r"(?:issue date|document date|dated)\s*[:\-]?\s*([^\n]{6,45})", text, re.I)
        document_date = date_match[1].strip() if date_match else None
        result = []
        for page in pages:
            cleaned = self.clean_text(page["text"])
            if not cleaned:
                continue
            printed = re.search(r"Page\s+\d+\s+(?:of|\|)\s*\d+", cleaned, re.I)
            section, buffer, table_header = "html:main" if page["page"] is None else "", [], ""

            def flush():
                if not buffer:
                    return
                body = "\n".join(buffer)
                digest = hashlib.sha256(body.encode()).hexdigest()
                identity = f"{bid_id}:{path.name}:{page['page']}:{len(result)}:{digest}"
                result.append(
                    Chunk(
                        chunk_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
                        bid_id=bid_id,
                        file_name=path.name,
                        doc_type=doc_type,
                        addendum_number=addendum,
                        page_number=page["page"],
                        section=section,
                        printed_page=printed[0] if printed else None,
                        document_date=document_date,
                        method=page["method"],
                        text=body,
                        content_hash=digest,
                    )
                )

            for line in cleaned.splitlines():
                is_heading = (
                    bool(line.strip())
                    and len(line) < 100
                    and (line.isupper() or bool(re.match(r"^(?:Section\s+\d|\d+(?:\.\d+)*\s+[A-Z])", line)))
                )
                if is_heading and buffer and sum(len(s.split()) for s in buffer) > 60:
                    flush()
                    buffer = []
                if is_heading:
                    section = line[:150]
                    table_header = ""
                if "|" in line and not table_header:
                    table_header = line
                # Oversized rows/prose are split without dropping the remainder.
                words = line.split()
                pieces = [
                    " ".join(words[i : i + self.settings.chunk_words])
                    for i in range(0, len(words), self.settings.chunk_words)
                ] or [""]
                for piece in pieces:
                    if (
                        buffer
                        and sum(len(s.split()) for s in buffer) + len(piece.split())
                        > self.settings.chunk_words
                    ):
                        flush()
                        overlap = (
                            " ".join(" ".join(buffer).split()[-self.settings.chunk_overlap :])
                            if self.settings.chunk_overlap
                            else ""
                        )
                        buffer = [s for s in (table_header if "|" in piece else "", overlap) if s]
                    buffer.append(piece)
            flush()
        return result
