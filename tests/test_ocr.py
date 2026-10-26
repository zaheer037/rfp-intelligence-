"""Real OCR integration check; runs in the Docker image that includes Tesseract."""

import io
import shutil
import tempfile
import unittest
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

from rfp.core.config import Settings
from rfp.ingestion.indexing import Indexing
from tests.support import TestEmbeddings


@unittest.skipUnless(shutil.which("tesseract"), "Tesseract is installed in the deployment image")
class OcrIntegrationTest(unittest.TestCase):
    def test_standalone_image_and_scanned_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                _env_file=None,
                environment="development",
                input_root=root,
                storage_dir=root / "store",
                ocr_enabled=True,
            )
            indexing = Indexing(settings, TestEmbeddings())
            image = Image.new("RGB", (1200, 200), "white")
            ImageDraw.Draw(image).text(
                (20, 50), "DELIVERY WITHIN 45 DAYS", fill="black", font=ImageFont.load_default(size=48)
            )
            source = root / "scan.png"
            image.save(source)
            pages = indexing.parse_file(source)
            self.assertIn("45", pages[0]["text"])
            self.assertEqual(pages[0]["method"], "ocr")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            pdf = root / "scanned.pdf"
            with pymupdf.open() as doc:
                page = doc.new_page(width=1200, height=200)
                page.insert_image(page.rect, stream=buffer.getvalue())
                doc.save(pdf)
            pages = indexing.parse_file(pdf)
            self.assertIn("45", pages[0]["text"])
            self.assertEqual(pages[0]["page"], 1)
            self.assertEqual(pages[0]["method"], "ocr")
