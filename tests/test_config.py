"""Deployment regressions for the two-key environment and Railway-provided settings."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from rfp.core.config import Settings
from rfp.main import create_app
from tests.support import TestAgent, TestEmbeddings, TestReranker


class ConfigurationTests(unittest.TestCase):
    def test_two_key_env_boots_authenticated_production_api(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            key = "test-backend-key-with-at-least-24-characters"
            env = root / ".env"
            env.write_text(f"RFP_API_KEY={key}\nGROQ_API_KEY=synthetic-groq-key\n")
            settings = Settings(
                _env_file=env,
                environment="production",
                input_root=root,
                storage_dir=root / "storage",
                log_level="ERROR",
            )
            app = create_app(
                settings, embeddings=TestEmbeddings(), reranker=TestReranker(), agent_factory=TestAgent
            )
            with TestClient(app) as client:
                self.assertEqual(client.get("/health/ready").status_code, 200)
                self.assertEqual(client.get("/status").status_code, 401)
                response = client.get("/status", headers={"X-API-Key": key})
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()["llm_configured"])

    def test_railway_domain_is_automatic_and_custom_url_wins(self):
        with patch.dict(os.environ, {"RAILWAY_PUBLIC_DOMAIN": "bidwell.up.railway.app"}, clear=True):
            settings = Settings(_env_file=None)
            self.assertEqual(settings.public_base_url, "https://bidwell.up.railway.app")
            custom = Settings(_env_file=None, public_base_url="https://rfp.example.com")
            self.assertEqual(custom.public_base_url, "https://rfp.example.com")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Settings(_env_file=None).public_base_url, "http://localhost:8000")

    def test_production_rejects_missing_and_short_api_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            for key in ("", "short"):
                with self.assertRaises(ValidationError):
                    Settings(_env_file=None, environment="production", api_key=key)
