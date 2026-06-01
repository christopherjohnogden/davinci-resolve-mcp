import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils.env_loader import load_local_env


class EnvLoaderTests(unittest.TestCase):
    def test_loads_env_and_env_local_without_overriding_existing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "README.md").write_text("# test", encoding="utf-8")
            (root / ".env").write_text(
                "RUNPOD_API_KEY=from-env\nRUNPOD_ANALYSIS_ENDPOINT_ID=endpoint-a\n",
                encoding="utf-8",
            )
            (root / ".env.local").write_text(
                "RUNPOD_API_KEY=from-local\nRUNPOD_MEDIA_LOCAL_PREFIX='/Volumes/Media'\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"RUNPOD_API_KEY": "already-set"}, clear=True):
                loaded = load_local_env(start=root / "src")
                self.assertEqual(os.environ["RUNPOD_API_KEY"], "already-set")
                self.assertEqual(os.environ["RUNPOD_ANALYSIS_ENDPOINT_ID"], "endpoint-a")
                self.assertEqual(os.environ["RUNPOD_MEDIA_LOCAL_PREFIX"], "/Volumes/Media")
                self.assertNotIn("RUNPOD_API_KEY", loaded)

    def test_explicit_env_file_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.env"
            path.write_text("export RUNPOD_ANALYSIS_ENDPOINT_ID=custom\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"RESOLVE_MCP_ENV_FILE": str(path)}, clear=True):
                load_local_env(start=Path(tmp))
                self.assertEqual(os.environ["RUNPOD_ANALYSIS_ENDPOINT_ID"], "custom")


if __name__ == "__main__":
    unittest.main()
