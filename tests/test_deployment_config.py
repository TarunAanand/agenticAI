import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class DeploymentConfigTests(unittest.TestCase):
    def test_compose_binds_only_to_loopback_by_default(self):
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
        self.assertIn('"127.0.0.1:8501:8501"', compose)
        self.assertIn("studymate-data:/app/data", compose)

    def test_container_copies_streamlit_settings_and_runs_non_root(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY --chown=studymate:studymate .streamlit ./.streamlit", dockerfile)
        self.assertIn("USER studymate", dockerfile)
        self.assertIn("/_stcore/health", dockerfile)

    def test_docker_context_excludes_local_secrets_and_data(self):
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        self.assertIn(".env", dockerignore)
        self.assertIn("data", dockerignore)
        self.assertIn("!.env.example", dockerignore)


if __name__ == "__main__":
    unittest.main()
