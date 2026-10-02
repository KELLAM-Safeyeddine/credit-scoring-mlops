"""Verify the phase-one environment and project scaffolding."""

import importlib.metadata
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_environment() -> None:
    expected = ROOT / (".ci-venv" if os.getenv("GITHUB_ACTIONS") == "true" else "creditS")
    assert Path(sys.prefix).resolve() == expected.resolve()
    assert sys.version_info[:2] in {(3, 10), (3, 11)}
    installed = {dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()}
    assert "apache-airflow" not in installed


def test_venv_and_archive_are_ignored() -> None:
    for name in (".gitignore", ".dockerignore"):
        content = (ROOT / name).read_text(encoding="utf-8-sig")
        assert "creditS/" in content
        assert "*.zip" in content


def test_notebook_skeletons_are_valid() -> None:
    notebooks = list((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 3
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
