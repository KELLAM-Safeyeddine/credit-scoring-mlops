"""Verify installed libraries without running a training pipeline."""

import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main() -> None:
    """Check the local environment and write reproducible verification evidence."""
    root = Path(__file__).resolve().parents[1]
    if Path(sys.prefix).resolve() != (root / "creditS").resolve():
        raise SystemExit("Use creditS.")
    modules = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
        "lightgbm": "lightgbm",
        "xgboost": "xgboost",
        "shap": "shap",
        "fairlearn": "fairlearn",
        "evidently": "evidently",
        "pandera": "pandera.pandas",
        "dvc": "dvc",
        "mlflow": "mlflow",
        "fastapi": "fastapi",
    }
    versions = {}
    for package, module in modules.items():
        importlib.import_module(module)
        versions[package] = importlib.metadata.version(package)
        print(f"Import OK: {package}=={versions[package]}", flush=True)
    evidence = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "venv_name": Path(sys.prefix).name,
        "versions": versions,
        "airflow_installed": any(
            dist.metadata["Name"].lower() == "apache-airflow"
            for dist in importlib.metadata.distributions()
        ),
    }
    assert not evidence["airflow_installed"]
    (root / "docs" / "phase1-environment.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
