"""Prepare private Docker state and start the project's Compose services."""

import base64
import secrets
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare() -> None:
    """Create secrets once and snapshot local MLflow without sharing its database."""
    if Path(sys.prefix).resolve() != (ROOT / "creditS").resolve():
        raise SystemExit("Use the root creditS interpreter.")
    archives = list(ROOT.glob("*.zip"))
    if len(archives) != 1:
        raise ValueError("Exactly one dataset ZIP is required at the project root.")
    path = ROOT / ".env"
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    values = dict(
        line.split("=", 1)
        for line in content.splitlines()
        if "=" in line and not line.startswith("#")
    )
    defaults = {
        "POSTGRES_USER": "credit",
        "POSTGRES_DB": "airflow",
        "POSTGRES_PASSWORD": secrets.token_hex(24),
        "AIRFLOW_SECRET_KEY": secrets.token_hex(32),
        "AIRFLOW_FERNET_KEY": base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
        "AIRFLOW_ADMIN_USERNAME": "credit_admin",
        "AIRFLOW_ADMIN_PASSWORD": secrets.token_hex(16),
        "AIRFLOW_PORT": "8081",
        "MLFLOW_DOCKER_PORT": "5001",
        "DATA_ARCHIVE": archives[0].name,
        "API_PORT": "8000",
        "API_LOG_HMAC_KEY": secrets.token_hex(32),
        "API_RELOAD_TOKEN": secrets.token_hex(32),
    }
    for key, value in defaults.items():
        if not values.get(key):
            # Replace empty example entries rather than adding duplicate settings.
            content = "\n".join(
                line for line in content.splitlines() if not line.startswith(key + "=")
            )
            content += f"\n{key}={value}\n"
    path.write_text(content.lstrip("\n"), encoding="utf-8")
    for directory in ("airflow", "mlflow", "mlartifacts"):
        (ROOT / "runtime" / directory).mkdir(parents=True, exist_ok=True)
    destination = ROOT / "runtime/mlflow/tracking.db"
    if not destination.exists():
        source = ROOT / "mlruns/tracking.db"
        if source.exists():
            with sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True) as src:
                with sqlite3.connect(destination) as dst:
                    src.backup(dst)
            shutil.copytree(ROOT / "mlartifacts", ROOT / "runtime/mlartifacts", dirs_exist_ok=True)


def main() -> None:
    """Validate Compose then build and launch only this project's services."""
    prepare()
    for command in (
        ["docker", "compose", "config", "--quiet"],
        ["docker", "compose", "build", "airflow-scheduler", "api"],
        ["docker", "compose", "up", "-d"],
    ):
        subprocess.run(command, cwd=ROOT, check=True)
    deadline = time.monotonic() + 180
    expected = {"postgres", "mlflow", "airflow-webserver", "airflow-scheduler", "api"}
    while time.monotonic() < deadline:
        output = subprocess.check_output(
            ["docker", "compose", "ps", "--format", "{{.Service}} {{.Health}}"],
            cwd=ROOT,
            text=True,
        )
        health = dict(
            line.split(maxsplit=1) for line in output.splitlines() if len(line.split()) == 2
        )
        if all(health.get(service) == "healthy" for service in expected):
            print("Airflow, MLflow, PostgreSQL and API are healthy (API process liveness).")
            return
        time.sleep(2)
    raise TimeoutError("Services did not become healthy within 180 seconds; inspect Compose logs.")


if __name__ == "__main__":
    main()
