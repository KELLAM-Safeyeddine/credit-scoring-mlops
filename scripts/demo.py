"""Run the complete synthetic lifecycle through Airflow and a disposable Docker API."""

import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
from http.client import HTTPException
from pathlib import Path
from uuid import uuid4

from scripts.demo_support import IMAGE, api, docker, request, wait

ROOT = Path(__file__).resolve().parents[1]


def home_credit_aliases(client) -> dict | None:
    """Keep an absent Home Credit Registry valid for a fresh-checkout synthetic demo."""
    from mlflow.exceptions import MlflowException

    try:
        return dict(client.get_registered_model("credit-scoring").aliases)
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return None


def main() -> None:
    """Preserve Home Credit aliases and clean up only the API created by this invocation."""
    if Path(sys.prefix).resolve() != (ROOT / "creditS").resolve():
        raise SystemExit("Use the root creditS interpreter.")
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri="http://127.0.0.1:5001")
    before = home_credit_aliases(client)
    uid = uuid4().hex[:12]
    directory = ROOT / "runtime/demo" / uid
    directory.mkdir(parents=True)
    name = "credit-demo-" + uid
    api_name = "credit-demo-api-" + uid
    token = secrets.token_urlsafe(32)
    network = "credit-scoring-mlops_default"
    write = {
        "mode": "synthetic",
        "name": name,
        "api_url": "http://" + api_name + ":8000",
        "reload_token": token,
    }
    (directory / "context.json").write_text(json.dumps(write), encoding="utf-8")
    from src.demo.lifecycle import bootstrap

    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5001"
    bootstrap(directory)
    try:
        docker(
            "run",
            "-d",
            "--name",
            api_name,
            "--network",
            network,
            "--init",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--tmpfs",
            "/tmp:rw,size=268435456,mode=1777",
            "--mount",
            f"type=bind,source={directory},target=/app/runtime/api",
            "-p",
            "127.0.0.1::8000",
            "-e",
            "MLFLOW_TRACKING_URI=http://mlflow:5000",
            "-e",
            "MLFLOW_MODEL_NAME=" + name,
            "-e",
            "API_RELOAD_TOKEN=" + token,
            "-e",
            "API_LOG_HMAC_KEY=" + secrets.token_urlsafe(32),
            IMAGE,
        )
        port = docker("port", api_name, "8000/tcp").rsplit(":", 1)[1]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                if request("http://127.0.0.1:" + port + "/live")[0] == 200:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, HTTPException):
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Demo API did not start.")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            dags = {item["dag_id"] for item in api("dags")["dags"]}
            if {"credit_demo", "credit_demo_retraining"} <= dags:
                break
            time.sleep(5)
        else:
            raise TimeoutError("Demo DAGs have not been loaded; run make up.")
        for dag in ("credit_demo", "credit_demo_retraining"):
            api(f"dags/{dag}", {"is_paused": False}, "PATCH")
        run_id = "demo__" + uid
        api("dags/credit_demo/dagRuns", {"dag_run_id": run_id, "conf": {"demo_id": uid}})
        print(f"Synthetic demo started: {run_id}; reports: {directory}", flush=True)
        parent = wait("credit_demo", run_id, 1800)
        child = wait("credit_demo_retraining", "retraining__" + run_id, 1800)
        gates = [json.loads((directory / f"v{i}/gate.json").read_text()) for i in (1, 2, 3)]
        assert [gate["gate_status"] for gate in gates] == ["accepted", "rejected", "accepted"]
        assert gates[1]["champion_version"] == gates[0]["version"]
        assert (
            request("http://127.0.0.1:" + port + "/model-info")[1]["version"] == gates[2]["version"]
        )
        aliases = client.get_registered_model(name).aliases
        assert aliases == {"champion": gates[2]["version"], "previous": gates[0]["version"]}
        assert home_credit_aliases(client) == before
        proof = {
            "mode": "synthetic-not-home-credit",
            "demo_id": uid,
            "parent_dag": parent,
            "retraining_dag": child,
            "gates": [
                {key: gate[key] for key in ("version", "gate_status", "failed_rules")}
                for gate in gates
            ],
            "aliases": aliases,
            "home_credit_aliases_unchanged": True,
            "api_reloaded_to_v3": True,
        }
        (directory / "validation.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
        print(json.dumps(proof, indent=2), flush=True)
    except Exception:
        logs = subprocess.run(["docker", "logs", api_name], capture_output=True, text=True)
        (directory / "api-error.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
        raise
    finally:
        subprocess.run(["docker", "rm", "-f", api_name], capture_output=True, check=False)


if __name__ == "__main__":
    main()
