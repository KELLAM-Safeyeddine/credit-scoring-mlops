"""Docker and Airflow clients used by the application lifecycle demonstration."""

import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "credit-scoring-api:0.1.0"


def settings() -> dict:
    """Read private local credentials without printing them."""
    if Path(sys.prefix).resolve() != (ROOT / "creditS").resolve():
        raise SystemExit("Use the root creditS interpreter.")
    return dict(
        line.split("=", 1)
        for line in (ROOT / ".env").read_text().splitlines()
        if "=" in line and not line.startswith("#")
    )


def api(path: str, body: dict | None = None, method: str | None = None) -> dict:
    """Call the local Airflow API with credentials kept out of logs."""
    env = settings()
    credentials = f"{env['AIRFLOW_ADMIN_USERNAME']}:{env['AIRFLOW_ADMIN_PASSWORD']}"
    request = urllib.request.Request(
        f"http://127.0.0.1:{env['AIRFLOW_PORT']}/api/v1/{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": "Basic " + base64.b64encode(credentials.encode()).decode(),
            "Content-Type": "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def wait(dag: str, run_id: str, timeout: int) -> dict:
    """Print state changes while waiting for an actual scheduler execution."""
    deadline = time.monotonic() + timeout
    last = None
    endpoint = f"dags/{dag}/dagRuns/{quote(run_id, safe='')}"
    while time.monotonic() < deadline:
        run = api(endpoint)
        tasks = api(endpoint + "/taskInstances")["task_instances"]
        states = {task["task_id"]: task["state"] for task in tasks}
        if states != last:
            print(
                json.dumps({"dag": dag, "run_id": run_id, "state": run["state"], "tasks": states}),
                flush=True,
            )
            last = states
        if run["state"] in {"success", "failed"}:
            if run["state"] != "success":
                raise RuntimeError(f"DAG failed: {dag} / {run_id}")
            return {"run_id": run_id, "state": run["state"], "tasks": states}
        time.sleep(5)
    raise TimeoutError(f"DAG not finished within {timeout}s: {run_id}")


def docker(*args: str) -> str:
    """Run Docker commands without shell interpolation or secret output."""
    return subprocess.check_output(["docker", *args], cwd=ROOT, text=True).strip()


def request(url: str, payload: dict | None = None) -> tuple[int, dict]:
    """Read JSON even when readiness returns an expected 503."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)
