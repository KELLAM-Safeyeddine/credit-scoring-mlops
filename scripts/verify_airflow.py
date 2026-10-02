"""Trigger and inspect actual Airflow DAG runs through its stable REST API."""

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


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


def main() -> None:
    """Validate DAG imports and optionally execute training or monitoring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger", choices=["training", "monitoring"])
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--batch-id", default="01")
    args = parser.parse_args()
    errors = api("importErrors")
    assert errors["total_entries"] == 0, "Airflow DAG import errors."
    dags = {dag["dag_id"] for dag in api("dags")["dags"]}
    assert {"training_pipeline", "monitoring_pipeline"}.issubset(dags)
    print("Two DAGs loaded without import errors.", flush=True)
    if args.trigger:
        dag = args.trigger + "_pipeline"
        run_id = args.run_id or "phase7__" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        conf = (
            {"min_auc": 1.0}
            if args.trigger == "training"
            else {"batch_id": args.batch_id, "max_score_psi": 0.0}
        )
        api(f"dags/{dag}", {"is_paused": False}, "PATCH")
        if args.resume:
            if not args.run_id:
                parser.error("--resume requires --run-id")
            api(f"dags/{dag}/dagRuns/{quote(run_id, safe='')}")
            print(f"Inspecting existing {dag}: {run_id}", flush=True)
        else:
            api(f"dags/{dag}/dagRuns", {"dag_run_id": run_id, "conf": conf})
            print(f"Triggered {dag}: {run_id}", flush=True)
        if args.wait:
            result = wait(dag, run_id, args.timeout)
            if args.trigger == "training":
                assert result["tasks"]["reject_model"] == "success"
                assert result["tasks"]["register_and_promote"] == "skipped"
            else:
                assert result["tasks"]["trigger_training"] == "success"
                result["triggered_training"] = wait(
                    "training_pipeline", "monitoring__" + run_id, args.timeout
                )
            output = ROOT / "runtime/airflow/validation"
            output.mkdir(exist_ok=True)
            (output / f"{args.trigger}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
