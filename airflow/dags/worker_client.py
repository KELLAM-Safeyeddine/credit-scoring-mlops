"""Only standard-library subprocess control runs in Airflow's interpreter."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


def run_worker(
    operation: str,
    workspace: str | None = None,
    run_id: str | None = None,
    conf: dict | None = None,
) -> dict:
    """Invoke the isolated worker without shell interpolation or large XComs."""
    command = [os.environ["CREDIT_WORKER_PYTHON"], "-m", "src.orchestration.worker", operation]
    if workspace:
        command.extend(["--workspace", workspace])
    else:
        command.extend(["--run-id", run_id, "--conf", json.dumps(conf or {})])
        workspace = (
            "/workspace/runtime/airflow/runs/" + hashlib.sha256(run_id.encode()).hexdigest()[:24]
        )
    directory = Path(workspace)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"worker_{operation}.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd="/workspace",
                check=True,
                timeout=5400,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print(log_path.read_text(encoding="utf-8", errors="replace")[-10000:])
        raise
    print(f"Completed {operation}; worker log: {log_path}")
    return json.loads((Path(workspace) / f"task_{operation}.json").read_text())
