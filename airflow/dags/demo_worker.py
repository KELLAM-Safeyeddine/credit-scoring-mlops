"""Subprocess boundary for the isolated synthetic lifecycle worker."""

import json
import os
import subprocess
from pathlib import Path


def demo_worker(operation: str, workspace: str) -> dict:
    """Keep ML dependencies out of Airflow and return only small control results."""
    directory = Path(workspace)
    if directory.parent != Path("/workspace/runtime/demo"):
        raise ValueError("Invalid demo workspace.")
    log_path = directory / f"worker_{operation}.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    os.environ["CREDIT_WORKER_PYTHON"],
                    "-m",
                    "src.demo.lifecycle",
                    operation,
                    "--workspace",
                    workspace,
                ],
                cwd="/workspace",
                check=True,
                timeout=900,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print(log_path.read_text(encoding="utf-8", errors="replace")[-10000:])
        raise
    return json.loads((directory / f"task_{operation}.json").read_text())
