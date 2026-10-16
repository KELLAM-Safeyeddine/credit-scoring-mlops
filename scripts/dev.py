"""Run application commands with the project's virtual environment."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Dispatch tasks using the exact creditS interpreter."""
    root = Path(__file__).resolve().parents[1]
    if Path(sys.prefix).resolve() != (root / "creditS").resolve():
        raise SystemExit("Activate creditS or call its Python executable directly.")
    task = sys.argv[1]
    commands = {
        "up": [sys.executable, "scripts/start_airflow.py"],
        "demo": [sys.executable, "-m", "scripts.demo"],
        "down": ["docker", "compose", "down"],
        "train": [sys.executable, "-m", "dvc", "repro"],
    }
    if task == "data":
        subprocess.run([sys.executable, "scripts/extract_data.py"], cwd=root, check=True)
        subprocess.run([sys.executable, "-m", "src.data.validate"], cwd=root, check=True)
        return
    if task not in commands:
        raise SystemExit(f"Unknown application command: {task}")
    subprocess.run(commands[task], cwd=root, check=True)


if __name__ == "__main__":
    main()
