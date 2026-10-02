"""Run developer commands with the project's virtual environment."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Dispatch tasks using the exact creditS interpreter."""
    root = Path(__file__).resolve().parents[1]
    if Path(sys.prefix).resolve() != (root / "creditS").resolve():
        raise SystemExit("Activate creditS or call its Python executable directly.")
    task, *files = sys.argv[1:]
    commands = {
        "test": [sys.executable, "-m", "pytest", *files],
        "lint": [sys.executable, "-m", "ruff", "check", *(files or ["."])],
        "format-check": [sys.executable, "-m", "ruff", "format", "--check", *(files or ["."])],
        "train": [sys.executable, "-m", "dvc", "repro"],
    }
    if task == "data":
        subprocess.run([sys.executable, "scripts/extract_data.py"], cwd=root, check=True)
        subprocess.run([sys.executable, "-m", "src.data.validate"], cwd=root, check=True)
        return
    if task not in commands:
        raise SystemExit(f"Task '{task}' will be implemented in its project phase.")
    subprocess.run(commands[task], cwd=root, check=True)


if __name__ == "__main__":
    main()
