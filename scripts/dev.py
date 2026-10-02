"""Run developer commands with the project's virtual environment."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Dispatch implemented tasks; fail clearly for future phases."""
    root = Path(__file__).resolve().parents[1]
    if Path(sys.prefix).resolve() != (root / "creditS").resolve():
        raise SystemExit("Activate creditS or call its Python executable directly.")
    task, *files = sys.argv[1:]
    commands = {
        "test": [sys.executable, "-m", "pytest", *files],
        "lint": [sys.executable, "-m", "ruff", "check", *(files or ["."])],
        "format-check": [sys.executable, "-m", "ruff", "format", "--check", *(files or ["."])],
    }
    if task not in commands:
        raise SystemExit(f"Task '{task}' will be implemented in its project phase.")
    subprocess.run(commands[task], cwd=root, check=True)


if __name__ == "__main__":
    main()
