"""Install local dependencies strictly inside creditS."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Validate the interpreter, install dependencies and enable Git hooks."""
    root = Path(__file__).resolve().parents[1]
    if Path(sys.prefix).resolve() != (root / "creditS").resolve():
        raise SystemExit("Use the Python executable inside this project's creditS.")
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        raise SystemExit("Python 3.10 or 3.11 is required.")
    if sys.version_info[:2] == (3, 11) and sys.version_info.micro < 9:
        raise SystemExit("Use Python 3.11.9 or newer in the 3.11 series.")
    print(f"Active environment: {sys.prefix}", flush=True)
    lock = root / "requirements-lock-windows.txt"
    lock_args = ["-c", str(lock)] if sys.platform == "win32" and lock.exists() else []
    for args in (
        ["pip", "install", "--upgrade", "pip"],
        ["pip", "install", "--no-compile", "-r", "requirements.txt", *lock_args],
        ["pip", "check"],
        ["dvc", "--version"],
    ):
        subprocess.run([sys.executable, "-m", *args], cwd=root, check=True)
    subprocess.run([sys.executable, "scripts/check_environment.py"], cwd=root, check=True)
    if (root / ".git").exists():
        subprocess.run([sys.executable, "-m", "pre_commit", "install"], cwd=root, check=True)


if __name__ == "__main__":
    main()
