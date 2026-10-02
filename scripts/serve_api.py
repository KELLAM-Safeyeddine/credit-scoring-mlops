"""Prepare private API secrets and serve locally using the root creditS Python."""

import os
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare_environment() -> None:
    """Preserve existing environment values and persist generated secrets once."""
    if Path(sys.prefix).resolve() != (ROOT / "creditS").resolve():
        raise SystemExit("Use the root creditS interpreter.")
    path = ROOT / ".env"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    values = dict(
        line.split("=", 1) for line in text.splitlines() if "=" in line and not line.startswith("#")
    )
    for key in ("API_LOG_HMAC_KEY", "API_RELOAD_TOKEN"):
        if not (os.getenv(key) or values.get(key)):
            value = secrets.token_hex(32)
            text = "\n".join(line for line in text.splitlines() if not line.startswith(key + "="))
            text += f"\n{key}={value}\n"
            values[key] = value
    path.write_text(text, encoding="utf-8")
    for key, value in values.items():
        if value:
            os.environ.setdefault(key, value)


def main() -> None:
    """Foreground server; Compose will manage its own API process in phase nine."""
    prepare_environment()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            os.getenv("API_PORT", "8000"),
            "--workers",
            "1",
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
