"""Start the local MLflow SQLite server using the active creditS Python."""

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Start a hidden persistent server, or reuse the healthy local server."""
    if Path(sys.prefix).resolve() != (ROOT / "creditS").resolve():
        raise SystemExit("Use the creditS interpreter.")
    url = "http://127.0.0.1:5000/health"

    def healthy() -> bool:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.status == 200 and response.read().strip() == b"OK"
        except (urllib.error.URLError, TimeoutError):
            return False

    if healthy():
        print("MLflow already responds at http://localhost:5000")
        return
    state = ROOT / "mlruns"
    state.mkdir(exist_ok=True)
    artifacts = ROOT / "mlartifacts"
    artifacts.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        f"sqlite:///{(state / 'tracking.db').as_posix()}",
        "--artifacts-destination",
        artifacts.as_uri(),
        "--serve-artifacts",
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
        "--workers",
        "1",
    ]
    with (state / "server.log").open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            start_new_session=sys.platform != "win32",
        )
    (state / "server.pid").write_text(str(process.pid), encoding="utf-8")
    for _ in range(45):
        if healthy():
            print(f"MLflow ready: http://localhost:5000 (PID {process.pid})")
            return
        if process.poll() is not None:
            break
        time.sleep(1)
    raise SystemExit("MLflow failed to start; inspect mlruns/server.log.")


if __name__ == "__main__":
    main()
