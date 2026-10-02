"""Start MLflow only for the disposable HTTP packaging verification."""

import os
import sys
from pathlib import Path


def main() -> None:
    """Use the disposable volume, with artifacts served over the test network."""
    if Path(sys.prefix).resolve() != Path("/opt/creditS").resolve():
        raise SystemExit("Run only inside the disposable API-image validation container.")
    root = Path("/app/runtime/api/checks")
    root.mkdir(parents=True, exist_ok=True)
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--host",
            "0.0.0.0",
            "--port",
            "5000",
            "--workers",
            "1",
            "--backend-store-uri",
            "sqlite:////app/runtime/api/checks/tracking.db",
            "--artifacts-destination",
            "file:///app/runtime/api/checks/artifacts",
            "--serve-artifacts",
            "--allowed-hosts",
            "registry,registry:5000,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000",
        ],
    )


if __name__ == "__main__":
    main()
