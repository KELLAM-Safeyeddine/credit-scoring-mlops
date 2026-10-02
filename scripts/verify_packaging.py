"""Exercise the built API image and Compose stack without promoting Home Credit."""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.client import HTTPException
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "credit-scoring-api:0.1.0"


def docker(*args: str) -> str:
    """Run Docker commands without shell interpolation or secret output."""
    return subprocess.check_output(["docker", *args], cwd=ROOT, text=True).strip()


def request(url: str, payload: dict | None = None) -> tuple[int, dict]:
    """Read JSON even when readiness returns an expected 503."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def main() -> None:
    """Run an isolated synthetic Registry inside a uniquely named disposable volume."""
    if Path(sys.prefix).resolve() != (ROOT / "creditS").resolve():
        raise SystemExit("Use the root creditS interpreter.")
    uid = uuid4().hex[:12]
    volume = "credit-packaging-check-" + uid
    log_volume = "credit-packaging-logs-" + uid
    name = "credit-packaging-api-" + uid
    registry_name = "credit-packaging-registry-" + uid
    network = "credit-packaging-network-" + uid
    docker("volume", "create", volume)
    docker("volume", "create", log_volume)
    docker("network", "create", network)
    mount = f"type=volume,source={volume},target=/app/runtime/api"
    scripts_mount = f"type=bind,source={ROOT / 'scripts'},target=/validation,readonly"
    proof = {}
    try:
        print("Starting an isolated MLflow HTTP server and synthetic Registry.", flush=True)
        docker(
            "run",
            "-d",
            "--name",
            registry_name,
            "--network",
            network,
            "--network-alias",
            "registry",
            "--no-healthcheck",
            "--mount",
            mount,
            "--mount",
            scripts_mount,
            "--entrypoint",
            "/opt/creditS/bin/python",
            IMAGE,
            "/validation/packaging_registry.py",
        )
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            response = subprocess.run(
                [
                    "docker",
                    "exec",
                    registry_name,
                    "/opt/creditS/bin/python",
                    "-c",
                    "import urllib.request; "
                    "urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3)",
                ],
                capture_output=True,
            )
            if response.returncode == 0:
                break
            time.sleep(2)
        else:
            raise TimeoutError("Isolated MLflow HTTP server did not start.")
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                network,
                "--mount",
                mount,
                "--mount",
                scripts_mount,
                "-e",
                "PYTHONPATH=/app",
                "-e",
                "MLFLOW_TRACKING_URI=http://registry:5000",
                "--entrypoint",
                "/opt/creditS/bin/python",
                IMAGE,
                "/validation/packaging_fixture.py",
            ],
            cwd=ROOT,
            check=True,
        )
        docker(
            "run",
            "-d",
            "--name",
            name,
            "--network",
            network,
            "--init",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--tmpfs",
            "/tmp:rw,size=268435456,mode=1777",
            "--mount",
            f"type=volume,source={log_volume},target=/app/runtime/api",
            "-p",
            "127.0.0.1::8000",
            "-e",
            "MLFLOW_TRACKING_URI=http://registry:5000",
            "-e",
            "MLFLOW_MODEL_NAME=packaging-fixture",
            "-e",
            "API_LOG_HMAC_KEY=synthetic-validation-key-never-use-in-production",
            "-e",
            "API_RELOAD_TOKEN=synthetic-validation-reload-token",
            IMAGE,
        )
        port = docker("port", name, "8000/tcp").rsplit(":", 1)[1]
        url = "http://127.0.0.1:" + port
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                status, body = request(url + "/health")
                if status == 200:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, HTTPException):
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Isolated API did not become ready.")
        payload = {
            "customer_id": "synthetic-private-client",
            "AMT_INCOME_TOTAL": 80000.0,
            "AMT_CREDIT": 200000.0,
            "AMT_ANNUITY": 12000.0,
            "EXT_SOURCE_1": 0.8,
        }
        predict_status, predicted = request(url + "/predict", payload)
        explain_status, explained = request(url + "/explain", payload)
        assert predict_status == explain_status == 200
        assert predicted["default_score"] == explained["default_score"]
        assert predicted["model_version"] == explained["model_version"] == "1"
        assert len(explained["reasons"]) == 3
        assert request(url + "/predict", payload | {"CODE_GENDER": "M"})[0] == 422
        info = request(url + "/model-info")[1]
        assert info["raw_dvc_hash"] == "synthetic-packaging-fixture-not-home-credit"
        security = json.loads(
            docker(
                "exec",
                name,
                "/opt/creditS/bin/python",
                "-c",
                "import os,json,importlib.util; from pathlib import Path; "
                "print(json.dumps({'uid':os.getuid(),"
                "'airflow_absent':importlib.util.find_spec('airflow') is None,"
                "'local_venv_absent':not Path('/app/creditS').exists(),"
                "'secret_file_absent':not Path('/app/.env').exists()}))",
            )
        )
        assert security["uid"] == 10001 and all(security[key] for key in security if key != "uid")
        events = json.loads(
            docker(
                "exec",
                name,
                "/opt/creditS/bin/python",
                "-c",
                "import json; from pathlib import Path; "
                "p=Path('/app/runtime/api/predictions.jsonl'); "
                "t=p.read_text(); print(json.dumps({'events':len(t.splitlines()),"
                "'original_id_absent':'synthetic-private-client' not in t}))",
            )
        )
        assert events == {"events": 2, "original_id_absent": True}
        subprocess.run(
            ["docker", "exec", name, "/opt/creditS/bin/python", "-m", "pip", "check"], check=True
        )
        proof = {
            "isolated_health": body,
            "predict_status": predict_status,
            "explain_status": explain_status,
            "security": security,
            "logging": events,
            "registry_transport": "HTTP; model artifacts served by separate MLflow container",
        }
        print("Isolated Docker scoring and SHAP passed.", flush=True)
    except Exception:
        output = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
        registry_output = subprocess.run(
            ["docker", "logs", registry_name], capture_output=True, text=True
        )
        diagnostics = ROOT / "runtime/packaging/fixture-error.log"
        diagnostics.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.write_text(
            output.stdout + output.stderr + registry_output.stdout + registry_output.stderr,
            encoding="utf-8",
        )
        raise
    finally:
        # Only containers and volumes whose unique names this invocation created.
        subprocess.run(
            ["docker", "rm", "-f", name, registry_name], capture_output=True, check=False
        )
        subprocess.run(
            ["docker", "volume", "rm", volume, log_volume], capture_output=True, check=False
        )
        subprocess.run(["docker", "network", "rm", network], capture_output=True, check=False)
    services = docker("compose", "ps", "--format", "{{.Service}} {{.Health}}")
    health = dict(line.split(maxsplit=1) for line in services.splitlines())
    assert all(
        health.get(name) == "healthy"
        for name in ("api", "mlflow", "postgres", "airflow-webserver", "airflow-scheduler")
    )
    env = dict(
        line.split("=", 1)
        for line in (ROOT / ".env").read_text().splitlines()
        if "=" in line and not line.startswith("#")
    )
    api_url = "http://127.0.0.1:" + env.get("API_PORT", "8000")
    live, _ = request(api_url + "/live")
    readiness, _ = request(api_url + "/health")
    assert live == 200 and readiness in {200, 503}
    proof.update(compose_health=health, compose_liveness=live, compose_readiness=readiness)
    for port, endpoint in (
        (env.get("MLFLOW_DOCKER_PORT", "5001"), "/health"),
        (env.get("AIRFLOW_PORT", "8081"), "/health"),
    ):
        with urllib.request.urlopen("http://127.0.0.1:" + port + endpoint, timeout=10) as response:
            assert response.status == 200
    path = ROOT / "runtime/packaging/validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
