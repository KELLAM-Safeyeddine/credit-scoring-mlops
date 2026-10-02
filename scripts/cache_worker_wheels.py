"""Optionally cache Linux wheels using the validated local dependency versions."""

import importlib.metadata as metadata
import json
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".docker-wheels"


def main() -> None:
    """Download binaries concurrently; reuse cached universal wheels when available."""
    if Path(sys.prefix).resolve() != (ROOT / "creditS").resolve():
        raise SystemExit("Use creditS.")
    roots = [
        Requirement(line)
        for line in (ROOT / "airflow/requirements-worker.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    versions, pending, visited = {}, list(roots), set()
    environment = {
        "sys_platform": "linux",
        "os_name": "posix",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
    }
    while pending:
        requirement = pending.pop()
        name = canonicalize_name(requirement.name)
        key = (name, tuple(sorted(requirement.extras)))
        if key in visited:
            continue
        visited.add(key)
        if name == "xgboost-cpu":
            version, dependencies = "3.0.5", ["numpy", "scipy"]
        else:
            try:
                distribution = metadata.distribution(name)
                version, dependencies = distribution.version, distribution.requires or []
            except metadata.PackageNotFoundError:
                with urllib.request.urlopen(
                    f"https://pypi.org/pypi/{name}/json", timeout=30
                ) as response:
                    index = json.load(response)
                version = str(
                    max(
                        Version(value)
                        for value in index["releases"]
                        if Version(value) in requirement.specifier
                        and not Version(value).is_prerelease
                    )
                )
                with urllib.request.urlopen(
                    f"https://pypi.org/pypi/{name}/{version}/json", timeout=30
                ) as response:
                    dependencies = json.load(response)["info"].get("requires_dist") or []
        if version not in requirement.specifier:
            raise ValueError(f"Local version violates the worker requirement: {requirement}")
        versions[name] = version
        for value in dependencies:
            dependency = Requirement(value)
            if dependency.marker is None or any(
                dependency.marker.evaluate({**environment, "extra": extra})
                for extra in {"", *requirement.extras}
            ):
                pending.append(dependency)
    (ROOT / "airflow/requirements-worker-lock.txt").write_text(
        "# Linux worker dependency lock; checked by pip check in the image.\n"
        + "".join(f"{name}=={version}\n" for name, version in sorted(versions.items())),
        encoding="utf-8",
    )
    if "--lock-only" in sys.argv:
        return
    platforms = [f"manylinux_2_{minor}_x86_64" for minor in range(28, 16, -1)] + [
        "manylinux2014_x86_64",
        "manylinux2010_x86_64",
    ]
    tags = list(cpython_tags((3, 11), ["cp311"], platforms)) + list(
        compatible_tags((3, 11), "cp311", platforms)
    )
    ranking = {tag: position for position, tag in enumerate(tags)}
    DEST.mkdir(exist_ok=True)

    def download(item: tuple[str, str]) -> None:
        name, version = item
        cached = list(
            (ROOT / ".wheel-cache").glob(f"{name.replace('-', '_')}-{version}-*none-any.whl")
        )
        if cached:
            shutil.copy2(cached[0], DEST / cached[0].name)
            return
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{name}/{version}/json", timeout=30
        ) as response:
            data = json.load(response)
        wheels = []
        for entry in data["urls"]:
            if not entry["filename"].endswith(".whl"):
                continue
            matching = parse_wheel_filename(entry["filename"])[3].intersection(ranking)
            if matching:
                wheels.append((min(ranking[tag] for tag in matching), entry))
        if not wheels:
            raise ValueError(f"No compatible Linux wheel: {name}=={version}")
        entry = min(wheels, key=lambda value: value[0])[1]
        path = DEST / entry["filename"]
        if path.exists() and path.stat().st_size == entry["size"]:
            return
        temporary = path.with_suffix(".partial")
        with (
            urllib.request.urlopen(entry["url"], timeout=60) as response,
            temporary.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream)
        if temporary.stat().st_size != entry["size"]:
            raise ValueError(f"Incomplete wheel: {name}")
        temporary.replace(path)
        print(f"Cached {name}=={version}", flush=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, sorted(versions.items())))
    (DEST / ".complete").write_text("Linux wheels ready.\n", encoding="utf-8")
    print(f"Cached {len(versions)} Linux-compatible distributions.", flush=True)


if __name__ == "__main__":
    main()
