"""Extract and inventory the Home Credit archive idempotently."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

REQUIRED_CSVS = (
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
    "HomeCredit_columns_description.csv",
)


def sha256_file(path: Path) -> str:
    """Hash in bounded memory on Python 3.10 and 3.11."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_archive(root: Path, output: Path | None = None) -> dict:
    """Extract one root archive; skip completed outputs from that same archive."""
    archives = sorted(root.glob("*.zip"))
    if len(archives) != 1:
        raise ValueError(f"Expected one root ZIP archive; found {len(archives)}.")
    archive = archives[0]
    output = output or root / "data/raw"
    output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(archive)
    marker = output / ".extraction_manifest.json"
    previous = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
    with zipfile.ZipFile(archive) as zipped:
        entries = {}
        for entry in zipped.infolist():
            name = PurePosixPath(entry.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or ":" in entry.filename:
                raise ValueError(f"Unsafe archive path: {entry.filename}")
            if entry.is_dir():
                continue
            if len(name.parts) != 1 or name.suffix != ".csv" or name.name in entries:
                raise ValueError(f"Unexpected or duplicate archive member: {entry.filename}")
            entries[name.name] = entry
        missing = sorted(set(REQUIRED_CSVS) - entries.keys())
        if missing:
            raise ValueError(f"Missing required CSVs: {missing}")
        sizes = {name: entry.file_size for name, entry in entries.items()}
        if previous.get("archive_sha256") == source_hash and all(
            (output / name).is_file() and (output / name).stat().st_size == size
            for name, size in sizes.items()
        ):
            print("Extraction skipped: source hash, inventory and file sizes match.")
            return previous
        if shutil.disk_usage(output).free < sum(sizes.values()):
            raise OSError("Insufficient disk space for extraction.")
        for name, entry in entries.items():
            temporary = output / f"{name}.partial"
            try:
                with zipped.open(entry) as source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                if temporary.stat().st_size != entry.file_size:
                    raise OSError(f"Incomplete extraction: {name}")
                temporary.replace(output / name)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"Extracted {name}: {entry.file_size:,} bytes", flush=True)
    manifest = {
        "archive_name": archive.name,
        "archive_sha256": source_hash,
        "required_csvs": list(REQUIRED_CSVS),
        "files": sizes,
        "integrity": "ZIP CRC checked during extraction; repeat runs check source hash and sizes.",
    }
    marker.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / ".gitkeep").unlink(missing_ok=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    extract_archive(args.root.resolve(), args.output)
