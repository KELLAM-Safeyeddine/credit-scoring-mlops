"""Run the same monitoring logic locally as the thin Airflow DAG."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import ROOT, load_config, project_path, write_json
from src.orchestration.monitoring import execute, load_batch


def run_batch(cfg: dict, batch_id: str, observed_batch: int) -> dict:
    """Archive a batch at its observation clock without dispatching a DAG locally."""
    workspace = (
        project_path(cfg, "reports") / "monitoring" / batch_id / f"clock_{observed_batch:02d}"
    )
    workspace.mkdir(parents=True, exist_ok=True)
    load_batch(workspace, {"batch_id": batch_id, "observed_batch": observed_batch}, cfg)
    for operation in ("drift_report", "fairness_monitor", "evaluate_alert_rules"):
        execute(operation, workspace, cfg)
    drift = json.loads((workspace / "drift.json").read_text())
    alerts = json.loads((workspace / "alerts.json").read_text())
    fairness = json.loads((workspace / "fairness.json").read_text())
    return {
        **alerts,
        "observed_batch": observed_batch,
        **{
            key: drift[key]
            for key in ("labels_available", "auc", "auc_upper", "score_psi", "drift_feature_share")
        },
        "gender_di": fairness["attributes"]["CODE_GENDER"]["disparate_impact"],
        "age_di": fairness["attributes"]["AGE_GROUP"]["disparate_impact"],
        "population_age_psi": drift["population_psi"]["AGE_GROUP"],
        "reports": str(workspace),
    }


def main() -> None:
    """Evaluate one clock or verify all eight scenarios after delayed labels arrive."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    parser.add_argument("--batch-id", choices=[f"{i:02d}" for i in range(1, 9)])
    parser.add_argument("--observed-batch", type=int)
    args = parser.parse_args()
    cfg = load_config(args.params)
    batches = [args.batch_id] if args.batch_id else [f"{i:02d}" for i in range(1, 9)]
    rows = []
    for batch in batches:
        clock = (
            args.observed_batch
            if args.observed_batch is not None
            else int(batch) + cfg["monitoring"]["label_delay_batches"]
        )
        value = run_batch(cfg, batch, clock)
        rows.append(value)
        print(json.dumps(value, ensure_ascii=True), flush=True)
    path = project_path(cfg, "reports") / "monitoring"
    if not args.batch_id:
        assert [row["trigger_retraining"] for row in rows] == [False] * 3 + [True] * 5
        assert "performance" in rows[-1]["alerts"]
        write_json(path / "validation.json", rows)
        pd.DataFrame(rows).to_csv(path / "summary.csv", index=False)
        print("All expected scenario alerts verified.", flush=True)


if __name__ == "__main__":
    main()
