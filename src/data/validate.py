"""Validate the labeled application table with Pandera."""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from src.config import load_config, parse_params, project_path

APPLICATION_SCHEMA = pa.DataFrameSchema(
    {
        "SK_ID_CURR": pa.Column("int64", nullable=False, unique=True),
        "TARGET": pa.Column("int64", pa.Check.isin([0, 1]), nullable=False),
        "AMT_INCOME_TOTAL": pa.Column("float64", pa.Check.gt(0), nullable=False),
        "CODE_GENDER": pa.Column(str, pa.Check.isin(["M", "F", "XNA"]), nullable=False),
        "DAYS_BIRTH": pa.Column("int64", pa.Check.lt(0), nullable=False),
    },
    coerce=True,
    strict=False,
)


def validate_application(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate all rows and report failures together."""
    return APPLICATION_SCHEMA.validate(frame, lazy=True)


def load_validated_application(path: Path) -> pd.DataFrame:
    """Read and validate the full labeled source before sampling."""
    return validate_application(pd.read_csv(path))


if __name__ == "__main__":
    cfg = load_config(parse_params())
    frame = load_validated_application(project_path(cfg, "raw") / "application_train.csv")
    print(f"Pandera OK: {len(frame):,} rows; default rate {frame.TARGET.mean():.4%}")
