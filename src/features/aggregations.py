"""Aggregate client histories in chunks, excluding observations after application."""

import tempfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

ID = "SK_ID_CURR"


def _history_chunks(
    path: Path,
    ids: pd.Index,
    columns: dict[str, str],
    chunksize: int,
    cutoff: int,
    dates: tuple[str, ...],
) -> Iterable[pd.DataFrame]:
    """Read selected columns with reduced types; retain only eligible client histories."""
    for chunk in pd.read_csv(path, usecols=list(columns), dtype=columns, chunksize=chunksize):
        eligible = chunk[ID].isin(ids)
        for date in dates:
            eligible &= chunk[date].notna() & chunk[date].le(cutoff)
        selected = chunk.loc[eligible].copy()
        if not selected.empty:
            yield selected


def _combine(parts: Iterable[pd.DataFrame], sums: list[str], maxima: list[str]) -> pd.DataFrame:
    """Merge sufficient statistics across chunks; never average chunk averages."""
    result = None
    for part in parts:
        if result is None:
            result = part
            continue
        merged = pd.concat([result, part]).groupby(level=0, sort=True)
        result = pd.concat([merged[sums].sum(), merged[maxima].max()], axis=1)
    if result is None:
        result = pd.DataFrame(columns=sums + maxima, index=pd.Index([], name=ID), dtype="float64")
    return result


def aggregate_bureau(
    path: Path, ids: pd.Index, chunksize: int = 100000, cutoff: int = 0
) -> pd.DataFrame:
    """Summarize bureau credit counts, active share, debt and overdue days."""
    types = {
        ID: "int32",
        "CREDIT_ACTIVE": "category",
        "DAYS_CREDIT": "float32",
        "AMT_CREDIT_SUM": "float64",
        "AMT_CREDIT_SUM_DEBT": "float64",
        "CREDIT_DAY_OVERDUE": "float32",
    }

    def parts() -> Iterable[pd.DataFrame]:
        for chunk in _history_chunks(path, ids, types, chunksize, cutoff, ("DAYS_CREDIT",)):
            chunk["active"] = chunk.CREDIT_ACTIVE.eq("Active").astype("int8")
            grouped = chunk.groupby(ID, sort=True)
            yield grouped.agg(
                count=("DAYS_CREDIT", "size"),
                active=("active", "sum"),
                credit_sum=("AMT_CREDIT_SUM", "sum"),
                credit_n=("AMT_CREDIT_SUM", "count"),
                debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
                debt_n=("AMT_CREDIT_SUM_DEBT", "count"),
                overdue=("CREDIT_DAY_OVERDUE", "max"),
            )

    total = _combine(
        parts(), ["count", "active", "credit_sum", "credit_n", "debt_sum", "debt_n"], ["overdue"]
    ).reindex(ids)
    result = pd.DataFrame(index=ids)
    result["BUREAU_COUNT"] = total["count"].fillna(0)
    result["BUREAU_ACTIVE_RATE"] = total.active / total["count"]
    result["BUREAU_CREDIT_SUM"] = total.credit_sum.where(total.credit_n > 0)
    result["BUREAU_DEBT_SUM"] = total.debt_sum.where(total.debt_n > 0)
    result["BUREAU_OVERDUE_MAX"] = total.overdue
    return result.astype("float32")


def aggregate_previous_application(
    path: Path, ids: pd.Index, chunksize: int = 100000, cutoff: int = 0
) -> pd.DataFrame:
    """Summarize historical application counts, refusals and mean requested credit."""
    types = {
        ID: "int32",
        "DAYS_DECISION": "float32",
        "NAME_CONTRACT_STATUS": "category",
        "AMT_APPLICATION": "float64",
        "AMT_CREDIT": "float64",
    }

    def parts() -> Iterable[pd.DataFrame]:
        for chunk in _history_chunks(path, ids, types, chunksize, cutoff, ("DAYS_DECISION",)):
            chunk["refused"] = chunk.NAME_CONTRACT_STATUS.eq("Refused").astype("int8")
            yield chunk.groupby(ID, sort=True).agg(
                count=("DAYS_DECISION", "size"),
                refused=("refused", "sum"),
                requested_sum=("AMT_APPLICATION", "sum"),
                requested_n=("AMT_APPLICATION", "count"),
                credit_sum=("AMT_CREDIT", "sum"),
                credit_n=("AMT_CREDIT", "count"),
            )

    total = _combine(
        parts(), ["count", "refused", "requested_sum", "requested_n", "credit_sum", "credit_n"], []
    ).reindex(ids)
    result = pd.DataFrame(index=ids)
    result["PREV_COUNT"] = total["count"].fillna(0)
    result["PREV_REFUSAL_RATE"] = total.refused / total["count"]
    result["PREV_APPLICATION_MEAN"] = total.requested_sum / total.requested_n.where(
        total.requested_n > 0
    )
    result["PREV_CREDIT_MEAN"] = total.credit_sum / total.credit_n.where(total.credit_n > 0)
    return result.astype("float32")


def aggregate_installments(
    path: Path, ids: pd.Index, chunksize: int = 100000, cutoff: int = 0, partitions: int = 64
) -> pd.DataFrame:
    """Combine fractional payments per installment using bounded disk partitions."""
    keys = [ID, "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"]
    types = {
        ID: "int32",
        "SK_ID_PREV": "int32",
        "NUM_INSTALMENT_VERSION": "float32",
        "NUM_INSTALMENT_NUMBER": "int32",
        "DAYS_INSTALMENT": "float32",
        "DAYS_ENTRY_PAYMENT": "float32",
        "AMT_INSTALMENT": "float64",
        "AMT_PAYMENT": "float64",
    }
    results = []
    # Keep each client's complete payment history in one temporary partition.
    with tempfile.TemporaryDirectory(prefix=".features-", dir=path.parent.parent) as directory:
        temporary = Path(directory).resolve()
        if temporary.parent != path.parent.parent.resolve():
            raise ValueError("Temporary partitions must stay within the data directory.")
        for chunk in _history_chunks(
            path, ids, types, chunksize, cutoff, ("DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT")
        ):
            for number, part in chunk.groupby(chunk[ID] % partitions, sort=True):
                destination = temporary / f"{int(number)}.csv"
                part.to_csv(
                    destination,
                    mode="a",
                    header=not destination.exists(),
                    index=False,
                    lineterminator="\n",
                )
        for file in sorted(temporary.glob("*.csv")):
            part = pd.read_csv(file, dtype=types)
            events = part.groupby(keys, sort=True).agg(
                due=("DAYS_INSTALMENT", "min"),
                last_payment=("DAYS_ENTRY_PAYMENT", "max"),
                owed=("AMT_INSTALMENT", "max"),
                paid=("AMT_PAYMENT", "sum"),
                paid_n=("AMT_PAYMENT", "count"),
                payment_count=("DAYS_ENTRY_PAYMENT", "size"),
            )
            events["paid"] = events.paid.where(events.paid_n > 0)
            events["delay"] = (events.last_payment - events.due).clip(lower=0)
            events["late"] = events.delay.gt(0).astype("int8")
            events["gap"] = events.owed - events.paid
            results.append(
                events.groupby(level=ID, sort=True).agg(
                    INSTALLMENTS_COUNT=("due", "size"),
                    INSTALLMENTS_PAYMENT_COUNT=("payment_count", "sum"),
                    INSTALLMENTS_DELAY_MEAN=("delay", "mean"),
                    INSTALLMENTS_LATE_RATE=("late", "mean"),
                    INSTALLMENTS_PAYMENT_GAP_MEAN=("gap", "mean"),
                )
            )
    names = [
        "INSTALLMENTS_COUNT",
        "INSTALLMENTS_PAYMENT_COUNT",
        "INSTALLMENTS_DELAY_MEAN",
        "INSTALLMENTS_LATE_RATE",
        "INSTALLMENTS_PAYMENT_GAP_MEAN",
    ]
    result = (
        pd.concat(results) if results else pd.DataFrame(columns=names, dtype="float64")
    ).reindex(ids)
    result.index.name = ID
    result[["INSTALLMENTS_COUNT", "INSTALLMENTS_PAYMENT_COUNT"]] = result[
        ["INSTALLMENTS_COUNT", "INSTALLMENTS_PAYMENT_COUNT"]
    ].fillna(0)
    return result.astype("float32")
