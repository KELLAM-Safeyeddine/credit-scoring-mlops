"""Strict public inputs: audit attributes are never accepted."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Money = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
Score = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class History(BaseModel):
    """Optional aggregates of history available before the request."""

    model_config = ConfigDict(extra="forbid", strict=True)
    BUREAU_COUNT: Nonnegative | None = None
    BUREAU_ACTIVE_RATE: Score | None = None
    BUREAU_CREDIT_SUM: Nonnegative | None = None
    BUREAU_DEBT_SUM: Nonnegative | None = None
    BUREAU_OVERDUE_MAX: Nonnegative | None = None
    PREV_COUNT: Nonnegative | None = None
    PREV_REFUSAL_RATE: Score | None = None
    PREV_APPLICATION_MEAN: Nonnegative | None = None
    PREV_CREDIT_MEAN: Nonnegative | None = None
    INSTALLMENTS_COUNT: Nonnegative | None = None
    INSTALLMENTS_PAYMENT_COUNT: Nonnegative | None = None
    INSTALLMENTS_DELAY_MEAN: Annotated[float, Field(allow_inf_nan=False)] | None = None
    INSTALLMENTS_LATE_RATE: Score | None = None
    INSTALLMENTS_PAYMENT_GAP_MEAN: Annotated[float, Field(allow_inf_nan=False)] | None = None


class Application(BaseModel):
    """Subset of application data; omitted inputs remain explicitly missing."""

    model_config = ConfigDict(extra="forbid", strict=True)
    customer_id: Annotated[str, Field(min_length=1, max_length=100)]
    AMT_INCOME_TOTAL: Money
    AMT_CREDIT: Money
    AMT_ANNUITY: Money
    AMT_GOODS_PRICE: Money | None = None
    DAYS_EMPLOYED: Annotated[int, Field(le=0)] | None = None
    EXT_SOURCE_1: Score | None = None
    EXT_SOURCE_2: Score | None = None
    EXT_SOURCE_3: Score | None = None
    history: History | None = None
