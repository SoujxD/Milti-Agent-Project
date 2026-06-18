"""Helpers for adapting raw ecommerce datasets into analysis-friendly tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PURCHASE_EVENTS = {"purchase", "purchased", "transaction", "order", "ordered"}
CART_EVENTS = {"cart", "add_to_cart", "addtocart", "checkout"}


@dataclass(slots=True)
class DatasetAdapterResult:
    """Normalized dataframe plus metadata about the detected grain."""

    dataframe: pd.DataFrame
    source_format: str
    raw_rows: int
    analysis_rows: int
    analysis_grain: str


def _normalize(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _find_column(columns: list[str], keywords: list[str]) -> str | None:
    normalized_map = {column: _normalize(column) for column in columns}
    for keyword in keywords:
        probe = _normalize(keyword)
        for column, normalized in normalized_map.items():
            if probe in normalized:
                return column
    return None


def is_event_log_dataset(dataset: pd.DataFrame) -> bool:
    """Detect common raw ecommerce event-log schemas."""
    columns = list(dataset.columns)
    has_event_type = _find_column(columns, ["event_type", "event", "action"]) is not None
    has_session = _find_column(columns, ["user_session", "session_id", "session"]) is not None
    has_time = _find_column(columns, ["event_time", "timestamp", "date_time", "datetime", "date"]) is not None
    return has_event_type and has_session and has_time


def adapt_dataset(dataset: pd.DataFrame) -> DatasetAdapterResult:
    """Return a session-level dataset when the source is a raw event log."""
    raw_rows = len(dataset)
    if not is_event_log_dataset(dataset):
        return DatasetAdapterResult(
            dataframe=dataset.copy(),
            source_format="tabular",
            raw_rows=raw_rows,
            analysis_rows=raw_rows,
            analysis_grain="rows",
        )

    working = dataset.copy()
    columns = list(working.columns)
    session_col = _find_column(columns, ["user_session", "session_id", "session"])
    event_col = _find_column(columns, ["event_type", "event", "action"])
    time_col = _find_column(columns, ["event_time", "timestamp", "date_time", "datetime", "date"])
    category_col = _find_column(columns, ["category_code", "category", "department"])
    brand_col = _find_column(columns, ["brand"])
    user_col = _find_column(columns, ["user_id", "customer_id", "visitor_id"])
    price_col = _find_column(columns, ["price", "amount", "value"])
    product_col = _find_column(columns, ["product_id", "sku", "item_id", "product"])

    if session_col is None or event_col is None or time_col is None:
        return DatasetAdapterResult(
            dataframe=dataset.copy(),
            source_format="tabular",
            raw_rows=raw_rows,
            analysis_rows=raw_rows,
            analysis_grain="rows",
        )

    working[time_col] = pd.to_datetime(working[time_col], errors="coerce", utc=True)
    working["_event_name"] = working[event_col].astype(str).str.strip().str.lower()
    working["_is_purchase"] = working["_event_name"].isin(PURCHASE_EVENTS).astype(int)
    working["_is_cart"] = working["_event_name"].isin(CART_EVENTS).astype(int)
    working["_is_view"] = (working["_event_name"] == "view").astype(int)
    working["event_month"] = working[time_col].dt.strftime("%Y-%m").fillna("Unknown")
    working["event_date"] = working[time_col].dt.date.astype(str)

    def _mode(series: pd.Series, fallback: str = "Unknown") -> str:
        values = series.dropna().astype(str)
        values = values[values.str.strip() != ""]
        if values.empty:
            return fallback
        mode = values.mode()
        return str(mode.iloc[0]) if not mode.empty else str(values.iloc[0])

    aggregations: dict[str, tuple[str, str] | tuple[str, callable]] = {
        "session_start_time": (time_col, "min"),
        "event_month": ("event_month", "first"),
        "event_date": ("event_date", "first"),
        "event_count": (event_col, "size"),
        "purchase_count": ("_is_purchase", "sum"),
        "cart_count": ("_is_cart", "sum"),
        "view_count": ("_is_view", "sum"),
    }
    if user_col:
        aggregations["user_id"] = (user_col, "first")
    if category_col:
        aggregations["dominant_category"] = (category_col, _mode)
        aggregations["unique_categories"] = (category_col, "nunique")
    if brand_col:
        aggregations["dominant_brand"] = (brand_col, _mode)
    if product_col:
        aggregations["unique_products"] = (product_col, "nunique")
    if price_col:
        aggregations["avg_price"] = (price_col, "mean")
        aggregations["max_price"] = (price_col, "max")

    session_df = working.groupby(session_col, dropna=False).agg(**aggregations).reset_index()
    session_df = session_df.rename(columns={session_col: "user_session"})
    session_df["purchased"] = (pd.to_numeric(session_df["purchase_count"], errors="coerce").fillna(0) > 0).astype(int)

    return DatasetAdapterResult(
        dataframe=session_df,
        source_format="event_log",
        raw_rows=raw_rows,
        analysis_rows=len(session_df),
        analysis_grain="sessions",
    )


def load_analysis_dataset(dataset_path: str | Path) -> DatasetAdapterResult:
    """Read a supported tabular file and convert it into an analysis-friendly dataframe when needed."""
    dataset = read_dataset(dataset_path)
    return adapt_dataset(dataset)


def read_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Read common analytics dataset formats used in ecommerce projects."""
    path = Path(dataset_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            with path.open() as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            return pd.DataFrame(records)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported dataset format: {suffix or 'no extension'}")
