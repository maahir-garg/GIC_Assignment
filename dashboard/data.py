from __future__ import annotations

import io
from typing import Dict, Any, List
import pandas as pd


def load_transactions(source: Any) -> pd.DataFrame:
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, bytes):
            data = io.BytesIO(data)
        return _read_csv(data)
    if isinstance(source, str):
        return _read_csv(source)
    raise ValueError("Unsupported source type for load_transactions")


def _read_csv(path_or_buffer: Any) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    df = df.copy()

    # Try to normalize common column names
    lower_cols = {c.lower().strip(): c for c in df.columns}

    # Heuristics
    rename_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in {"date", "timestamp", "time", "datetime"}:
            rename_map[c] = "timestamp"
        elif lc in {"amount", "amt", "value", "price"}:
            rename_map[c] = "amount"
        elif lc in {"category", "cat", "type"}:
            rename_map[c] = "category"
        elif lc in {"merchant", "vendor", "counterparty", "payee"}:
            rename_map[c] = "merchant"
        elif lc in {"description", "desc", "memo", "note"}:
            rename_map[c] = "description"
        elif lc in {"id", "txn_id", "transaction_id"}:
            rename_map[c] = "transaction_id"

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Parse timestamp if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Coerce amount
    if "amount" in df.columns:
        df["amount"] = (
            pd.to_numeric(df["amount"], errors="coerce")
            .astype(float)
        )

    # Drop obviously empty rows
    df.dropna(how="all", inplace=True)

    return df


def infer_columns(df: pd.DataFrame) -> Dict[str, Any]:
    columns: Dict[str, Any] = {}
    columns["timestamp"] = "timestamp" if "timestamp" in df.columns else None
    columns["amount"] = "amount" if "amount" in df.columns else None

    # Categorical candidates
    for cand in ["category", "merchant", "payment_method", "account_type", "transaction_type"]:
        columns[cand] = cand if cand in df.columns else None

    # Optional text field
    columns["description"] = "description" if "description" in df.columns else None

    return columns


def filter_dataframe(df: pd.DataFrame, cols: Dict[str, Any], filters: Dict[str, Any]) -> pd.DataFrame:
    output = df.copy()
    ts_col = cols.get("timestamp")
    amt_col = cols.get("amount")

    if ts_col and filters.get("date_range") is not None:
        dr = filters["date_range"]
        if isinstance(dr, (list, tuple)) and len(dr) == 2:
            start, end = dr
            if start is not None:
                output = output[output[ts_col] >= pd.to_datetime(start)]
            if end is not None:
                output = output[output[ts_col] < pd.to_datetime(end) + pd.Timedelta(days=1)]

    if amt_col and (filters.get("min_amount") is not None or filters.get("max_amount") is not None):
        min_amt = filters.get("min_amount")
        max_amt = filters.get("max_amount")
        if min_amt is not None:
            output = output[output[amt_col] >= float(min_amt)]
        if max_amt is not None:
            output = output[output[amt_col] <= float(max_amt)]

    for cat_col in [
        c
        for c in [
            cols.get("category"),
            cols.get("merchant"),
            cols.get("payment_method"),
            cols.get("account_type"),
            cols.get("transaction_type"),
        ]
        if c
    ]:
        selected = filters.get(f"sel_{cat_col}")
        if selected:
            output = output[output[cat_col].isin(selected)]

    # Text contains filter
    desc_col = cols.get("description")
    if desc_col and filters.get("text_search"):
        needle = str(filters["text_search"]).strip()
        if needle:
            output = output[output[desc_col].astype(str).str.contains(needle, case=False, na=False)]

    return output


