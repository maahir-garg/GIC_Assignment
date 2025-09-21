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


def filter_dataframe(df: pd.DataFrame, cols: Dict[str, str], filters: Dict[str, Any]) -> pd.DataFrame:
    filtered = df.copy()
    
    # Date range filter
    if 'date_range' in filters and isinstance(filters['date_range'], tuple):
        start_date, end_date = filters['date_range']
        filtered[cols['timestamp']] = pd.to_datetime(filtered[cols['timestamp']])
        mask = (
            (filtered[cols['timestamp']].dt.date >= start_date) & 
            (filtered[cols['timestamp']].dt.date <= end_date)
        )
        filtered = filtered[mask]
    
    # Amount range filter
    if 'amount_range' in filters and isinstance(filters['amount_range'], tuple):
        min_amt, max_amt = filters['amount_range']
        filtered = filtered[
            (filtered[cols['amount']] >= min_amt) & 
            (filtered[cols['amount']] <= max_amt)
        ]
    
    # Category filter
    if 'categories' in filters and filters['categories']:
        filtered = filtered[filtered[cols['category']].isin(filters['categories'])]
    
    # Merchant filter
    if 'merchants' in filters and filters['merchants']:
        filtered = filtered[filtered[cols['merchant']].isin(filters['merchants'])]
    
    # Transaction type filter
    if 'transaction_types' in filters and filters['transaction_types']:
        filtered = filtered[filtered[cols['transaction_type']].isin(filters['transaction_types'])]
    
    # Payment method filter
    if 'payment_methods' in filters and filters['payment_methods']:
        filtered = filtered[filtered[cols['payment_method']].isin(filters['payment_methods'])]
    
    # Amount direction filter
    if 'amount_direction' in filters:
        if filters['amount_direction'] == 'Positive Only':
            filtered = filtered[filtered[cols['amount']] > 0]
        elif filters['amount_direction'] == 'Negative Only':
            filtered = filtered[filtered[cols['amount']] < 0]
    
    # Text search in description
    if 'text_search' in filters and filters['text_search']:
        search_text = filters['text_search'].lower()
        if cols.get('description') in filtered.columns:
            filtered = filtered[
                filtered[cols['description']].str.lower().fillna('').str.contains(search_text)
            ]
    
    return filtered


