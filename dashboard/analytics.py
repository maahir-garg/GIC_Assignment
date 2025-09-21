from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def compute_kpis(df: pd.DataFrame, cols: Dict[str, Any]) -> Dict[str, Any]:
    amount_col = cols.get("amount")
    ts_col = cols.get("timestamp")

    kpis: Dict[str, Any] = {}
    if amount_col and amount_col in df:
        amounts = df[amount_col].dropna()
        kpis["total_amount"] = float(amounts.sum()) if not amounts.empty else 0.0
        kpis["avg_amount"] = float(amounts.mean()) if not amounts.empty else 0.0
        kpis["num_transactions"] = int(amounts.shape[0])

    if ts_col and ts_col in df:
        ts_series = pd.to_datetime(df[ts_col], errors="coerce").dropna()
        if not ts_series.empty:
            kpis["date_range"] = (ts_series.min(), ts_series.max())
            # Peak day by total amount
            if amount_col and amount_col in df:
                daily = (
                    df.dropna(subset=[ts_col])
                    .assign(_date=pd.to_datetime(df[ts_col]).dt.date)
                    .groupby("_date")[amount_col]
                    .sum()
                )
                if not daily.empty:
                    kpis["peak_day"] = (daily.idxmax(), float(daily.max()))

    return kpis

