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


def detect_anomalies(
    df: pd.DataFrame,
    cols: Dict[str, Any],
    contamination: float = 0.05,
    random_state: int = 42,
) -> Tuple[pd.DataFrame | None, Dict[str, Any]]:
    amount_col = cols.get("amount")
    ts_col = cols.get("timestamp")

    if not amount_col or amount_col not in df:
        return None, {"num_anomalies": 0, "num_samples": 0}

    work = df.dropna(subset=[amount_col]).copy()

    # Feature engineering: amount, log_amount, day_of_week, hour
    work["amount"] = pd.to_numeric(work[amount_col], errors="coerce").astype(float)
    work["log_amount"] = np.log1p(work["amount"].clip(lower=0))
    if ts_col and ts_col in df:
        ts = pd.to_datetime(work[ts_col], errors="coerce")
        work["dow"] = ts.dt.weekday.fillna(-1).astype(int)
        work["hour"] = ts.dt.hour.fillna(-1).astype(int)
    else:
        work["dow"] = -1
        work["hour"] = -1

    features = work[["amount", "log_amount", "dow", "hour"]].fillna(0.0)

    if features.shape[0] < 10:
        return None, {"num_anomalies": 0, "num_samples": int(features.shape[0])}

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    preds = model.fit_predict(features)
    scores = model.decision_function(features)

    work["anomaly"] = (preds == -1)
    work["anomaly_score"] = scores

    anomalies = work[work["anomaly"]].sort_values("anomaly_score")

    summary = {
        "num_anomalies": int(anomalies.shape[0]),
        "num_samples": int(work.shape[0]),
    }

    return anomalies, summary


