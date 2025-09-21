from __future__ import annotations

from typing import Dict, Any, Iterable, Tuple
import re
import pandas as pd


class InsightBot:
    def __init__(self, df: pd.DataFrame, cols: Dict[str, Any]):
        self.df = df
        self.cols = cols
        self.history: list[Tuple[str, str]] = [("assistant", "Ask me about totals, averages, peaks, payment methods, anomalies, or trends.")]

    def ask(self, query: str) -> Iterable[Tuple[str, str]]:
        self.history.append(("user", query))
        response = self._answer(query)
        self.history.append(("assistant", response))
        return self.history[-2:]

    def _answer(self, query: str) -> str:
        q = query.lower().strip()
        amount_col = self.cols.get("amount")
        ts_col = self.cols.get("timestamp")
        cat_col = self.cols.get("category") or self.cols.get("merchant")

        if not amount_col or amount_col not in self.df:
            return "I need an 'amount' column to compute insights."

        df = self.df

        # Totals / averages
        if any(k in q for k in ["total", "sum"]):
            total = pd.to_numeric(df[amount_col], errors="coerce").sum()
            return f"Total amount is {total:,.2f}."
        if any(k in q for k in ["average", "avg", "mean"]):
            avg = pd.to_numeric(df[amount_col], errors="coerce").mean()
            return f"Average amount is {avg:,.2f}."

        # Peak day
        if any(k in q for k in ["peak", "max"]) and ts_col and ts_col in df:
            temp = df.dropna(subset=[ts_col]).copy()
            temp["date"] = pd.to_datetime(temp[ts_col]).dt.date
            daily = temp.groupby("date")[amount_col].sum()
            if not daily.empty:
                day = daily.idxmax()
                val = daily.max()
                return f"Peak day is {day} with {val:,.2f}."

        # Top categories/merchants
        if cat_col and cat_col in df and any(k in q for k in ["top", "category", "merchant"]):
            agg = df.groupby(cat_col)[amount_col].sum().sort_values(ascending=False).head(5)
            items = ", ".join([f"{k}: {v:,.2f}" for k, v in agg.items()])
            return f"Top {cat_col}: {items}."

        # Payment methods
        pm_col = self.cols.get("payment_method")
        if pm_col and pm_col in df and any(k in q for k in ["payment", "method", "card"]):
            agg = df.groupby(pm_col)[amount_col].sum()
            total = float(agg.sum()) or 1.0
            shares = (agg / total).sort_values(ascending=False).head(5)
            items = ", ".join([f"{k}: {v:.1%}" for k, v in shares.items()])
            return f"Payment method share: {items}."

        # Trend (month)
        if any(k in q for k in ["trend", "month", "season"]):
            if ts_col and ts_col in df:
                temp = df.dropna(subset=[ts_col]).copy()
                ts = pd.to_datetime(temp[ts_col], errors="coerce")
                temp["month"] = ts.dt.to_period('M').astype(str)
                series = temp.groupby("month")[amount_col].sum().sort_index()
                if len(series) >= 2:
                    first, last = float(series.iloc[0]), float(series.iloc[-1])
                    change = (last - first) / (abs(first) if first else 1.0)
                    direction = "up" if last >= first else "down"
                    return f"Monthly total moved {direction} {change:.1%} from {series.index[0]} to {series.index[-1]}."

        # Explain anomalies
        if "anomal" in q or "outlier" in q:
            return (
                "We flag anomalies using either Isolation Forest or Z-Score. Isolation Forest isolates points via random splits; points isolated with fewer splits are anomalies. Z-Score flags points beyond 3 standard deviations from the mean. Use filters to narrow context and re-run detection."
            )

        return "Try: 'top merchants', 'payment method share', 'monthly trend', or 'explain anomalies'."


