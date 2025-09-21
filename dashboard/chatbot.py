from __future__ import annotations

from typing import Dict, Any, Iterable, Tuple
import re
import pandas as pd


class InsightBot:
    def __init__(self, df: pd.DataFrame, cols: Dict[str, str]):
        self.df = df
        self.cols = cols
        self.commands = {
            "total": self._get_total,
            "average": self._get_average,
            "top": self._get_top_items,
            "trend": self._get_trend,
            "summary": self._get_summary,
            "peak": self._get_peak_times,
            "compare": self._compare_periods,
            "help": self._get_help
        }
        self.history: list[Tuple[str, str]] = [("assistant", "Ask me about totals, averages, peaks, payment methods, anomalies, or trends.")]

    def ask(self, query: str) -> Iterable[Tuple[str, str]]:
        """Process user query and return responses"""
        query = query.lower().strip()
        self.history.append(("user", query))

        try:
            # Handle help request
            if "help" in query or "?" in query:
                response = self._get_help()
            # Process specific analysis requests
            elif "total" in query:
                response = self._get_total()
            elif "average" in query:
                response = self._get_average()
            elif "top" in query:
                n = 5  # default
                response = self._get_top_items(n)
            elif "trend" in query:
                response = self._get_trend()
            elif "compare" in query:
                response = self._compare_periods()
            elif "peak" in query:
                response = self._get_peak_times()
            elif "summary" in query:
                response = self._get_summary()
            else:
                response = "I can help you analyze your financial data. Try asking about:\n" + self._get_help()
        except Exception as e:
            response = f"I encountered an error: {str(e)}\nTry asking for 'help' to see available commands."

        self.history.append(("assistant", response))
        return self.history[-2:]

    def _get_total(self) -> str:
        """Calculate total transaction amount"""
        if self.cols.get("amount") in self.df.columns:
            total = self.df[self.cols["amount"]].sum()
            return f"Total transaction amount: {total:,.2f}"
        return "Could not find amount column in the data."

    def _get_average(self) -> str:
        """Calculate average transaction amount"""
        if self.cols.get("amount") in self.df.columns:
            avg = self.df[self.cols["amount"]].mean()
            median = self.df[self.cols["amount"]].median()
            return f"Average transaction: {avg:,.2f}\nMedian transaction: {median:,.2f}"
        return "Could not find amount column in the data."

    def _get_top_items(self, n: int = 5) -> str:
        """Get top categories/merchants by transaction volume"""
        if self.cols.get("category") in self.df.columns:
            top_cats = self.df.groupby(self.cols["category"])[self.cols["amount"]].sum()
            top_cats = top_cats.sort_values(ascending=False).head(n)
            result = "Top categories by volume:\n"
            for cat, amount in top_cats.items():
                result += f"- {cat}: {amount:,.2f}\n"
            return result
        return "Could not find category information in the data."

    def _get_trend(self) -> str:
        """Analyze transaction trends"""
        if self.cols.get("timestamp") in self.df.columns:
            monthly = self.df.set_index(self.cols["timestamp"]).resample('M')[self.cols["amount"]].sum()
            trend = "increasing" if monthly.iloc[-1] > monthly.iloc[0] else "decreasing"
            return f"Transaction volume is {trend}. Last month: {monthly.iloc[-1]:,.2f}, First month: {monthly.iloc[0]:,.2f}"
        return "Could not analyze trends - missing timestamp information."

    def _get_peak_times(self) -> str:
        """Find peak transaction times"""
        if self.cols.get("timestamp") in self.df.columns:
            df = self.df.copy()
            df['hour'] = pd.to_datetime(df[self.cols["timestamp"]]).dt.hour
            df['day'] = pd.to_datetime(df[self.cols["timestamp"]]).dt.day_name()
            peak_hour = df.groupby('hour')[self.cols["amount"]].sum().idxmax()
            peak_day = df.groupby('day')[self.cols["amount"]].sum().idxmax()
            return f"Peak transaction day: {peak_day}\nPeak transaction hour: {peak_hour}:00"
        return "Could not determine peak times - missing timestamp information."

    def _get_summary(self) -> str:
        """Provide a comprehensive summary"""
        summary = []
        summary.append(self._get_total())
        summary.append(self._get_average())
        summary.append(self._get_trend())
        return "\n\n".join(summary)

    def _compare_periods(self) -> str:
        """Compare current period with previous"""
        if self.cols.get("timestamp") in self.df.columns:
            df = self.df.copy()
            df['month'] = pd.to_datetime(df[self.cols["timestamp"]]).dt.to_period('M')
            current = df[df['month'] == df['month'].max()][self.cols["amount"]].sum()
            previous = df[df['month'] == df['month'].max() - 1][self.cols["amount"]].sum()
            change = ((current - previous) / previous) * 100
            return f"Current period: {current:,.2f}\nPrevious period: {previous:,.2f}\nChange: {change:.1f}%"
        return "Could not compare periods - missing timestamp information."

    def _get_help(self) -> str:
        """Return help text with available commands"""
        return """Available commands:
- total: Get total transaction amount
- average: Get average and median transaction amounts
- top: View top categories by volume
- trend: Analyze transaction trends
- peak: Find peak transaction times
- summary: Get comprehensive analysis
- compare: Compare current with previous period
- help: Show this help message"""


