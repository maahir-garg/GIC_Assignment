from __future__ import annotations

from typing import Dict, Any
import pandas as pd
import streamlit as st
import altair as alt
from typing import Dict
from . import viz, analytics, data


def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Neon vibes */
        .stApp {background: radial-gradient(1000px 600px at 10% 10%, #0d0b1a, #0a0a0a);} 
        h1, h2, h3, h4, h5 {text-shadow: 0 0 10px rgba(123,97,255,0.3);} 
        .metric {backdrop-filter: blur(10px);} 
        .stMetric {border-radius: 12px; padding: 8px 12px; background: rgba(255,255,255,0.05);} 
        .block-container {padding-top: 1.5rem;} 
        .sidebar .sidebar-content {background: rgba(255,255,255,0.02);} 
        .css-1v0mbdj, .st-emotion-cache {backdrop-filter: blur(8px);} 
        </style>
        """,
        unsafe_allow_html=True,
    )


def fun_header(title: str) -> None:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;">
            <h1 style="margin:0">{title}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_filters(df: pd.DataFrame, cols: Dict[str, Any]) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    ts_col = cols.get("timestamp")
    amt_col = cols.get("amount")

    st.subheader("Filters")

    if ts_col and ts_col in df:
        ts = pd.to_datetime(df[ts_col], errors="coerce").dropna()
        if not ts.empty:
            min_date = ts.min().date()
            max_date = ts.max().date()
            dr = st.date_input("Date Range", value=(min_date, max_date))
            # Streamlit returns a single date when using a single date picker; guard for that
            if isinstance(dr, tuple) or isinstance(dr, list):
                if len(dr) == 2:
                    filters["date_range"] = (dr[0], dr[1])
            else:
                # Fallback to full range
                filters["date_range"] = (min_date, max_date)

    if amt_col and amt_col in df:
        vals = pd.to_numeric(df[amt_col], errors="coerce").dropna()
        if not vals.empty:
            min_v, max_v = float(vals.min()), float(vals.max())
            filters["min_amount"], filters["max_amount"] = st.slider(
                "Amount Range",
                min_value=min_v,
                max_value=max_v,
                value=(min_v, max_v),
            )

    for cat_col in [
        c for c in [
            cols.get("category"),
            cols.get("merchant"),
            cols.get("payment_method"),
            cols.get("account_type"),
            cols.get("transaction_type"),
        ] if c and c in df
    ]:
        choices = sorted([str(x) for x in df[cat_col].dropna().unique()])
        filters[f"sel_{cat_col}"] = st.multiselect(f"{cat_col.title()}", choices)

    # Text search in description if available
    desc_col = cols.get("description")
    if desc_col and desc_col in df:
        filters["text_search"] = st.text_input("Search description contains")

    return filters


def render_dashboard(df, inferred_cols: Dict[str, str], run_anomaly_fn):
    """
    Clean, professional dashboard layout.
    df: loaded DataFrame
    inferred_cols: mapping like {'timestamp': 'timestamp', 'amount': 'amount', ...}
    run_anomaly_fn: callable(df, cols, **kwargs) -> (anomalies_df, summary)
    """

    st.set_page_config(page_title="Transactions Dashboard", layout="wide", initial_sidebar_state="expanded")

    # Header
    st.markdown("# Transaction Insights")
    st.markdown("A compact, interactive dashboard to explore transaction patterns, categories, merchants and anomalies. "
                "Use the sidebar to filter data and the info panels for guidance.")

    # Sidebar - filters
    with st.sidebar:
        st.header("Filters & Settings")
        st.markdown("Filter the dataset before visualizing. Tip: start with a small date range for large files.")
        ts_col = inferred_cols.get("timestamp")
        amount_col = inferred_cols.get("amount")

        if ts_col in df.columns:
            min_ts = df[ts_col].min()
            max_ts = df[ts_col].max()
            date_range = st.date_input("Date range", value=(min_ts.date() if min_ts is not None else None, max_ts.date() if max_ts is not None else None))
        else:
            date_range = None

        category_col = inferred_cols.get("category")
        categories = []
        if category_col and category_col in df.columns:
            categories = st.multiselect("Categories", options=sorted(df[category_col].dropna().unique()), default=None)

        amount_min, amount_max = None, None
        if amount_col and amount_col in df.columns:
            amin = float(df[amount_col].min() if not df[amount_col].isna().all() else 0.0)
            amax = float(df[amount_col].max() if not df[amount_col].isna().all() else 1.0)
            amount_min, amount_max = st.slider("Amount range", min_value=amin, max_value=amax, value=(amin, amax))

        st.markdown("---")
        st.header("Anomaly Detection")
        contamination = st.slider("Expected anomaly proportion", min_value=0.01, max_value=0.2, value=0.05, step=0.01)
        run_detection = st.checkbox("Enable anomaly detection", value=False, help="Run an unsupervised model to flag unusual transactions")

    # Apply filters to dataframe (non-destructive)
    filtered = df.copy()
    if ts_col in filtered.columns and date_range and len(date_range) == 2 and all(date_range):
        start, end = date_range
        filtered = filtered[(filtered[ts_col].dt.date >= start) & (filtered[ts_col].dt.date <= end)]
    if category_col and categories:
        filtered = filtered[filtered[category_col].isin(categories)]
    if amount_col in filtered.columns and amount_min is not None:
        filtered = filtered[(filtered[amount_col] >= amount_min) & (filtered[amount_col] <= amount_max)]

    # Top KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns([1.3,1,1,1])
    with kpi1:
        total_amt = filtered[amount_col].sum() if amount_col in filtered.columns else 0
        st.metric("Total amount", f"${total_amt:,.2f}")
        st.caption("Sum of amounts in the filtered dataset")
    with kpi2:
        tx_count = len(filtered)
        st.metric("Transactions", f"{tx_count:,}")
        st.caption("Number of transactions in view")
    with kpi3:
        avg = filtered[amount_col].mean() if amount_col in filtered.columns else 0
        st.metric("Average amount", f"${avg:,.2f}")
        st.caption("Mean transaction amount")
    with kpi4:
        unique_merchants = filtered[inferred_cols.get("merchant")].nunique() if inferred_cols.get("merchant") in filtered.columns else 0
        st.metric("Merchants", f"{unique_merchants:,}")
        st.caption("Distinct merchant/counts")

    st.markdown("---")

    # Charts layout
    col_left, col_right = st.columns([2,1], gap="large")

    with col_left:
        st.subheader("Time series")
        st.caption("Daily aggregate and interactive zoom. Hover points for exact values.")
        ts_chart = viz.time_series_chart(filtered, timestamp_col=inferred_cols.get("timestamp"), amount_col=inferred_cols.get("amount"))
        st.altair_chart(ts_chart, use_container_width=True)

        st.subheader("Category breakdown")
        st.caption("Top categories by total amount; click legend to highlight.")
        cat_chart = viz.category_bar_chart(filtered, category_col=inferred_cols.get("category"), amount_col=inferred_cols.get("amount"))
        st.altair_chart(cat_chart, use_container_width=True)

    with col_right:
        st.subheader("Merchant Pareto")
        st.caption("Top merchants by amount. Cumulative line shows concentration.")
        pareto = viz.merchant_analysis(filtered, merchant_col=inferred_cols.get("merchant"), amount_col=inferred_cols.get("amount"))
        st.altair_chart(pareto, use_container_width=True)

        st.subheader("Correlation (numeric)")
        st.caption("Heatmap of numeric correlations (amount, hour, day). Use to spot relationships.")
        corr_heat = viz.numeric_correlation_heatmap(filtered, amount_col=inferred_cols.get("amount"), timestamp_col=inferred_cols.get("timestamp"))
        st.altair_chart(corr_heat, use_container_width=True)

    st.markdown("---")

    # Anomalies (explainable, optional)
    if run_detection:
        st.subheader("Anomalies")
        st.markdown("Anomalies are flagged using an unsupervised model. They may indicate fraud, data errors, or unusual activity. Review before taking action.")
        anomalies, summary = run_anomaly_fn(filtered, inferred_cols, contamination=contamination)
        if anomalies is None or anomalies.empty:
            st.info("No anomalies found or not enough data to run detection.")
        else:
            st.write(f"Detected {summary.get('num_anomalies', 0)} anomalies out of {summary.get('num_samples', 0)} samples.")
            # show small sample and time-series overlay
            st.dataframe(anomalies.head(50))
            overlay = viz.time_series_with_anomalies(filtered, anomalies, timestamp_col=inferred_cols.get("timestamp"), amount_col=inferred_cols.get("amount"))
            st.altair_chart(overlay, use_container_width=True)

    # Footer guidance
    with st.expander("How to interpret these panels"):
        st.markdown("""
        - Time series: shows aggregated amount over time. Look for spikes or shifts.
        - Category breakdown: shows where money is concentrated; sort and filter to explore.
        - Merchant Pareto: identifies the small number of merchants responsible for majority of spend.
        - Correlation: helps find relationships (e.g., higher amounts at certain hours).
        - Anomalies: unsupervised flags. Investigate raw rows before labeling as fraud.
        """)


