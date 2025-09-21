import os
from pathlib import Path
import pandas as pd
import streamlit as st

from dashboard.data import load_transactions, infer_columns, filter_dataframe
from dashboard.analytics import compute_kpis, detect_anomalies
from dashboard.viz import (
    kpi_cards,
    time_series_chart,
    category_bar_chart,
    weekday_hour_heatmap,
    amount_distribution_section,
    stacked_daily_by_type,
    payment_method_donut,
    merchant_pareto,
    numeric_correlation_heatmap,
    monthly_seasonality,
        inflow_outflow_monthly,
        avg_ticket_by_merchant,
)
from dashboard.ui import inject_css, sidebar_filters, fun_header
from dashboard.chatbot import InsightBot


APP_TITLE = "GIC Transaction Galaxy ✨"
DEFAULT_CSV_PATH = str(Path(__file__).resolve().parent / "financial_transactions.csv")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="💸", layout="wide")
    inject_css()

    fun_header(APP_TITLE)

    # Always load bundled dataset
    if os.path.exists(DEFAULT_CSV_PATH):
        df = load_transactions(DEFAULT_CSV_PATH)
    else:
        st.error("Bundled dataset not found. Expected financial_transactions.csv alongside app.py")
        st.stop()

    if df.empty:
        st.warning("The dataset is empty after loading.")
        st.stop()

    cols = infer_columns(df)

    # Sidebar: dynamic filters
    with st.sidebar:
        filters = sidebar_filters(df, cols)

    # Apply filters
    filtered_df = filter_dataframe(df, cols, filters)

    # KPI Row
    kpis = compute_kpis(filtered_df, cols)
    kpi_cards(kpis)

    # Tabs for visualizations and anomalies
    tabs = st.tabs(["Visualizations", "Data", "Anomalies", "Chatbot 🤖"])

    with tabs[0]:
        st.markdown("### Visual Explorations")
        time_series_chart(filtered_df, cols)
        col1, col2 = st.columns(2)
        with col1:
            category_bar_chart(filtered_df, cols)
        with col2:
            weekday_hour_heatmap(filtered_df, cols)
        amount_distribution_section(filtered_df, cols)
        stacked_daily_by_type(filtered_df, cols)
        col3, col4 = st.columns(2)
        with col3:
            payment_method_donut(filtered_df, cols)
        with col4:
            merchant_pareto(filtered_df, cols)
        numeric_correlation_heatmap(filtered_df, cols)
        monthly_seasonality(filtered_df, cols)
        inflow_outflow_monthly(filtered_df, cols)
        avg_ticket_by_merchant(filtered_df, cols)

    with tabs[1]:
        st.markdown("### Data Preview & Export")
        st.dataframe(filtered_df.head(500), use_container_width=True)
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_transactions.csv", mime="text/csv")

    with tabs[2]:
        st.markdown("### Outlier Detection")
        st.caption("Anomalies are transactions that deviate strongly from typical patterns. We compute features amount, log(amount), day-of-week, and hour, then score each transaction.")
        method = st.selectbox("Method", ["Isolation Forest", "Z-Score"], index=0, help="Choose algorithm for anomaly detection")
        with st.expander("Configure anomaly detection", expanded=False):
            contamination = st.slider("Expected proportion of anomalies", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
            random_state = st.number_input("Random state", value=42, step=1)

        if st.toggle("Run anomaly detection", value=False, help="Toggle to run detection"):
            if method == "Isolation Forest":
                anomalies_df, summary = detect_anomalies(filtered_df, cols, contamination=contamination, random_state=int(random_state))
                st.caption("Isolation Forest isolates points via random splits; points isolated with fewer splits get lower scores and are flagged as anomalies.")
            else:
                # Simple robust z-score on amount
                import numpy as np
                amt_col = cols.get("amount")
                if amt_col and amt_col in filtered_df:
                    s = pd.to_numeric(filtered_df[amt_col], errors="coerce").dropna()
                    if not s.empty:
                        z = (s - s.mean()) / (s.std() if s.std() else 1)
                        mask = z.abs() > 3
                        anomalies_df = filtered_df.loc[s.index[mask]].assign(z_score=z[mask])
                        summary = {"num_anomalies": int(mask.sum()), "num_samples": int(s.shape[0])}
                    else:
                        anomalies_df, summary = None, {"num_anomalies": 0, "num_samples": 0}
                else:
                    anomalies_df, summary = None, {"num_anomalies": 0, "num_samples": 0}
            if anomalies_df is not None and not anomalies_df.empty:
                st.success(f"Detected {summary['num_anomalies']} anomalies out of {summary['num_samples']} samples")
                st.dataframe(anomalies_df.head(100), use_container_width=True)
                st.caption("Tip: Filter by merchant/category/time to contextualize anomalies. Large positive amounts may be refunds/reversals; large negatives might be one-offs or splits.")
            else:
                st.info("No anomalies detected or insufficient data for detection.")

    with tabs[3]:
        st.markdown("### Insight Chatbot")
        st.caption("The bot answers using the currently FILTERED data, so adjust filters first for targeted insights.")
        bot = InsightBot(filtered_df, cols)
        user_query = st.chat_input("Ask about the data: e.g., 'top categories last month', 'total spend', 'peak hour?' ")
        if user_query:
            for role, content in bot.ask(user_query):
                st.chat_message(role).write(content)

    # Removed celebrate button per request


if __name__ == "__main__":
    main()


