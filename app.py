import os
from pathlib import Path
import pandas as pd
import streamlit as st

from dashboard.data import load_transactions, infer_columns, filter_dataframe
from dashboard.analytics import compute_kpis
from dashboard.viz import (
    kpi_cards,
    time_series_chart,
    category_bar_chart,
    weekday_hour_heatmap,
    amount_distribution_section,
    stacked_daily_by_type,
    payment_method_donut,
    merchant_analysis,
    monthly_seasonality,
        inflow_outflow_monthly,
        avg_ticket_by_merchant,
)
from dashboard.ui import inject_css, sidebar_filters, fun_header
from dashboard.chatbot import InsightBot
from dashboard.ml import TransactionAnalytics


APP_TITLE = "GIC Transaction Galaxy"
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

    # Overview section
    st.markdown("## Transaction Analysis Dashboard")
    st.markdown("Analyze your financial data using advanced visualizations and machine learning.")

    # Main navigation
    tabs = st.tabs([
        "📊 Overview & Trends",
        "📋 Data Preview",
        "🤖 ML Insights",
        "🔍 Anomaly Detection",
        "💬 Chatbot Assistant"
    ])

    with tabs[0]:
        st.markdown("### Key Metrics & Trends")
        kpis = compute_kpis(filtered_df, cols)
        kpi_cards(kpis)
        
        st.markdown("### Time Series Analysis")
        st.info("Track transaction patterns and seasonal trends over time")
        time_series_chart(filtered_df, cols)
        
        st.info("🕒 Weekly Activity Pattern: Shows transaction frequency by weekday. Helps identify peak activity periods.")
        weekday_hour_heatmap(filtered_df, cols)
        
        st.markdown("### Transaction Breakdown")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📊 Category Distribution")
            category_bar_chart(filtered_df, cols)
        with col2:
            st.info("🏪 Merchant Analysis")
            merchant_analysis(filtered_df, cols)
        with col3:
            st.info("💳 Payment Methods")
            payment_method_donut(filtered_df, cols)

        st.info("📉 Amount Distribution: Visualizes the spread of transaction amounts. Helps identify typical transaction ranges and outliers.")
        amount_distribution_section(filtered_df, cols)
        
        st.info("📋 Daily Transaction Types: Shows how different transaction types stack up each day. Useful for understanding daily composition of transactions.")
        stacked_daily_by_type(filtered_df, cols)
        
        st.info("📅 Monthly Patterns: Displays seasonal patterns in transaction activity. Useful for identifying recurring trends.")
        monthly_seasonality(filtered_df, cols)
        
        st.info("💰 Cash Flow Analysis: Shows monthly inflows and outflows. Helps track net transaction flow over time.")
        inflow_outflow_monthly(filtered_df, cols)
        
        st.info("💸 Average Transaction Size: Displays average transaction amounts by merchant. Helps identify high-value relationships.")
        avg_ticket_by_merchant(filtered_df, cols)

    with tabs[1]:
        st.markdown("### Data Preview & Export")
        st.info("Examine and export filtered transaction data")
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(filtered_df.describe(), width="stretch")
        
        # Data preview
        st.subheader("Raw Data Preview")
        st.dataframe(filtered_df, width="stretch")
        
        # Export options
        st.download_button(
            "📥 Download Filtered Data",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name="filtered_transactions.csv",
            mime="text/csv",
            help="Download current filtered dataset as CSV"
        )

    with tabs[2]:
        st.markdown("### Machine Learning Insights")
        
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Transaction Prediction", "Category Prediction"],
            key="ml_analysis_type"
        )
        
        if analysis_type == "Transaction Prediction":
            st.info("""
            ML-powered analysis provides:
            - Future transaction predictions
            - Pattern detection in historical data
            - Trend forecasting for next 30 days
            """)
            
            if st.toggle("Generate ML Insights", value=False):
                ml_analytics = TransactionAnalytics()
                predictions_df = ml_analytics.predict_trend(filtered_df, cols)
                st.line_chart(predictions_df.set_index('date')['predicted_amount'])
                st.caption("Predicted transaction amounts for next 30 days")
                
        else:
            st.info("""
            Category Prediction:
            - Uses transaction descriptions to predict categories
            - Learns from existing categorized transactions
            - Helps maintain consistent categorization
            """)
            
            description = st.text_input("Enter transaction description")
            if description:
                ml_analytics = TransactionAnalytics()
                vectorizer, model = ml_analytics.suggest_categories(filtered_df, cols)
                if vectorizer and model:
                    features = vectorizer.transform([description])
                    prediction = model.predict(features)[0]
                    confidence = model.predict_proba(features).max()
                    st.success(f"Predicted Category: {prediction} (Confidence: {confidence:.2%})")

    with tabs[3]:
        st.markdown("### Advanced Anomaly Detection")
        method = st.selectbox(
            "Select Analysis Method", 
            ["ML-Enhanced Detection", "Statistical Detection"],
            help="Choose between ML (multiple features) or statistical (amount-based) approach",
            key="anomaly_detection_method"
        )
        
        if method == "ML-Enhanced Detection":
            st.info("""
            🔍 ML-Enhanced Anomaly Detection
            
            This advanced detection method uses multiple features to identify unusual transactions:
            - Transaction timing patterns (hour, day, month)
            - Amount relative to historical patterns
            - Seasonal and cyclic variations
            - Combined behavior scoring
            
            Understanding Anomaly Scores:
            - Score Range: -1 (most anomalous) to 1 (most normal)
            - Scores below 0 are considered anomalies
            - Lower scores indicate more unusual transactions
            - Scores consider both amount and timing patterns
            - The model learns what's "normal" from your data
            """)
            
            ml_analytics = TransactionAnalytics()
            with st.expander("Configure ML Detection", expanded=False):
                contamination = st.slider(
                    "Expected proportion of anomalies",
                    min_value=0.01,
                    max_value=0.20,
                    value=0.05,
                    step=0.01,
                    key="ml_anomaly_contamination"  # Add unique key
                )
                random_state = st.number_input(
                    "Random state",
                    value=42,
                    step=1,
                    key="ml_random_state"  # Add unique key
                )

            anomalies = ml_analytics.detect_complex_anomalies(filtered_df, cols, contamination=contamination, random_state=random_state)
            st.line_chart(anomalies.set_index('timestamp')[['amount', 'anomaly_score']])
            st.caption("""
            Chart shows transaction amounts (light blue) and anomaly scores (dark blue).
            Lower scores indicate more unusual transactions.
            Hover over points to see exact values.
            """)
            st.dataframe(anomalies[anomalies['is_anomaly']].sort_values('anomaly_score'), width="stretch")
            st.caption("""
            Transactions are sorted by anomaly score (most unusual first).
            Review these carefully - they deviate most from normal patterns.
            """)

        else:
            st.markdown("### Statistical Outlier Detection")
            st.info("""
            📊 Statistical Anomaly Detection
            
            This method uses standard statistical approaches to identify outliers:
            - Z-Score Method: Flags values that deviate significantly from the mean
            - Uses 3 standard deviations as the threshold
            - Simple but effective for amount-based anomalies
            - Best for normally distributed transaction amounts
            """)
            
            with st.expander("Configure Detection", expanded=False):
                z_threshold = st.slider(
                    "Z-Score threshold",
                    min_value=2.0,
                    max_value=4.0,
                    value=3.0,
                    step=0.1,
                    help="Number of standard deviations from mean"
                )

            if st.toggle("Run Detection", value=False, help="Toggle to run detection"):
                # Simple robust z-score on amount
                import numpy as np
                amt_col = cols.get("amount")
                if amt_col and amt_col in filtered_df:
                    s = pd.to_numeric(filtered_df[amt_col], errors="coerce").dropna()
                    if not s.empty:
                        z = (s - s.mean()) / (s.std() if s.std() else 1)
                        mask = z.abs() > z_threshold
                        anomalies_df = filtered_df.loc[s.index[mask]].assign(z_score=z[mask])
                        summary = {"num_anomalies": int(mask.sum()), "num_samples": int(s.shape[0])}
                        
                        if anomalies_df is not None and not anomalies_df.empty:
                            st.success(f"Detected {summary['num_anomalies']} outliers out of {summary['num_samples']} samples")
                            st.dataframe(anomalies_df.head(100), width="stretch")
                            st.caption("Tip: Large Z-scores indicate more extreme values. Positive scores are above mean, negative below.")
                        else:
                            st.info("No outliers detected with current threshold.")
                    else:
                        st.warning("No numeric data available for analysis")
                else:
                    st.error("Amount column not found in the data")

    with tabs[4]:
        st.markdown("### AI Analysis Assistant")
        st.info("""
        Natural language interface to analyze your data.
        Try asking about:
        - Transaction trends
        - Category summaries
        - Peak periods
        - Unusual patterns
        """)
        bot = InsightBot(filtered_df, cols)
        user_query = st.chat_input("Type help to understand commands")
        if user_query:
            for role, content in bot.ask(user_query):
                st.chat_message(role).write(content)



if __name__ == "__main__":
    main()


