from __future__ import annotations

from typing import Dict, Any
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st

PALETTE = ["#0F4C81", "#1982C4", "#3CB9A6", "#F6C85F", "#F26B38", "#D7263D"]

def _clean_df_time(df: pd.DataFrame, timestamp_col: str, amount_col: str):
    tmp = df.copy()
    if timestamp_col in tmp.columns:
        tmp[timestamp_col] = pd.to_datetime(tmp[timestamp_col], errors="coerce")
        tmp["_date"] = tmp[timestamp_col].dt.date
    else:
        tmp["_date"] = pd.NaT
    tmp["_amount"] = pd.to_numeric(tmp[amount_col], errors="coerce").fillna(0.0)
    return tmp

def time_series_chart(df: pd.DataFrame, timestamp_col: str, amount_col: str):
    tmp = _clean_df_time(df, timestamp_col, amount_col)
    daily = tmp.groupby("_date", dropna=True)["_amount"].sum().reset_index().dropna()
    if daily.empty:
        return alt.Chart(pd.DataFrame({"_date":[],"_amount":[]})).mark_line().encode()

    base = alt.Chart(daily).encode(
        x=alt.X("_date:T", title="Date"),
        y=alt.Y("_amount:Q", title="Total amount", axis=alt.Axis(format="$,.0f")),
        tooltip=[alt.Tooltip("_date:T", title="Date"), alt.Tooltip("_amount:Q", title="Amount", format="$,.2f")]
    )

    selection = alt.selection_interval(encodings=["x"])
    line = base.mark_line(color=PALETTE[0], strokeWidth=2)
    points = base.mark_circle(size=40, color=PALETTE[1]).transform_filter(selection)
    area = base.mark_area(color=PALETTE[0], opacity=0.1)

    chart = (area + line + points).add_selection(selection).properties(height=280)
    return chart

def category_bar_chart(df: pd.DataFrame, category_col: str, amount_col: str, top_n=10):
    if category_col not in df.columns:
        return alt.Chart(pd.DataFrame({category_col:[], "amount":[]})).mark_bar()
    tmp = df.copy()
    tmp["amount"] = pd.to_numeric(tmp[amount_col], errors="coerce").fillna(0.0)
    grp = tmp.groupby(category_col)["amount"].sum().reset_index().sort_values("amount", ascending=False).head(top_n)
    if grp.empty:
        return alt.Chart(grp).mark_bar()
    chart = alt.Chart(grp).mark_bar().encode(
        x=alt.X("amount:Q", title="Total amount", axis=alt.Axis(format="$,.0f")),
        y=alt.Y(f"{category_col}:N", sort=alt.SortField("amount", order="descending"), title="Category"),
        color=alt.Color(f"{category_col}:N", legend=None, scale=alt.Scale(range=PALETTE)),
        tooltip=[alt.Tooltip(f"{category_col}:N", title="Category"), alt.Tooltip("amount:Q", title="Amount", format="$,.2f")]
    ).properties(height=280)
    return chart

def merchant_pareto(df: pd.DataFrame, merchant_col: str, amount_col: str, top_n=15):
    if merchant_col not in df.columns:
        return alt.Chart(pd.DataFrame()).mark_line()
    tmp = df.copy()
    tmp["amount"] = pd.to_numeric(tmp[amount_col], errors="coerce").fillna(0.0)
    agg = tmp.groupby(merchant_col)["amount"].sum().reset_index().sort_values("amount", ascending=False).head(top_n)
    if agg.empty:
        return alt.Chart(agg).mark_bar()
    agg["cum_pct"] = agg["amount"].cumsum() / agg["amount"].sum()
    base = alt.Chart(agg).encode(x=alt.X("index:O", title="Merchant", axis=alt.Axis(labelAngle=-45), sort=None))
    bars = alt.Chart(agg.reset_index()).mark_bar().encode(
        x=alt.X("index:O", title="Merchant"),
        y=alt.Y("amount:Q", title="Amount", axis=alt.Axis(format="$,.0f")),
        tooltip=[alt.Tooltip(merchant_col+':N', title="Merchant"), alt.Tooltip("amount:Q", title="Amount", format="$,.2f")],
        color=alt.value(PALETTE[2])
    )
    line = alt.Chart(agg.reset_index()).mark_line(color=PALETTE[4], strokeWidth=2).encode(
        x="index:O",
        y=alt.Y("cum_pct:Q", title="Cumulative %", axis=alt.Axis(format='%')),
        tooltip=[alt.Tooltip("cum_pct:Q", title="Cumulative %", format='.0%')]
    )
    chart = alt.layer(bars, line).resolve_scale(y='independent').properties(height=260)
    return chart

def numeric_correlation_heatmap(df: pd.DataFrame, amount_col: str, timestamp_col: str):
    cols = {}
    if amount_col in df.columns:
        cols["amount"] = pd.to_numeric(df[amount_col], errors="coerce")
    if timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        cols["hour"] = ts.dt.hour
        cols["dow"] = ts.dt.weekday
    if not cols:
        return alt.Chart(pd.DataFrame()).mark_rect()
    corr_df = pd.DataFrame(cols).corr().stack().reset_index()
    corr_df.columns = ["x", "y", "corr"]
    chart = alt.Chart(corr_df).mark_rect().encode(
        x=alt.X("x:N", title=""),
        y=alt.Y("y:N", title=""),
        color=alt.Color("corr:Q", scale=alt.Scale(scheme="blueorange", domain=[-1,1]), title="Correlation"),
        tooltip=[alt.Tooltip("x:N"), alt.Tooltip("y:N"), alt.Tooltip("corr:Q", format=".2f")]
    ).properties(width=250, height=250)
    return chart

def time_series_with_anomalies(df: pd.DataFrame, anomalies: pd.DataFrame, timestamp_col: str, amount_col: str):
    base = time_series_chart(df, timestamp_col, amount_col)
    if anomalies is None or anomalies.empty:
        return base
    # overlay anomaly points
    a = anomalies.copy()
    a[timestamp_col] = pd.to_datetime(a[timestamp_col], errors="coerce")
    a["_amount"] = pd.to_numeric(a[amount_col], errors="coerce").fillna(0.0)
    pts = alt.Chart(a).mark_circle(color="#D7263D", size=80).encode(
        x=alt.X(f"{timestamp_col}:T"),
        y=alt.Y("_amount:Q", axis=None),
        tooltip=[alt.Tooltip(timestamp_col+':T', title="Date"), alt.Tooltip('_amount:Q', title="Amount", format="$,.2f")]
    )
    return (base + pts).properties(height=300)

def kpi_cards(kpis: Dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Amount", f"{kpis.get('total_amount', 0.0):,.2f}")
    with col2:
        st.metric("Average Amount", f"{kpis.get('avg_amount', 0.0):,.2f}")
    with col3:
        st.metric("Transactions", f"{kpis.get('num_transactions', 0)}")
    with col4:
        if kpis.get("peak_day"):
            day, amt = kpis["peak_day"]
            st.metric("Peak Day", f"{day}", delta=f"{amt:,.2f}")
        else:
            st.metric("Peak Day", "—")


def time_series_chart(df: pd.DataFrame, cols: Dict[str, Any]) -> None:
    ts_col = cols.get("timestamp")
    amt_col = cols.get("amount")
    if not ts_col or not amt_col or ts_col not in df or amt_col not in df:
        st.info("Time series not available: missing timestamp or amount column.")
        return
    data = df.dropna(subset=[ts_col, amt_col]).copy()
    data["date"] = pd.to_datetime(data[ts_col]).dt.date
    agg = data.groupby("date", as_index=False)[amt_col].sum().sort_values("date")
    agg["rolling_7"] = agg[amt_col].rolling(7, min_periods=1).mean()
    base = alt.Chart(agg).encode(x=alt.X("date:T", title="Date"))
    lines = base.mark_line(point=True, color="#7B61FF").encode(
        y=alt.Y(f"{amt_col}:Q", title="Total Amount", axis=alt.Axis(format=",.2f")),
        tooltip=["date:T", alt.Tooltip(f"{amt_col}:Q", format=",.2f"), alt.Tooltip("rolling_7:Q", title="7D Avg", format=",.2f")],
    )
    rolling = base.mark_line(color="#22D3EE", strokeDash=[4, 2]).encode(y="rolling_7:Q")
    chart = (lines + rolling).properties(height=300).interactive()
    st.altair_chart(chart, use_container_width=True)


def category_bar_chart(df: pd.DataFrame, cols: Dict[str, Any]) -> None:
    cat_col = cols.get("category") or cols.get("merchant")
    amt_col = cols.get("amount")
    if not cat_col or not amt_col or cat_col not in df or amt_col not in df:
        st.info("Category bar chart not available: missing categorical or amount column.")
        return
    top_n = st.slider("Top N categories", min_value=5, max_value=30, value=15, step=1, key=f"topn_{cat_col}")
    agg = df.dropna(subset=[cat_col, amt_col]).groupby(cat_col, as_index=False)[amt_col].sum()
    agg = agg.sort_values(amt_col, ascending=False).head(top_n)
    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X(f"{amt_col}:Q", title="Total Amount", axis=alt.Axis(format=",.2f")),
            y=alt.Y(f"{cat_col}:N", sort='-x', title=cat_col.title()),
            tooltip=[alt.Tooltip(f"{cat_col}:N", title="Entity"), alt.Tooltip(f"{amt_col}:Q", title="Total", format=",.2f")],
            color=alt.Color(f"{cat_col}:N", legend=None),
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def weekday_hour_heatmap(df: pd.DataFrame, cols: Dict[str, Any]) -> None:
    ts_col = cols.get("timestamp")
    amt_col = cols.get("amount")
    if not ts_col or not amt_col or ts_col not in df or amt_col not in df:
        st.info("Heatmap not available: missing timestamp or amount column.")
        return
    data = df.dropna(subset=[ts_col, amt_col]).copy()
    ts = pd.to_datetime(data[ts_col], errors="coerce")
    data["weekday"] = ts.dt.day_name()
    data["hour"] = ts.dt.hour
    # Dual heatmaps: counts and average amount
    cnt = data.groupby(["weekday", "hour"], as_index=False).size().rename(columns={"size": "count"})
    amt = data.groupby(["weekday", "hour"], as_index=False)[amt_col].mean().rename(columns={amt_col: "avg_amount"})
    agg = pd.merge(cnt, amt, on=["weekday", "hour"], how="outer").fillna(0)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    heat_count = (
        alt.Chart(agg)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", title="Hour"),
            y=alt.Y("weekday:N", sort=order, title="Weekday"),
            color=alt.Color("count:Q", title="# Transactions", scale=alt.Scale(scheme="blues")),
            tooltip=["weekday:N", "hour:O", alt.Tooltip("count:Q", title="# Txns")],
        )
        .properties(title="# Transactions", height=320)
    )
    heat_avg = (
        alt.Chart(agg)
        .mark_rect()
        .encode(
            x=alt.X("hour:O", title="Hour"),
            y=alt.Y("weekday:N", sort=order, title="Weekday"),
            color=alt.Color("avg_amount:Q", title="Avg Amount", scale=alt.Scale(scheme="magma")),
            tooltip=["weekday:N", "hour:O", alt.Tooltip("avg_amount:Q", title="Avg", format=",.2f")],
        )
        .properties(title="Average Amount", height=320)
    )
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(heat_count, use_container_width=True)
    with c2:
        st.altair_chart(heat_avg, use_container_width=True)


def amount_distribution_section(df: pd.DataFrame, cols: Dict[str, Any]) -> None:
    amt_col = cols.get("amount")
    if not amt_col or amt_col not in df:
        st.info("Amount distribution not available: missing amount column.")
        return
    data = df.dropna(subset=[amt_col]).copy()
    data[amt_col] = pd.to_numeric(data[amt_col], errors="coerce")
    data = data.dropna(subset=[amt_col])
    hist = (
        alt.Chart(data)
        .mark_bar(opacity=0.85, color="#7B61FF")
        .encode(
            x=alt.X(f"{amt_col}:Q", bin=alt.Bin(maxbins=60), title="Amount"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("count():Q", title="Count")],
        )
        .properties(height=240)
    )
    q1 = float(data[amt_col].quantile(0.25)) if not data.empty else 0.0
    med = float(data[amt_col].quantile(0.5)) if not data.empty else 0.0
    q3 = float(data[amt_col].quantile(0.75)) if not data.empty else 0.0
    p95 = float(data[amt_col].quantile(0.95)) if not data.empty else 0.0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Q1", f"{q1:,.2f}")
    col2.metric("Median", f"{med:,.2f}")
    col3.metric("Q3", f"{q3:,.2f}")
    col4.metric("P95", f"{p95:,.2f}")
    st.altair_chart(hist, use_container_width=True)


def stacked_daily_by_type(df: pd.DataFrame, cols: Dict[str, Any]) -> None:
    ts_col = cols.get("timestamp")
    amt_col = cols.get("amount")
    type_col = cols.get("transaction_type")
    if not ts_col or not amt_col or not type_col or not all(c in df for c in [ts_col, amt_col, type_col]):
        return
    data = df.dropna(subset=[ts_col, amt_col, type_col]).copy()
    data["date"] = pd.to_datetime(data[ts_col]).dt.date
    agg = data.groupby(["date", type_col], as_index=False)[amt_col].sum()
    chart = (
        alt.Chart(agg)
        .mark_area()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(f"{amt_col}:Q", stack="zero", title="Total Amount", axis=alt.Axis(format=",.2f")),
            color=alt.Color(f"{type_col}:N", title="Transaction Type"),
            tooltip=["date:T", f"{type_col}:N", alt.Tooltip(f"{amt_col}:Q", format=",.2f")],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def payment_method_donut(df: pd.DataFrame, cols: Dict[str, Any]) -> None:
    pm_col = cols.get("payment_method")
    amt_col = cols.get("amount")
    if not pm_col or not amt_col or pm_col not in df or amt_col not in df:
        return
    agg = df.dropna(subset=[pm_col, amt_col]).groupby(pm_col, as_index=False)[amt_col].sum()
    total = float(agg[amt_col].sum())
    if total == 0:
        return
    agg["share"] = agg[amt_col] / total
    chart = (
        alt.Chart(agg)
        .mark_arc(outerRadius=120, innerRadius=60)
        .encode(
            theta=alt.Theta("share:Q", title="Share"),
            color=alt.Color(f"{pm_col}:N", title="Payment Method"),
            tooltip=[alt.Tooltip(f"{pm_col}:N", title="Method"), alt.Tooltip(f"{amt_col}:Q", title="Total", format=",.2f"), alt.Tooltip("share:Q", title="Share", format=".1%")],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)


def merchant_pareto(df: pd.DataFrame, cols: Dict[str, Any]) -> None:
    merch_col = cols.get("merchant") or cols.get("category")
    amt_col = cols.get("amount")
    if not merch_col or not amt_col or merch_col not in df or amt_col not in df:
        return
    top_n = st.slider("Top N merchants", 5, 30, 15, 1, key="pareto_topn", help="How many top merchants to include")
    agg = df.dropna(subset=[merch_col, amt_col]).groupby(merch_col, as_index=False)[amt_col].sum().sort_values(amt_col, ascending=False)
    agg = agg.head(top_n)
    if agg.empty:
        return
    agg["cum_share"] = agg[amt_col].cumsum() / agg[amt_col].sum()
    base = alt.Chart(agg).encode(x=alt.X(f"{merch_col}:N", sort='-y', title=merch_col.title()))
    bars = base.mark_bar(color="#7C3AED").encode(y=alt.Y(f"{amt_col}:Q", title="Total Amount", axis=alt.Axis(format=",.2f")), tooltip=[alt.Tooltip(f"{merch_col}:N", title="Entity"), alt.Tooltip(f"{amt_col}:Q", title="Total", format=",.2f")])
    # line = base.mark_line(color="#10B981").encode(y=alt.Y("cum_share:Q", axis=alt.Axis(format='%', title="Cumulative Share")))
    st.altair_chart((bars).properties(height=320).interactive(), use_container_width=True)


def numeric_correlation_heatmap(df: pd.DataFrame, cols: Dict[str, Any]) -> None:
    nums = df.select_dtypes(include=['number']).copy()
    if nums.shape[1] < 2:
        return
    corr = nums.corr().stack().reset_index(name='corr')
    corr.columns = ['x', 'y', 'corr']
    chart = (
        alt.Chart(corr)
        .mark_rect()
        .encode(
            x=alt.X('x:O', title=''),
            y=alt.Y('y:O', title=''),
            color=alt.Color('corr:Q', scale=alt.Scale(scheme='purpleorange', domain=[-1, 1]), title='Correlation'),
            tooltip=['x:O', 'y:O', alt.Tooltip('corr:Q', format='.2f')],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)


def monthly_seasonality(df: pd.DataFrame, cols: Dict[str, Any]) -> None:
    ts_col = cols.get("timestamp")
    amt_col = cols.get("amount")
    type_col = cols.get("transaction_type")
    if not ts_col or not amt_col or ts_col not in df or amt_col not in df:
        return
    data = df.dropna(subset=[ts_col, amt_col]).copy()
    ts = pd.to_datetime(data[ts_col], errors="coerce")
    data["month"] = ts.dt.to_period('M').astype(str)
    if type_col and type_col in df:
        agg = data.groupby(["month", type_col], as_index=False)[amt_col].sum()
        chart = (
            alt.Chart(agg)
            .mark_line(point=True)
            .encode(
                x=alt.X("month:O", title="Month"),
                y=alt.Y(f"{amt_col}:Q", title="Total Amount", axis=alt.Axis(format=",.2f")),
                color=alt.Color(f"{type_col}:N", title="Transaction Type"),
                tooltip=["month:O", f"{type_col}:N", alt.Tooltip(f"{amt_col}:Q", format=",.2f")],
            )
            .properties(height=280)
            .interactive()
        )
    else:
        agg = data.groupby(["month"], as_index=False)[amt_col].sum()
        chart = (
            alt.Chart(agg)
            .mark_line(point=True)
            .encode(
                x=alt.X("month:O", title="Month"),
                y=alt.Y(f"{amt_col}:Q", title="Total Amount", axis=alt.Axis(format=",.2f")),
                tooltip=["month:O", alt.Tooltip(f"{amt_col}:Q", format=",.2f")],
            )
            .properties(height=280)
            .interactive()
        )
    st.altair_chart(chart, use_container_width=True)


def inflow_outflow_monthly(df: pd.DataFrame, cols: Dict[str, Any]) -> None:
    ts_col = cols.get("timestamp")
    amt_col = cols.get("amount")
    type_col = cols.get("transaction_type")
    if not ts_col or not amt_col or ts_col not in df or amt_col not in df:
        return
    data = df.dropna(subset=[ts_col, amt_col]).copy()
    ts = pd.to_datetime(data[ts_col], errors="coerce")
    data["month"] = ts.dt.to_period('M').astype(str)
    if type_col and type_col in df:
        pivot = data.pivot_table(index="month", columns=type_col, values=amt_col, aggfunc="sum").fillna(0)
        pivot = pivot.reset_index().melt(id_vars="month", var_name="type", value_name="amount")
    else:
        data["type"] = (data[amt_col] >= 0).map({True: "Inflow", False: "Outflow"})
        pivot = data.groupby(["month", "type"], as_index=False)[amt_col].sum().rename(columns={amt_col: "amount"})
    chart = (
        alt.Chart(pivot)
        .mark_bar()
        .encode(
            x=alt.X("month:O", title="Month"),
            y=alt.Y("amount:Q", title="Amount", axis=alt.Axis(format=",.2f")),
            color=alt.Color("type:N", title="Type"),
            tooltip=["month:O", "type:N", alt.Tooltip("amount:Q", format=",.2f")],
        )
        .properties(height=280)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def avg_ticket_by_merchant(df: pd.DataFrame, cols: Dict[str, Any]) -> None:
    amt_col = cols.get("amount")
    merch_col = cols.get("merchant") or cols.get("category")
    if not amt_col or not merch_col or amt_col not in df or merch_col not in df:
        return
    top_n = st.slider("Top N merchants by avg ticket", 5, 30, 15, 1, key="ticket_topn")
    g = df.dropna(subset=[amt_col, merch_col]).groupby(merch_col)
    agg = g[amt_col].agg(count='count', total='sum', avg='mean').reset_index()
    agg = agg.sort_values('avg', ascending=False).head(top_n)
    chart = (
        alt.Chart(agg)
        .mark_bar(color="#2563EB")
        .encode(
            x=alt.X("avg:Q", title="Average Ticket", axis=alt.Axis(format=",.2f")),
            y=alt.Y(f"{merch_col}:N", sort='-x', title=merch_col.title()),
            tooltip=[alt.Tooltip(f"{merch_col}:N", title="Merchant"), alt.Tooltip("avg:Q", title="Avg", format=",.2f"), alt.Tooltip("count:Q", title="# Txns"), alt.Tooltip("total:Q", title="Total", format=",.2f")],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


