from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "output" / "tables"
BONUS_DIR = PROJECT_ROOT / "output" / "bonus_tables"


st.set_page_config(page_title="Trader Sentiment Dashboard", layout="wide")
st.title("Trader Performance vs Sentiment Dashboard")
st.caption("Interactive exploration of Fear/Greed impact, trader behavior, and account archetypes")

if not (TABLES_DIR / "daily_account_with_sentiment.csv").exists():
    st.error("Run src/analyze_sentiment_trader.py first to generate core tables.")
    st.stop()

core = pd.read_csv(TABLES_DIR / "daily_account_with_sentiment.csv")
summary = pd.read_csv(TABLES_DIR / "fear_vs_greed_summary.csv")
segments = pd.read_csv(TABLES_DIR / "account_segments.csv")
core["date"] = pd.to_datetime(core["date"], errors="coerce")

bonus_available = (BONUS_DIR / "trader_archetypes.csv").exists()
if bonus_available:
    archetypes = pd.read_csv(BONUS_DIR / "trader_archetypes.csv")

st.sidebar.header("Filters")
sentiments = st.sidebar.multiselect(
    "Sentiment Group",
    options=sorted(core["sentiment_group"].dropna().unique().tolist()),
    default=[s for s in ["Fear", "Greed"] if s in core["sentiment_group"].unique()],
)

date_min = core["date"].min()
date_max = core["date"].max()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

available_accounts = sorted(core["Account"].dropna().unique().tolist())
selected_accounts = st.sidebar.multiselect(
    "Accounts (optional)",
    options=available_accounts,
    default=[],
)

if sentiments:
    view = core[core["sentiment_group"].isin(sentiments)].copy()
else:
    view = core.copy()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])
    view = view[(view["date"] >= start_date) & (view["date"] <= end_date)]

if selected_accounts:
    view = view[view["Account"].isin(selected_accounts)]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(view):,}")
col2.metric("Accounts", f"{view['Account'].nunique():,}")
col3.metric("Avg Daily PnL", f"{view['daily_pnl'].mean():.2f}")
col4.metric("Avg Win Rate", f"{view['win_rate'].mean():.3f}")

fear_mean = view.loc[view["sentiment_group"] == "Fear", "daily_pnl"].mean()
greed_mean = view.loc[view["sentiment_group"] == "Greed", "daily_pnl"].mean()
delta = greed_mean - fear_mean
st.metric("Greed minus Fear PnL", f"{delta:.2f}")

tab_overview, tab_behavior, tab_segments = st.tabs(["Overview", "Behavior Deep-Dive", "Segments & Archetypes"])

with tab_overview:
    st.subheader("Fear vs Greed Summary")
    st.dataframe(summary, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(view, x="sentiment_group", y="daily_pnl", points=False, title="Daily PnL Distribution by Sentiment")
        st.plotly_chart(fig, width="stretch")

    with c2:
        fig = px.box(view, x="sentiment_group", y="risk_proxy", points=False, title="Risk Proxy Distribution by Sentiment")
        st.plotly_chart(fig, width="stretch")

    daily_curve = (
        view.groupby(["date", "sentiment_group"], as_index=False)["daily_pnl"].mean()
        .sort_values("date")
    )
    fig = px.line(
        daily_curve,
        x="date",
        y="daily_pnl",
        color="sentiment_group",
        title="Average Daily PnL Over Time",
    )
    st.plotly_chart(fig, width="stretch")

with tab_behavior:
    st.subheader("Behavioral Metrics")
    metric_choice = st.selectbox(
        "Metric",
        ["trades_per_day", "avg_trade_size_usd", "long_ratio", "win_rate", "daily_pnl", "risk_proxy"],
    )
    fig = px.violin(
        view,
        x="sentiment_group",
        y=metric_choice,
        box=True,
        points=False,
        title=f"{metric_choice} by Sentiment",
    )
    st.plotly_chart(fig, width="stretch")

    corr_cols = ["daily_pnl", "win_rate", "trades_per_day", "avg_trade_size_usd", "long_ratio", "risk_proxy"]
    corr = view[corr_cols].corr(numeric_only=True)
    heatmap = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
        )
    )
    heatmap.update_layout(title="Metric Correlation Heatmap")
    st.plotly_chart(heatmap, width="stretch")

    top_accounts = (
        view.groupby("Account", as_index=False)
        .agg(
            avg_daily_pnl=("daily_pnl", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_risk_proxy=("risk_proxy", "mean"),
            active_days=("date", "nunique"),
        )
        .sort_values("avg_daily_pnl", ascending=False)
    )
    st.subheader("Top Accounts by Average Daily PnL")
    st.dataframe(top_accounts.head(20), width="stretch")

with tab_segments:
    st.subheader("Segment Table")
    st.dataframe(segments, width="stretch")

    if bonus_available:
        st.subheader("Bonus: Trader Archetypes")
        st.dataframe(
            archetypes[["Account", "cluster", "archetype", "mean_daily_pnl", "mean_win_rate", "mean_risk_proxy"]],
            width="stretch",
        )

        fig = px.scatter(
            archetypes,
            x="mean_risk_proxy",
            y="mean_daily_pnl",
            color="archetype",
            size="mean_trades",
            hover_data=["Account", "mean_win_rate"],
            title="Trader Archetypes: Risk vs PnL",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Run bonus/bonus_modeling.py to enable archetype visuals.")

st.subheader("Download Filtered Dataset")
csv_bytes = view.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download current filtered view as CSV",
    data=csv_bytes,
    file_name="filtered_trader_sentiment_view.csv",
    mime="text/csv",
)
