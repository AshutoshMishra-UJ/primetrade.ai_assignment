from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
TABLES_DIR = OUTPUT_DIR / "tables"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHARTS_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    sentiment = pd.read_csv(DATA_DIR / "fear_greed.csv")
    traders = pd.read_csv(DATA_DIR / "trader_data.csv", low_memory=False)
    return sentiment, traders


def profile_dataframes(sentiment: pd.DataFrame, traders: pd.DataFrame) -> pd.DataFrame:
    profile = pd.DataFrame(
        {
            "dataset": ["fear_greed", "trader_data"],
            "rows": [len(sentiment), len(traders)],
            "columns": [sentiment.shape[1], traders.shape[1]],
            "missing_values": [int(sentiment.isna().sum().sum()), int(traders.isna().sum().sum())],
            "duplicate_rows": [int(sentiment.duplicated().sum()), int(traders.duplicated().sum())],
        }
    )
    profile.to_csv(TABLES_DIR / "data_profile.csv", index=False)
    return profile


def standardize_data(sentiment: pd.DataFrame, traders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sentiment = sentiment.copy()
    traders = traders.copy()

    sentiment.columns = [c.strip() for c in sentiment.columns]
    traders.columns = [c.strip() for c in traders.columns]

    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce").dt.date
    sentiment["classification"] = sentiment["classification"].astype(str).str.strip()

    traders["trade_dt"] = pd.to_datetime(traders["Timestamp IST"], dayfirst=True, errors="coerce")
    traders["date"] = traders["trade_dt"].dt.date

    numeric_cols = ["Execution Price", "Size Tokens", "Size USD", "Start Position", "Closed PnL", "Fee"]
    for col in numeric_cols:
        traders[col] = pd.to_numeric(traders[col], errors="coerce")

    traders["side_norm"] = traders["Side"].astype(str).str.upper().str.strip()
    traders["is_long"] = traders["side_norm"].isin(["BUY", "LONG"]).astype(int)
    traders["is_short"] = traders["side_norm"].isin(["SELL", "SHORT"]).astype(int)
    traders["is_win"] = (traders["Closed PnL"] > 0).astype(int)
    traders["abs_size_usd"] = traders["Size USD"].abs()
    # Leverage is unavailable directly, so this acts as a risk-intensity proxy.
    traders["risk_proxy"] = traders["abs_size_usd"] / (traders["Start Position"].abs() + 1.0)

    sentiment = sentiment.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    traders = traders.dropna(subset=["date", "Account"])

    return sentiment, traders


def build_daily_account_metrics(traders: pd.DataFrame) -> pd.DataFrame:
    agg = (
        traders.groupby(["Account", "date"], as_index=False)
        .agg(
            daily_pnl=("Closed PnL", "sum"),
            trades_per_day=("Closed PnL", "size"),
            win_rate=("is_win", "mean"),
            avg_trade_size_usd=("abs_size_usd", "mean"),
            long_ratio=("is_long", "mean"),
            short_ratio=("is_short", "mean"),
            risk_proxy=("risk_proxy", "mean"),
            fees=("Fee", "sum"),
        )
        .reset_index(drop=True)
    )
    return agg


def join_sentiment(daily_metrics: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    merged = daily_metrics.merge(sentiment[["date", "classification", "value"]], on="date", how="left")
    merged["classification"] = merged["classification"].fillna("Unknown")

    def coarse_label(label: str) -> str:
        lbl = label.lower()
        if "fear" in lbl:
            return "Fear"
        if "greed" in lbl:
            return "Greed"
        return "Neutral/Other"

    merged["sentiment_group"] = merged["classification"].map(coarse_label)
    return merged


def compute_drawdown_proxy(df: pd.DataFrame) -> pd.DataFrame:
    # Approximate drawdown with worst daily PnL and lower-tail quantile per account.
    draw = (
        df.groupby("Account", as_index=False)
        .agg(
            worst_daily_pnl=("daily_pnl", "min"),
            pnl_q10=("daily_pnl", lambda s: s.quantile(0.10)),
        )
        .reset_index(drop=True)
    )
    return draw


def sentiment_comparison_table(merged: pd.DataFrame) -> pd.DataFrame:
    subset = merged[merged["sentiment_group"].isin(["Fear", "Greed"])].copy()
    table = (
        subset.groupby("sentiment_group", as_index=False)
        .agg(
            observations=("Account", "size"),
            avg_daily_pnl=("daily_pnl", "mean"),
            median_daily_pnl=("daily_pnl", "median"),
            avg_win_rate=("win_rate", "mean"),
            avg_trades_per_day=("trades_per_day", "mean"),
            avg_risk_proxy=("risk_proxy", "mean"),
            avg_long_ratio=("long_ratio", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
        )
        .sort_values("sentiment_group")
    )
    table.to_csv(TABLES_DIR / "fear_vs_greed_summary.csv", index=False)
    return table


def build_segments(merged: pd.DataFrame) -> pd.DataFrame:
    account_level = (
        merged.groupby("Account", as_index=False)
        .agg(
            mean_risk_proxy=("risk_proxy", "mean"),
            mean_trades_per_day=("trades_per_day", "mean"),
            mean_win_rate=("win_rate", "mean"),
            mean_daily_pnl=("daily_pnl", "mean"),
        )
        .fillna(0)
    )

    risk_q70 = account_level["mean_risk_proxy"].quantile(0.7)
    risk_q30 = account_level["mean_risk_proxy"].quantile(0.3)
    freq_q70 = account_level["mean_trades_per_day"].quantile(0.7)
    freq_q30 = account_level["mean_trades_per_day"].quantile(0.3)

    def risk_segment(v: float) -> str:
        if v >= risk_q70:
            return "High Leverage Proxy"
        if v <= risk_q30:
            return "Low Leverage Proxy"
        return "Mid Leverage Proxy"

    def freq_segment(v: float) -> str:
        if v >= freq_q70:
            return "Frequent Trader"
        if v <= freq_q30:
            return "Infrequent Trader"
        return "Moderate Frequency"

    pnl_median = account_level["mean_daily_pnl"].median()

    def consistency_segment(row: pd.Series) -> str:
        if row["mean_win_rate"] >= 0.55 and row["mean_daily_pnl"] >= pnl_median:
            return "Consistent Winner"
        if row["mean_win_rate"] <= 0.45 or row["mean_daily_pnl"] < 0:
            return "Inconsistent"
        return "Mixed"

    account_level["risk_segment"] = account_level["mean_risk_proxy"].map(risk_segment)
    account_level["frequency_segment"] = account_level["mean_trades_per_day"].map(freq_segment)
    account_level["consistency_segment"] = account_level.apply(consistency_segment, axis=1)

    account_level.to_csv(TABLES_DIR / "account_segments.csv", index=False)
    return account_level


def segment_sentiment_performance(merged: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    seg = merged.merge(
        segments[["Account", "risk_segment", "frequency_segment", "consistency_segment"]],
        on="Account",
        how="left",
    )
    filtered = seg[seg["sentiment_group"].isin(["Fear", "Greed"])].copy()

    def summarize(by_col: str) -> pd.DataFrame:
        out = (
            filtered.groupby([by_col, "sentiment_group"], as_index=False)
            .agg(
                observations=("Account", "size"),
                avg_daily_pnl=("daily_pnl", "mean"),
                avg_win_rate=("win_rate", "mean"),
                avg_risk_proxy=("risk_proxy", "mean"),
                avg_trades=("trades_per_day", "mean"),
            )
            .sort_values([by_col, "sentiment_group"])
        )
        out.to_csv(TABLES_DIR / f"segment_{by_col}_summary.csv", index=False)
        return out

    risk = summarize("risk_segment")
    freq = summarize("frequency_segment")
    cons = summarize("consistency_segment")

    combined = pd.concat(
        [
            risk.assign(segment_type="risk"),
            freq.assign(segment_type="frequency"),
            cons.assign(segment_type="consistency"),
        ],
        ignore_index=True,
    )
    return combined


def create_charts(merged: pd.DataFrame, sentiment_table: pd.DataFrame, segment_table: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fear_greed = merged[merged["sentiment_group"].isin(["Fear", "Greed"])].copy()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=sentiment_table,
        x="sentiment_group",
        y="avg_daily_pnl",
        hue="sentiment_group",
        legend=False,
        palette="Set2",
    )
    plt.title("Average Daily PnL: Fear vs Greed")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "avg_daily_pnl_fear_vs_greed.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=sentiment_table,
        x="sentiment_group",
        y="avg_win_rate",
        hue="sentiment_group",
        legend=False,
        palette="Set1",
    )
    plt.title("Average Win Rate: Fear vs Greed")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "avg_winrate_fear_vs_greed.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=sentiment_table,
        x="sentiment_group",
        y="avg_trades_per_day",
        hue="sentiment_group",
        legend=False,
        palette="Blues",
    )
    plt.title("Trade Frequency: Fear vs Greed")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "trade_frequency_fear_vs_greed.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=fear_greed,
        x="sentiment_group",
        y="risk_proxy",
        hue="sentiment_group",
        legend=False,
        palette="Pastel1",
        showfliers=False,
    )
    plt.title("Risk Proxy Distribution by Sentiment")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "risk_proxy_distribution.png", dpi=150)
    plt.close()

    risk_seg = segment_table[(segment_table["segment_type"] == "risk") & (segment_table["risk_segment"].isin(["High Leverage Proxy", "Low Leverage Proxy"]))]
    plt.figure(figsize=(10, 5))
    sns.barplot(data=risk_seg, x="risk_segment", y="avg_daily_pnl", hue="sentiment_group", palette="Set2")
    plt.title("Segment PnL: High vs Low Leverage Proxy")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "segment_risk_pnl.png", dpi=150)
    plt.close()

    freq_seg = segment_table[(segment_table["segment_type"] == "frequency") & (segment_table["frequency_segment"].isin(["Frequent Trader", "Infrequent Trader"]))]
    plt.figure(figsize=(10, 5))
    sns.barplot(data=freq_seg, x="frequency_segment", y="avg_win_rate", hue="sentiment_group", palette="Set3")
    plt.title("Segment Win Rate: Frequent vs Infrequent")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "segment_frequency_winrate.png", dpi=150)
    plt.close()


def generate_insights_and_strategy(sentiment_table: pd.DataFrame, segment_table: pd.DataFrame) -> Dict[str, List[str]]:
    table = sentiment_table.set_index("sentiment_group")
    insights: List[str] = []

    if {"Fear", "Greed"}.issubset(table.index):
        pnl_diff = float(table.loc["Greed", "avg_daily_pnl"] - table.loc["Fear", "avg_daily_pnl"])
        wr_diff = float(table.loc["Greed", "avg_win_rate"] - table.loc["Fear", "avg_win_rate"])
        freq_diff = float(table.loc["Greed", "avg_trades_per_day"] - table.loc["Fear", "avg_trades_per_day"])

        insights.append(
            f"Performance shifts by sentiment: average daily PnL changes by {pnl_diff:.2f} (Greed minus Fear), with win rate change of {wr_diff:.3f}."
        )
        insights.append(
            f"Behavior shifts by sentiment: traders execute {freq_diff:.2f} more trades per account-day during Greed than Fear."
        )
        insights.append(
            f"Risk posture differs: risk proxy is {table.loc['Greed', 'avg_risk_proxy']:.2f} in Greed vs {table.loc['Fear', 'avg_risk_proxy']:.2f} in Fear."
        )

    risk_rows = segment_table[
        (segment_table["segment_type"] == "risk")
        & (segment_table["risk_segment"].isin(["High Leverage Proxy", "Low Leverage Proxy"]))
    ]
    freq_rows = segment_table[
        (segment_table["segment_type"] == "frequency")
        & (segment_table["frequency_segment"].isin(["Frequent Trader", "Infrequent Trader"]))
    ]

    strategies: List[str] = []

    if not risk_rows.empty:
        pivot = risk_rows.pivot(index="risk_segment", columns="sentiment_group", values="avg_daily_pnl")
        if {"High Leverage Proxy", "Low Leverage Proxy"}.issubset(pivot.index) and {"Fear", "Greed"}.issubset(pivot.columns):
            high_fear = float(pivot.loc["High Leverage Proxy", "Fear"])
            low_fear = float(pivot.loc["Low Leverage Proxy", "Fear"])
            if high_fear < low_fear:
                strategies.append(
                    "During Fear days, cap position size (or reduce effective leverage proxy) for high-risk accounts and route more capital to low-risk accounts."
                )
            else:
                strategies.append(
                    "During Fear days, maintain exposure for high-risk accounts only when account-level hit-rate remains above baseline."
                )

    if not freq_rows.empty:
        fp = freq_rows.pivot(index="frequency_segment", columns="sentiment_group", values="avg_win_rate")
        if {"Frequent Trader", "Infrequent Trader"}.issubset(fp.index) and {"Fear", "Greed"}.issubset(fp.columns):
            frequent_greed = float(fp.loc["Frequent Trader", "Greed"])
            infrequent_greed = float(fp.loc["Infrequent Trader", "Greed"])
            if frequent_greed > infrequent_greed:
                strategies.append(
                    "On Greed days, allow higher trade cadence for frequent traders while enforcing tighter stop-loss controls for infrequent traders."
                )
            else:
                strategies.append(
                    "On Greed days, avoid forcing high turnover; prioritize selective entries and keep trade count near infrequent-trader baseline."
                )

    if len(strategies) < 2:
        strategies.extend(
            [
                "Use sentiment as a regime switch: conservative sizing under Fear, selective risk-on under Greed.",
                "Trigger account-level alerts when risk proxy spikes above the 70th percentile during Fear regimes.",
            ]
        )

    return {"insights": insights[:3], "strategies": strategies[:2]}


def write_report(
    profile: pd.DataFrame,
    sentiment_summary: pd.DataFrame,
    drawdown: pd.DataFrame,
    recommendations: Dict[str, List[str]],
) -> None:
    draw_stats = drawdown.select_dtypes(include=["number"]).describe().T
    draw_stats.to_csv(TABLES_DIR / "drawdown_proxy_stats.csv")

    lines = [
        "# Trader Performance vs Market Sentiment",
        "",
        "## Methodology",
        "- Loaded and profiled both datasets (shape, missing values, duplicates).",
        "- Converted timestamps to daily granularity and aligned trader activity with sentiment by date.",
        "- Computed account-day metrics: daily PnL, win rate, trade frequency, average trade size, long/short ratio, and a risk-intensity proxy.",
        "- Built segment-level views for risk proxy, trading frequency, and consistency.",
        "",
        "## Data Preparation Snapshot",
        profile.to_markdown(index=False),
        "",
        "## Fear vs Greed Summary",
        sentiment_summary.to_markdown(index=False),
        "",
        "## Drawdown Proxy Snapshot",
        draw_stats.to_markdown(),
        "",
        "## Key Insights",
    ]

    for insight in recommendations["insights"]:
        lines.append(f"- {insight}")

    lines.extend(["", "## Actionable Output (2 Rules of Thumb)"])
    for rec in recommendations["strategies"]:
        lines.append(f"- {rec}")

    lines.extend(
        [
            "",
            "## Artifacts",
            "- Charts: output/charts/",
            "- Tables: output/tables/",
            "",
            "## Notes",
            "- The raw trader file does not expose an explicit leverage column, so this report uses a risk proxy: |Size USD| / (|Start Position| + 1).",
        ]
    )

    (PROJECT_ROOT / "docs" / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()

    sentiment, traders = load_data()
    profile = profile_dataframes(sentiment, traders)

    sentiment, traders = standardize_data(sentiment, traders)
    daily_account = build_daily_account_metrics(traders)
    merged = join_sentiment(daily_account, sentiment)
    drawdown = compute_drawdown_proxy(merged)

    sentiment_summary = sentiment_comparison_table(merged)
    segments = build_segments(merged)
    segment_table = segment_sentiment_performance(merged, segments)

    create_charts(merged, sentiment_summary, segment_table)

    recommendations = generate_insights_and_strategy(sentiment_summary, segment_table)
    write_report(profile, sentiment_summary, drawdown, recommendations)

    merged.to_csv(TABLES_DIR / "daily_account_with_sentiment.csv", index=False)

    print("Analysis complete.")
    print(f"Report: {PROJECT_ROOT / 'docs' / 'REPORT.md'}")
    print(f"Charts: {CHARTS_DIR}")
    print(f"Tables: {TABLES_DIR}")


if __name__ == "__main__":
    main()
