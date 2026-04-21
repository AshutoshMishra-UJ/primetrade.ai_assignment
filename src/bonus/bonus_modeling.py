from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "output"
BONUS_TABLES = OUTPUT_DIR / "bonus_tables"


def ensure_dirs() -> None:
    BONUS_TABLES.mkdir(parents=True, exist_ok=True)


def prepare_daily_account_dataset() -> pd.DataFrame:
    sentiment = pd.read_csv(DATA_DIR / "fear_greed.csv")
    traders = pd.read_csv(DATA_DIR / "trader_data.csv", low_memory=False)

    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce").dt.date
    sentiment["classification"] = sentiment["classification"].astype(str).str.strip()

    traders["trade_dt"] = pd.to_datetime(traders["Timestamp IST"], dayfirst=True, errors="coerce")
    traders["date"] = traders["trade_dt"].dt.date

    num_cols = ["Execution Price", "Size Tokens", "Size USD", "Start Position", "Closed PnL", "Fee"]
    for col in num_cols:
        traders[col] = pd.to_numeric(traders[col], errors="coerce")

    traders["side_norm"] = traders["Side"].astype(str).str.upper().str.strip()
    traders["is_long"] = traders["side_norm"].isin(["BUY", "LONG"]).astype(int)
    traders["is_short"] = traders["side_norm"].isin(["SELL", "SHORT"]).astype(int)
    traders["is_win"] = (traders["Closed PnL"] > 0).astype(int)
    traders["abs_size_usd"] = traders["Size USD"].abs()
    traders["risk_proxy"] = traders["abs_size_usd"] / (traders["Start Position"].abs() + 1.0)

    daily = (
        traders.dropna(subset=["date", "Account"])
        .groupby(["Account", "date"], as_index=False)
        .agg(
            daily_pnl=("Closed PnL", "sum"),
            trades_per_day=("Closed PnL", "size"),
            win_rate=("is_win", "mean"),
            avg_trade_size_usd=("abs_size_usd", "mean"),
            long_ratio=("is_long", "mean"),
            risk_proxy=("risk_proxy", "mean"),
            fee_sum=("Fee", "sum"),
        )
    )

    merged = daily.merge(sentiment[["date", "classification", "value"]], on="date", how="left")
    merged["classification"] = merged["classification"].fillna("Unknown")

    # Build next-day target for each account.
    merged = merged.sort_values(["Account", "date"])
    merged["next_day_pnl"] = merged.groupby("Account")["daily_pnl"].shift(-1)

    def pnl_bucket(v: float) -> str:
        if pd.isna(v):
            return "Unknown"
        if v > 0:
            return "Profit"
        if v < 0:
            return "Loss"
        return "Flat"

    merged["next_day_bucket"] = merged["next_day_pnl"].map(pnl_bucket)
    merged = merged[merged["next_day_bucket"] != "Unknown"].copy()

    merged.to_csv(BONUS_TABLES / "bonus_model_dataset.csv", index=False)
    return merged


def run_predictive_model(df: pd.DataFrame) -> None:
    features = [
        "daily_pnl",
        "trades_per_day",
        "win_rate",
        "avg_trade_size_usd",
        "long_ratio",
        "risk_proxy",
        "fee_sum",
        "value",
        "classification",
    ]
    target = "next_day_bucket"

    model_df = df[features + [target]].copy().dropna(subset=[target])

    X = model_df[features]
    y = model_df[target]

    num_features = [
        "daily_pnl",
        "trades_per_day",
        "win_rate",
        "avg_trade_size_usd",
        "long_ratio",
        "risk_proxy",
        "fee_sum",
        "value",
    ]
    cat_features = ["classification"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_features,
            ),
        ]
    )

    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(BONUS_TABLES / "predictive_model_classification_report.csv")

    cm = confusion_matrix(y_test, preds, labels=sorted(y.unique()))
    cm_df = pd.DataFrame(cm, index=sorted(y.unique()), columns=sorted(y.unique()))
    cm_df.to_csv(BONUS_TABLES / "predictive_model_confusion_matrix.csv")


def run_clustering(df: pd.DataFrame) -> None:
    account_df = (
        df.groupby("Account", as_index=False)
        .agg(
            mean_daily_pnl=("daily_pnl", "mean"),
            std_daily_pnl=("daily_pnl", "std"),
            mean_trades=("trades_per_day", "mean"),
            mean_win_rate=("win_rate", "mean"),
            mean_trade_size=("avg_trade_size_usd", "mean"),
            mean_long_ratio=("long_ratio", "mean"),
            mean_risk_proxy=("risk_proxy", "mean"),
            sentiment_value_mean=("value", "mean"),
        )
        .fillna(0)
    )

    features = [
        "mean_daily_pnl",
        "std_daily_pnl",
        "mean_trades",
        "mean_win_rate",
        "mean_trade_size",
        "mean_long_ratio",
        "mean_risk_proxy",
        "sentiment_value_mean",
    ]

    X = account_df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_scaled)

    account_df["cluster"] = clusters

    archetype_labels = {}
    cluster_profile = account_df.groupby("cluster")[features].mean()

    for c, row in cluster_profile.iterrows():
        if row["mean_risk_proxy"] > cluster_profile["mean_risk_proxy"].median() and row["std_daily_pnl"] > cluster_profile["std_daily_pnl"].median():
            archetype_labels[c] = "High-Risk Opportunists"
        elif row["mean_win_rate"] > cluster_profile["mean_win_rate"].median() and row["mean_daily_pnl"] > cluster_profile["mean_daily_pnl"].median():
            archetype_labels[c] = "Consistent Performers"
        else:
            archetype_labels[c] = "Conservative/Flat Traders"

    account_df["archetype"] = account_df["cluster"].map(archetype_labels)

    account_df.to_csv(BONUS_TABLES / "trader_archetypes.csv", index=False)
    cluster_profile.to_csv(BONUS_TABLES / "trader_archetype_profiles.csv")


def main() -> None:
    ensure_dirs()
    dataset = prepare_daily_account_dataset()
    run_predictive_model(dataset)
    run_clustering(dataset)
    print("Bonus modeling complete.")
    print(f"Saved bonus outputs to: {BONUS_TABLES}")


if __name__ == "__main__":
    main()
