# Data Science Intern Assignment

Trader Performance vs Market Sentiment (Fear/Greed)

This repository is organized as a clean, reproducible GitHub deliverable with:
- Core must-have analysis (Part A, Part B, Part C)
- Bonus predictive modeling and trader clustering
- Lightweight Streamlit dashboard for exploration

## Repository Structure

- src/analyze_sentiment_trader.py: Core analysis pipeline.
- src/bonus/bonus_modeling.py: Bonus model + clustering pipeline.
- src/bonus/dashboard.py: Streamlit dashboard app.
- run_all.py: One-command runner for core + bonus pipelines.
- docs/REPORT.md: One-page style summary (methodology, insights, strategy ideas).
- docs/DELIVERABLES.md: Submission checklist.
- docs/assignment_requirements.docx: Original assignment prompt.
- data/raw/fear_greed.csv: Sentiment dataset.
- data/raw/trader_data.csv: Trader history dataset.
- output/charts/: Core charts.
- output/tables/: Core tables.
- output/bonus_tables/: Bonus outputs.

## Environment Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Core Analysis

```bash
python src/analyze_sentiment_trader.py
```

This produces:
- Data profiling table (rows, columns, missing values, duplicates)
- Date-aligned account-day metrics
- Fear vs Greed performance/behavior tables
- Segment analysis tables
- Charts supporting at least 3 insights
- docs/REPORT.md with strategy recommendations

## Run Bonus Modeling

```bash
python src/bonus/bonus_modeling.py
```

This produces:
- output/bonus_tables/predictive_model_classification_report.csv
- output/bonus_tables/predictive_model_confusion_matrix.csv
- output/bonus_tables/trader_archetypes.csv
- output/bonus_tables/trader_archetype_profiles.csv

## Run Dashboard

```bash
python -m streamlit run src/bonus/dashboard.py
```

Dashboard sections:
- Sentiment filters and KPI snapshot
- Fear vs Greed distribution visuals
- Behavioral metric comparison
- Segment table explorer
- Bonus archetype scatter view

## Standout Advancements Added

- Advanced dashboard UX:
  - tabbed navigation (Overview, Behavior Deep-Dive, Segments)
  - date-range and account-level filtering
  - Fear-vs-Greed KPI delta cards
  - metric correlation heatmap
  - top-account table and filtered CSV download button
- Reproducibility:
  - dependency lock in requirements.txt
  - one-command orchestrator via run_all.py

## Run Everything At Once

```bash
python run_all.py
```

## Deliverable Mapping to Assignment

- Part A (Data preparation): covered by src/analyze_sentiment_trader.py + output/tables/data_profile.csv
- Part B (Analysis + evidence): covered by output/charts and output/tables/fear_vs_greed_summary.csv
- Part C (Actionable output): covered by docs/REPORT.md strategy section
- Bonus (optional): covered by src/bonus/bonus_modeling.py and src/bonus/dashboard.py

## Notes

- Daily alignment uses date and parsed Timestamp IST.
- The dataset has no explicit leverage field, so a risk proxy is used:
  abs(Size USD) / (abs(Start Position) + 1)

## Push To GitHub

Target repository:
- https://github.com/AshutoshMishra-UJ/primetrade.ai_assignment.git

Commands:

```bash
git init
git add .
git commit -m "Restructure project and finalize deliverables"
git branch -M main
git remote add origin https://github.com/AshutoshMishra-UJ/primetrade.ai_assignment.git
git push -u origin main
```
