# Deliverables Checklist

## Included in This Repository

- Core analysis script: src/analyze_sentiment_trader.py
- Bonus modeling script: src/bonus/bonus_modeling.py
- Streamlit dashboard: src/bonus/dashboard.py
- Setup instructions: README.md
- One-page summary: docs/REPORT.md
- Core output charts: output/charts/
- Core output tables: output/tables/
- Bonus output tables: output/bonus_tables/
- Dependency file: requirements.txt
- Raw datasets: data/raw/

## Quick Reproduction

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run core analysis

```bash
python src/analyze_sentiment_trader.py
```

3. Run bonus

```bash
python src/bonus/bonus_modeling.py
```

4. Run dashboard

```bash
python -m streamlit run src/bonus/dashboard.py
```

## Suggested GitHub Submission Steps

1. Initialize git repository (if needed).
2. Add all files except ignored ones.
3. Commit with message: "Complete DS assignment with bonus modeling and dashboard".
4. Push to GitHub.
5. Submit repository link in the provided Google Form.
