from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_step(label: str, args: list[str]) -> None:
    print(f"\n=== {label} ===")
    result = subprocess.run([PYTHON, *args], cwd=BASE_DIR)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {label}")


def main() -> None:
    run_step("Core analysis", ["src/analyze_sentiment_trader.py"])
    run_step("Bonus modeling", ["src/bonus/bonus_modeling.py"])

    print("\nAll pipelines completed successfully.")
    print("Next: run dashboard with `python -m streamlit run src/bonus/dashboard.py`")


if __name__ == "__main__":
    main()
