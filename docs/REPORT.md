# Trader Performance vs Market Sentiment

## Methodology
- Loaded and profiled both datasets (shape, missing values, duplicates).
- Converted timestamps to daily granularity and aligned trader activity with sentiment by date.
- Computed account-day metrics: daily PnL, win rate, trade frequency, average trade size, long/short ratio, and a risk-intensity proxy.
- Built segment-level views for risk proxy, trading frequency, and consistency.

## Data Preparation Snapshot
| dataset     |   rows |   columns |   missing_values |   duplicate_rows |
|:------------|-------:|----------:|-----------------:|-----------------:|
| fear_greed  |   2644 |         4 |                0 |                0 |
| trader_data | 146316 |        16 |               13 |                0 |

## Fear vs Greed Summary
| sentiment_group   |   observations |   avg_daily_pnl |   median_daily_pnl |   avg_win_rate |   avg_trades_per_day |   avg_risk_proxy |   avg_long_ratio |   avg_trade_size_usd |
|:------------------|---------------:|----------------:|-------------------:|---------------:|---------------------:|-----------------:|-----------------:|---------------------:|
| Fear              |            587 |         5496.17 |            94.0774 |       0.353715 |              88.2675 |          1692.13 |         0.53522  |              7388.76 |
| Greed             |            991 |         3339.25 |           263.914  |       0.354915 |              68.8123 |          2020.55 |         0.483855 |              6234.32 |

## Drawdown Proxy Snapshot
|                 |   count |       mean |      std |       min |         25% |      50% |      75% |     max |
|:----------------|--------:|-----------:|---------:|----------:|------------:|---------:|---------:|--------:|
| worst_daily_pnl |      23 | -40715.5   | 81530.3  | -358963   | -32543.8    | -8394.89 | -479.162 | 0       |
| pnl_q10         |      23 |   -927.159 |  3136.34 |  -14858.9 |    -80.3553 |     0    |    0     | 1.24088 |

## Key Insights
- Performance shifts by sentiment: average daily PnL changes by -2156.92 (Greed minus Fear), with win rate change of 0.001.
- Behavior shifts by sentiment: traders execute -19.46 more trades per account-day during Greed than Fear.
- Risk posture differs: risk proxy is 2020.55 in Greed vs 1692.13 in Fear.

## Actionable Output (2 Rules of Thumb)
- During Fear days, maintain exposure for high-risk accounts only when account-level hit-rate remains above baseline.
- On Greed days, allow higher trade cadence for frequent traders while enforcing tighter stop-loss controls for infrequent traders.

## Artifacts
- Charts: output/charts/
- Tables: output/tables/

## Notes
- The raw trader file does not expose an explicit leverage column, so this report uses a risk proxy: |Size USD| / (|Start Position| + 1).