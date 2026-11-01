# PROGRESS — Air Traffic Delay Dashboard

Assumptions:
- Delayed flight = ARRIVAL_DELAY > 15 minutes.
- Cancelled/diverted flights excluded from delay analysis.
- Numeric airport codes not in airports.csv are dropped.
- Airports with missing lat/lon are excluded from map/graph charts.
- Top-N "worst" lists require >= 500 flights to avoid noise.
- Python 3.13, Windows (PowerShell). App runs at http://127.0.0.1:8050.

## Stages
- [x] Stage 0: Check data/raw contents (flights.csv, airports.csv, airlines.csv present)
- [x] Stage 1: src/preprocess.py — processed parquet/CSV files built
- [x] Stage 2: src/graph_analysis.py — graph_metrics.csv + edge list saved
- [x] Stage 3: src/charts.py — 8 chart functions
- [x] Stage 4: app.py — Dash app with tabs, filters, KPIs, insights
- [x] Stage 5: README.md written; full pipeline run OK; app starts OK

## Notes
- Dashboard loads only data/processed/*; raw flights.csv touched once in Stage 1.
