# Time Series Forecasting: TimesFM vs ARIMA

A comparison of Google's TimesFM foundation model against traditional ARIMA for retail demand forecasting.

## Overview

This project demonstrates demand forecasting on the UCI Online Retail II dataset using:
- **Google TimesFM** - A 200M parameter foundation model for time series
- **ARIMA** - Traditional statistical forecasting method

## Installation

```bash
# Clone and navigate to the project
cd timeseries_forecasting

# Install dependencies with uv
uv sync
```

## Usage

### Web Interface (Gradio)

```bash
uv run python timeseries_forecasting/app.py
```

Then open http://localhost:7860 in your browser to:
- Upload your CSV time series data
- Select TimesFM or ARIMA model
- Get forecasts with visualizations
- Use the chat assistant to fix CSV format issues

### Command Line Demo

```bash
uv run python timeseries_forecasting/demand_forecasting.py
```

## Dataset

The demo uses the [UCI Online Retail II Dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii):
- 525,461 transactions from a UK-based online retailer
- Date range: December 2009 - December 2011
- Aggregated to daily demand per product

## Results

### Aggregate Comparison (14-day forecast horizon)

| Model | Avg MAE | Avg RMSE | Avg sMAPE |
|-------|---------|----------|-----------|
| **TimesFM** | **136.19** | **302.64** | **101.83%** |
| ARIMA | 206.44 | 331.56 | 111.62% |

TimesFM outperforms ARIMA on all metrics:
- 34% lower MAE
- 9% lower RMSE
- 10% lower sMAPE

### Per-Product Results

| Product | TimesFM MAE | ARIMA MAE | Winner |
|---------|-------------|-----------|--------|
| 21212 | 104.5 | 145.8 | TimesFM |
| 85123A | 86.6 | 133.0 | TimesFM |
| 84077 | 277.4 | 346.4 | TimesFM |
| 85099B | 128.1 | 171.5 | TimesFM |
| 17003 | 84.5 | 235.4 | TimesFM |

## Output Files

- `forecast_comparison.png` - Side-by-side visualization of both models
- `forecast_metrics.csv` - Detailed metrics per product per model
- `model_comparison.csv` - Aggregate comparison summary

## Configuration

Key parameters in `demand_forecasting.py`:
```python
HORIZON = 14          # Forecast 14 days ahead
CONTEXT_LENGTH = 256  # Use 256 days of history
N_PRODUCTS = 5        # Number of top products to forecast
```

## Dependencies

- timesfm[torch] >= 1.3.0
- torch >= 2.0.0
- statsmodels >= 0.14.0
- pandas, numpy, matplotlib, scikit-learn

## References

- [Google TimesFM](https://github.com/google-research/timesfm)
- [UCI Online Retail II Dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
