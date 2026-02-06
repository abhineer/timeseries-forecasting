"""
Demand Forecasting Demo using Google TimesFM on Retail Data

This script demonstrates the performance of Google's TimesFM model
for demand forecasting on the UCI Online Retail II dataset.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from zipfile import ZipFile
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import timesfm
from huggingface_hub import hf_hub_download
from statsmodels.tsa.arima.model import ARIMA


def download_retail_data() -> pd.DataFrame:
    """
    Download and load the UCI Online Retail II dataset.
    This dataset contains transactions from a UK-based online retailer.
    """
    print("Downloading UCI Online Retail II dataset...")
    url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

    response = requests.get(url)
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as z:
        # The dataset contains an Excel file
        excel_file = [f for f in z.namelist() if f.endswith('.xlsx')][0]
        with z.open(excel_file) as f:
            df = pd.read_excel(f, sheet_name=0)

    print(f"Downloaded {len(df):,} transactions")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the retail data for time series forecasting.
    Aggregates daily demand by product.
    """
    print("Preprocessing data...")

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Convert invoice date to datetime
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])

    # Filter out returns (negative quantities) and invalid records
    df = df[df['quantity'] > 0]
    df = df[df['price'] > 0]
    df = df.dropna(subset=['stockcode', 'invoicedate', 'quantity'])

    # Extract date (remove time component)
    df['date'] = df['invoicedate'].dt.date

    # Aggregate daily demand per product
    daily_demand = df.groupby(['stockcode', 'date'])['quantity'].sum().reset_index()
    daily_demand['date'] = pd.to_datetime(daily_demand['date'])

    print(f"Processed into {len(daily_demand):,} daily demand records")
    return daily_demand


def select_top_products(daily_demand: pd.DataFrame, n_products: int = 10) -> list:
    """
    Select top N products by total quantity sold for forecasting.
    """
    product_totals = daily_demand.groupby('stockcode')['quantity'].sum()
    top_products = product_totals.nlargest(n_products).index.tolist()
    return top_products


def prepare_time_series(daily_demand: pd.DataFrame, stockcode: str) -> tuple:
    """
    Prepare a complete time series for a specific product.
    Fills missing dates with zero demand.
    """
    product_data = daily_demand[daily_demand['stockcode'] == stockcode].copy()
    product_data = product_data.set_index('date').sort_index()

    # Create complete date range
    date_range = pd.date_range(
        start=product_data.index.min(),
        end=product_data.index.max(),
        freq='D'
    )

    # Reindex to fill missing dates with 0
    product_ts = product_data['quantity'].reindex(date_range, fill_value=0)

    return product_ts.index, product_ts.values


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate forecasting accuracy metrics."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # MAPE (avoiding division by zero)
    mask = actual != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = np.nan

    # Symmetric MAPE (handles zeros better)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denominator != 0
    if mask.sum() > 0:
        smape = np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100
    else:
        smape = np.nan

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'sMAPE': smape
    }


def run_timesfm_forecast(
    model: timesfm.TimesFm,
    time_series: np.ndarray,
    horizon: int,
    context_length: int
) -> np.ndarray:
    """Run TimesFM forecast on a single time series."""
    # Use the last context_length points for prediction
    context = time_series[-context_length:].astype(np.float32)

    # Frequency: 0 for daily data
    point_forecast, _ = model.forecast(
        inputs=[context],
        freq=[0]  # 0 = daily frequency
    )

    # Return only the requested horizon
    return point_forecast[0][:horizon]


def run_arima_forecast(
    time_series: np.ndarray,
    horizon: int,
    order: tuple = (5, 1, 2)
) -> np.ndarray:
    """
    Run ARIMA forecast on a single time series.

    Args:
        time_series: Historical data for training
        horizon: Number of steps to forecast
        order: ARIMA (p, d, q) parameters

    Returns:
        Array of forecasted values
    """
    try:
        model = ARIMA(time_series, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=horizon)
        return np.array(forecast)
    except Exception as e:
        # Fallback to simpler model if ARIMA fails
        print(f"    ARIMA fitting failed ({e}), using simple mean forecast")
        return np.full(horizon, np.mean(time_series[-30:]))


def main():
    print("=" * 60)
    print("TimesFM vs ARIMA Demand Forecasting Comparison")
    print("=" * 60)

    # Configuration
    HORIZON = 14  # Forecast 14 days ahead
    CONTEXT_LENGTH = 256  # Use 256 days of history (dataset is ~1 year)
    N_PRODUCTS = 5  # Number of products to forecast

    # Download and preprocess data
    raw_data = download_retail_data()
    daily_demand = preprocess_data(raw_data)

    # Select top products
    top_products = select_top_products(daily_demand, N_PRODUCTS)
    print(f"\nSelected top {N_PRODUCTS} products: {top_products}")

    # Initialize TimesFM model
    print("\nLoading TimesFM model...")
    torch.set_float32_matmul_precision("high")

    # Download checkpoint from HuggingFace
    print("  Downloading checkpoint from HuggingFace...")
    checkpoint_path = hf_hub_download(
        repo_id="google/timesfm-1.0-200m-pytorch",
        filename="torch_model.ckpt"
    )

    # Configure model hyperparameters
    hparams = timesfm.TimesFmHparams(
        context_len=CONTEXT_LENGTH,
        horizon_len=HORIZON,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        num_heads=16,
        model_dims=1280,
        per_core_batch_size=32,
        backend="cpu",  # Use "gpu" if CUDA is available
    )

    # Configure checkpoint
    checkpoint = timesfm.TimesFmCheckpoint(
        version="torch",
        path=checkpoint_path,
    )

    model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
    print("Model loaded successfully!")

    # Store results for both models
    timesfm_results = []
    arima_results = []

    # Create figure for visualizations (2 columns: TimesFM and ARIMA)
    fig, axes = plt.subplots(N_PRODUCTS, 2, figsize=(16, 4 * N_PRODUCTS))
    if N_PRODUCTS == 1:
        axes = [axes]

    print("\n" + "=" * 60)
    print("Running forecasts (TimesFM vs ARIMA)...")
    print("=" * 60)

    for idx, stockcode in enumerate(top_products):
        print(f"\nProduct {idx + 1}/{N_PRODUCTS}: {stockcode}")

        # Prepare time series
        dates, values = prepare_time_series(daily_demand, stockcode)

        # Need enough data for context + horizon
        min_length = CONTEXT_LENGTH + HORIZON
        if len(values) < min_length:
            print(f"  Skipping: insufficient data ({len(values)} < {min_length})")
            continue

        # Split into train (context) and test (horizon)
        train_values = values[:-HORIZON]
        test_values = values[-HORIZON:]
        test_dates = dates[-HORIZON:]

        # === TimesFM Forecast ===
        print("  [TimesFM]")
        timesfm_preds = run_timesfm_forecast(
            model, train_values, HORIZON, CONTEXT_LENGTH
        )
        timesfm_preds = np.maximum(timesfm_preds, 0)

        timesfm_metrics = calculate_metrics(test_values, timesfm_preds)
        timesfm_metrics['stockcode'] = stockcode
        timesfm_metrics['model'] = 'TimesFM'
        timesfm_results.append(timesfm_metrics)

        print(f"    MAE:   {timesfm_metrics['MAE']:.2f}")
        print(f"    sMAPE: {timesfm_metrics['sMAPE']:.2f}%")

        # === ARIMA Forecast ===
        print("  [ARIMA]")
        arima_preds = run_arima_forecast(train_values, HORIZON)
        arima_preds = np.maximum(arima_preds, 0)

        arima_metrics = calculate_metrics(test_values, arima_preds)
        arima_metrics['stockcode'] = stockcode
        arima_metrics['model'] = 'ARIMA'
        arima_results.append(arima_metrics)

        print(f"    MAE:   {arima_metrics['MAE']:.2f}")
        print(f"    sMAPE: {arima_metrics['sMAPE']:.2f}%")

        # Plot historical data (last 60 days before forecast)
        history_start = max(0, len(train_values) - 60)
        history_dates = dates[history_start:-HORIZON]
        history_values = train_values[history_start:]

        # Plot TimesFM results (left column)
        ax1 = axes[idx][0]
        ax1.plot(history_dates, history_values, 'b-', label='Historical', linewidth=1.5)
        ax1.plot(test_dates, test_values, 'g-', label='Actual', linewidth=2)
        ax1.plot(test_dates, timesfm_preds, 'r--', label='TimesFM', linewidth=2)
        ax1.set_title(f'TimesFM | {stockcode} | MAE: {timesfm_metrics["MAE"]:.1f} | sMAPE: {timesfm_metrics["sMAPE"]:.1f}%')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Daily Demand')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot ARIMA results (right column)
        ax2 = axes[idx][1]
        ax2.plot(history_dates, history_values, 'b-', label='Historical', linewidth=1.5)
        ax2.plot(test_dates, test_values, 'g-', label='Actual', linewidth=2)
        ax2.plot(test_dates, arima_preds, 'm--', label='ARIMA', linewidth=2)
        ax2.set_title(f'ARIMA | {stockcode} | MAE: {arima_metrics["MAE"]:.1f} | sMAPE: {arima_metrics["sMAPE"]:.1f}%')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Daily Demand')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('forecast_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: forecast_comparison.png")

    # Combine results for comparison
    all_results = timesfm_results + arima_results

    # Summary statistics
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON: TimesFM vs ARIMA")
    print("=" * 60)

    if not timesfm_results:
        print("\nNo products had sufficient data for forecasting.")
        print("Try reducing CONTEXT_LENGTH or using a larger dataset.")
        return pd.DataFrame()

    # Create comparison dataframes
    timesfm_df = pd.DataFrame(timesfm_results)
    arima_df = pd.DataFrame(arima_results)

    print("\n--- TimesFM Results ---")
    print(timesfm_df[['stockcode', 'MAE', 'RMSE', 'sMAPE']].to_string(index=False))

    print("\n--- ARIMA Results ---")
    print(arima_df[['stockcode', 'MAE', 'RMSE', 'sMAPE']].to_string(index=False))

    # Aggregate comparison
    print("\n" + "=" * 60)
    print("AGGREGATE COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame({
        'Model': ['TimesFM', 'ARIMA'],
        'Avg MAE': [timesfm_df['MAE'].mean(), arima_df['MAE'].mean()],
        'Avg RMSE': [timesfm_df['RMSE'].mean(), arima_df['RMSE'].mean()],
        'Avg sMAPE': [timesfm_df['sMAPE'].mean(), arima_df['sMAPE'].mean()],
    })
    print(comparison.to_string(index=False))

    # Determine winner for each metric
    print("\n--- Winner by Metric ---")
    if timesfm_df['MAE'].mean() < arima_df['MAE'].mean():
        print(f"  MAE:   TimesFM wins ({timesfm_df['MAE'].mean():.2f} vs {arima_df['MAE'].mean():.2f})")
    else:
        print(f"  MAE:   ARIMA wins ({arima_df['MAE'].mean():.2f} vs {timesfm_df['MAE'].mean():.2f})")

    if timesfm_df['RMSE'].mean() < arima_df['RMSE'].mean():
        print(f"  RMSE:  TimesFM wins ({timesfm_df['RMSE'].mean():.2f} vs {arima_df['RMSE'].mean():.2f})")
    else:
        print(f"  RMSE:  ARIMA wins ({arima_df['RMSE'].mean():.2f} vs {timesfm_df['RMSE'].mean():.2f})")

    if timesfm_df['sMAPE'].mean() < arima_df['sMAPE'].mean():
        print(f"  sMAPE: TimesFM wins ({timesfm_df['sMAPE'].mean():.2f}% vs {arima_df['sMAPE'].mean():.2f}%)")
    else:
        print(f"  sMAPE: ARIMA wins ({arima_df['sMAPE'].mean():.2f}% vs {timesfm_df['sMAPE'].mean():.2f}%)")

    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('forecast_metrics.csv', index=False)
    comparison.to_csv('model_comparison.csv', index=False)
    print(f"\nResults saved to: forecast_metrics.csv, model_comparison.csv")

    print("\n" + "=" * 60)
    print("Comparison completed successfully!")
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    results = main()
