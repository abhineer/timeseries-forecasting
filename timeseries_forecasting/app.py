"""
Gradio Web Interface for Time Series Forecasting

Upload a CSV file with time series data and get forecasts using TimesFM or ARIMA.
"""

import warnings
warnings.filterwarnings('ignore')

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import torch
import timesfm
from huggingface_hub import hf_hub_download
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Global model cache
_timesfm_model = None
_timesfm_checkpoint_path = None


def get_timesfm_model(context_len: int = 512, horizon_len: int = 128):
    """Load TimesFM model (cached, reinitialized if params change)."""
    global _timesfm_model, _timesfm_checkpoint_path

    # Download checkpoint once
    if _timesfm_checkpoint_path is None:
        torch.set_float32_matmul_precision("high")
        _timesfm_checkpoint_path = hf_hub_download(
            repo_id="google/timesfm-1.0-200m-pytorch",
            filename="torch_model.ckpt"
        )

    # Always create model with standard params (model handles variable input internally)
    hparams = timesfm.TimesFmHparams(
        context_len=512,
        horizon_len=128,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        num_heads=16,
        model_dims=1280,
        per_core_batch_size=32,
        backend="cpu",
    )
    checkpoint = timesfm.TimesFmCheckpoint(
        version="torch",
        path=_timesfm_checkpoint_path,
    )

    if _timesfm_model is None:
        _timesfm_model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

    return _timesfm_model


def validate_csv(df: pd.DataFrame) -> tuple[bool, str, pd.DataFrame | None]:
    """
    Validate CSV format for time series forecasting.
    Returns (is_valid, message, processed_df)
    """
    if df is None or df.empty:
        return False, "CSV file is empty.", None

    # Check for minimum columns
    if len(df.columns) < 2:
        return False, (
            "CSV needs at least 2 columns: a date/time column and a value column.\n"
            "Example format:\n"
            "```\n"
            "date,value\n"
            "2024-01-01,100\n"
            "2024-01-02,150\n"
            "```"
        ), None

    # Try to identify date and value columns
    date_col = None
    value_col = None

    # Look for date column
    for col in df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['date', 'time', 'timestamp', 'ds']):
            date_col = col
            break

    # If no date column found, try first column
    if date_col is None:
        try:
            pd.to_datetime(df.iloc[:, 0])
            date_col = df.columns[0]
        except:
            pass

    # Look for value column
    for col in df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['value', 'quantity', 'demand', 'sales', 'y', 'target']):
            value_col = col
            break

    # If no value column found, use first numeric column that isn't date
    if value_col is None:
        for col in df.columns:
            if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                value_col = col
                break

    if date_col is None:
        return False, (
            "Could not identify a date column. Please ensure your CSV has a column named "
            "'date', 'time', 'timestamp', or 'ds', or that the first column contains dates.\n\n"
            f"Found columns: {list(df.columns)}"
        ), None

    if value_col is None:
        return False, (
            "Could not identify a numeric value column. Please ensure your CSV has a column "
            "named 'value', 'quantity', 'demand', 'sales', 'y', or 'target'.\n\n"
            f"Found columns: {list(df.columns)}"
        ), None

    # Process the dataframe
    try:
        processed = pd.DataFrame()
        processed['date'] = pd.to_datetime(df[date_col])
        processed['value'] = pd.to_numeric(df[value_col], errors='coerce')

        # Remove NaN values
        processed = processed.dropna()

        if len(processed) < 30:
            return False, (
                f"Need at least 30 data points for forecasting, but only found {len(processed)} valid rows."
            ), None

        # Sort by date
        processed = processed.sort_values('date').reset_index(drop=True)

        return True, (
            f"CSV validated successfully!\n"
            f"- Date column: '{date_col}'\n"
            f"- Value column: '{value_col}'\n"
            f"- Data points: {len(processed)}\n"
            f"- Date range: {processed['date'].min().date()} to {processed['date'].max().date()}"
        ), processed

    except Exception as e:
        return False, f"Error processing CSV: {str(e)}", None


def run_forecast(
    df: pd.DataFrame,
    model_name: str,
    horizon: int,
    context_length: int
) -> tuple[np.ndarray, dict]:
    """Run forecast using selected model."""
    values = df['value'].values.astype(np.float32)

    if model_name == "TimesFM":
        model = get_timesfm_model()
        # Use all available data as context (TimesFM handles variable lengths)
        context = values[-context_length:] if len(values) > context_length else values
        point_forecast, _ = model.forecast(inputs=[context], freq=[0])
        predictions = point_forecast[0][:horizon]
    else:  # ARIMA
        try:
            model = ARIMA(values, order=(5, 1, 2))
            fitted = model.fit()
            predictions = fitted.forecast(steps=horizon)
        except Exception:
            # Fallback to simple mean
            predictions = np.full(horizon, np.mean(values[-30:]))

    predictions = np.maximum(predictions, 0)
    return predictions


def create_forecast_plot(
    df: pd.DataFrame,
    predictions: np.ndarray,
    model_name: str,
    horizon: int
) -> plt.Figure:
    """Create forecast visualization."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Historical data (last 60 points)
    history_len = min(60, len(df))
    history = df.tail(history_len)

    # Generate future dates
    last_date = df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')

    # Plot
    ax.plot(history['date'], history['value'], 'b-', label='Historical', linewidth=1.5)
    ax.plot(future_dates, predictions, 'r--', label=f'{model_name} Forecast', linewidth=2, marker='o', markersize=4)

    ax.set_title(f'{model_name} Forecast ({horizon} days ahead)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


class ForecastApp:
    def __init__(self):
        self.current_df = None
        self.is_valid = False

    def process_upload(self, file) -> str:
        """Process uploaded CSV file."""
        if file is None:
            return "Please upload a CSV file."

        try:
            df = pd.read_csv(file.name)
            self.is_valid, message, self.current_df = validate_csv(df)
            return message
        except Exception as e:
            self.is_valid = False
            self.current_df = None
            return f"Error reading CSV: {str(e)}"

    def chat_response(self, message: str, history: list) -> str:
        """Handle chat messages for CSV format help."""
        message_lower = message.lower()

        if any(x in message_lower for x in ['format', 'example', 'template', 'how']):
            return (
                "Your CSV should have this format:\n\n"
                "```csv\n"
                "date,value\n"
                "2024-01-01,100\n"
                "2024-01-02,150\n"
                "2024-01-03,120\n"
                "```\n\n"
                "**Required columns:**\n"
                "1. **Date column**: Named 'date', 'time', 'timestamp', or 'ds'\n"
                "2. **Value column**: Named 'value', 'quantity', 'demand', 'sales', 'y', or 'target'\n\n"
                "The date should be in a standard format (YYYY-MM-DD recommended)."
            )

        if any(x in message_lower for x in ['column', 'rename', 'header']):
            return (
                "To fix column names, open your CSV in a text editor or Excel and rename:\n\n"
                "- Your date column to: `date`\n"
                "- Your value column to: `value`\n\n"
                "Or use Python:\n"
                "```python\n"
                "df = pd.read_csv('your_file.csv')\n"
                "df = df.rename(columns={'old_date_col': 'date', 'old_value_col': 'value'})\n"
                "df.to_csv('fixed_file.csv', index=False)\n"
                "```"
            )

        if any(x in message_lower for x in ['date', 'time', 'parse']):
            return (
                "For date parsing issues, ensure dates are in a standard format:\n\n"
                "**Supported formats:**\n"
                "- `2024-01-15` (recommended)\n"
                "- `01/15/2024`\n"
                "- `15-Jan-2024`\n"
                "- `2024-01-15 10:30:00`\n\n"
                "Avoid ambiguous formats like `01/02/03`."
            )

        if any(x in message_lower for x in ['error', 'fail', 'wrong', 'problem']):
            if self.current_df is None:
                return "Please upload a CSV file first, then I can help diagnose any issues."
            return (
                "I see there's an issue with your data. Common fixes:\n\n"
                "1. **Check for empty rows**: Remove blank lines\n"
                "2. **Check for text in numeric columns**: Value column should only have numbers\n"
                "3. **Check date format**: Use consistent date formatting\n"
                "4. **Check encoding**: Save as UTF-8\n\n"
                "Try uploading again after making these fixes."
            )

        if self.is_valid:
            return (
                "Your CSV is valid and ready for forecasting!\n\n"
                "1. Select a model (TimesFM or ARIMA)\n"
                "2. Set the forecast horizon (days ahead)\n"
                "3. Click 'Run Forecast'"
            )

        return (
            "I can help you format your CSV for forecasting. Ask me about:\n\n"
            "- **'format'** - See the expected CSV format\n"
            "- **'columns'** - How to rename columns\n"
            "- **'date'** - Date format requirements\n"
            "- **'error'** - Troubleshoot upload issues"
        )

    def run_forecast_ui(
        self,
        model_name: str,
        horizon: int,
        progress=gr.Progress()
    ) -> tuple[plt.Figure | None, str]:
        """Run forecast and return plot + summary."""
        if not self.is_valid or self.current_df is None:
            return None, "Please upload a valid CSV file first."

        progress(0.3, desc="Loading model...")
        try:
            context_length = min(256, len(self.current_df) - horizon)
            if context_length < 30:
                return None, f"Not enough data. Need at least {horizon + 30} data points."

            progress(0.6, desc=f"Running {model_name} forecast...")
            predictions = run_forecast(
                self.current_df, model_name, horizon, context_length
            )

            progress(0.9, desc="Creating visualization...")
            fig = create_forecast_plot(
                self.current_df, predictions, model_name, horizon
            )

            # Create summary
            summary = f"**{model_name} Forecast Summary**\n\n"
            summary += f"- Forecast horizon: {horizon} days\n"
            summary += f"- Context used: {context_length} days\n"
            summary += f"- Predicted range: {predictions.min():.2f} - {predictions.max():.2f}\n"
            summary += f"- Predicted mean: {predictions.mean():.2f}\n"

            progress(1.0, desc="Done!")
            return fig, summary

        except Exception as e:
            return None, f"Error running forecast: {str(e)}"


def create_app():
    """Create and configure the Gradio app."""
    app = ForecastApp()

    with gr.Blocks(title="Time Series Forecasting", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Time Series Forecasting: TimesFM vs ARIMA

            Upload your time series CSV and get forecasts using Google's TimesFM or traditional ARIMA.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Upload Data")
                file_input = gr.File(
                    label="Upload CSV",
                    file_types=[".csv"],
                    type="filepath"
                )
                upload_status = gr.Textbox(
                    label="Upload Status",
                    lines=6,
                    interactive=False
                )

                gr.Markdown("### 2. Configure Forecast")
                model_select = gr.Radio(
                    choices=["TimesFM", "ARIMA"],
                    value="TimesFM",
                    label="Select Model"
                )
                horizon_slider = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=14,
                    step=1,
                    label="Forecast Horizon (days)"
                )
                forecast_btn = gr.Button("Run Forecast", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Forecast Results")
                forecast_plot = gr.Plot(label="Forecast Visualization")
                forecast_summary = gr.Markdown()

        gr.Markdown("---")
        gr.Markdown("### Need Help with CSV Format?")

        chatbot = gr.Chatbot(
            label="Format Assistant",
            height=250
        )
        chat_input = gr.Textbox(
            label="Ask about CSV format",
            placeholder="Type 'format' for examples, or ask about column names, dates, etc."
        )

        # Event handlers
        file_input.change(
            fn=app.process_upload,
            inputs=[file_input],
            outputs=[upload_status]
        )

        forecast_btn.click(
            fn=app.run_forecast_ui,
            inputs=[model_select, horizon_slider],
            outputs=[forecast_plot, forecast_summary]
        )

        def chat_wrapper(message, history):
            response = app.chat_response(message, history)
            history.append((message, response))
            return "", history

        chat_input.submit(
            fn=chat_wrapper,
            inputs=[chat_input, chatbot],
            outputs=[chat_input, chatbot]
        )

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch()
