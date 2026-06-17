# %%
"""
Notebook-style walkthrough for the Sales Forecasting project.

This file is designed to feel similar to Google Colab when opened in VS Code:
- run cell by cell,
- inspect tables and charts inline,
- and present the project in a step-by-step narrative.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.sales_forecasting_pipeline import (  # noqa: E402
    FORECAST_DAYS,
    FORECASTS_DIR,
    REPORTS_DIR,
    download_dataset,
    ensure_directories,
    fit_arima_model,
    fit_prophet_model,
    forecast_with_arima,
    forecast_with_prophet,
    load_source_data,
    prepare_daily_sales,
    retrain_and_forecast,
    run_validation_models,
    save_forecast_outputs,
    save_metrics,
    split_train_validation,
    write_business_summary,
)


sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


def show_section(title: str, subtitle: str = "") -> None:
    """Display a section heading for easier presentation."""
    text = f"## {title}"
    if subtitle:
        text += f"\n\n{subtitle}"
    display(Markdown(text))


# %%
show_section("1. Dataset Download", "Download the Rossmann dataset and stage the project folders.")
ensure_directories()
downloaded_files = download_dataset()
display(pd.DataFrame({"file_name": list(downloaded_files.keys()), "path": [str(path) for path in downloaded_files.values()]}))


# %%
show_section("2. Load the Source Tables", "Read the training and store metadata tables.")
train_df, store_df = load_source_data()
print("Train shape:", train_df.shape)
print("Store shape:", store_df.shape)
display(train_df.head())
display(store_df.head())


# %%
show_section("3. Build the Daily Time Series", "Aggregate sales across all stores into one daily forecasting series.")
daily_df = prepare_daily_sales(train_df, store_df)
print("Daily series shape:", daily_df.shape)
display(daily_df.head())


# %%
show_section("4. Historical Trend", "Review the historical sales movement before forecasting.")
plt.figure(figsize=(14, 5))
plt.plot(daily_df["ds"], daily_df["y"], color="#1f77b4", linewidth=1.5)
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Daily Sales")
plt.tight_layout()
plt.show()


# %%
show_section("5. Monthly Trend", "The monthly view makes business seasonality easier to explain during a demo.")
monthly_df = (
    daily_df.assign(month=daily_df["ds"].dt.to_period("M").dt.to_timestamp())
    .groupby("month", as_index=False)["y"]
    .sum()
    .rename(columns={"y": "monthly_sales"})
)
display(monthly_df.tail(12))

plt.figure(figsize=(14, 5))
plt.plot(monthly_df["month"], monthly_df["monthly_sales"], color="#2ca02c", linewidth=2)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Monthly Sales")
plt.tight_layout()
plt.show()


# %%
show_section("6. Train and Validation Split", "Reserve the latest 42 days so we can compare forecasting performance fairly.")
train_window, validation_window = split_train_validation(daily_df)
split_summary = pd.DataFrame(
    {
        "dataset": ["Train", "Validation"],
        "rows": [len(train_window), len(validation_window)],
        "date_range": [
            f"{train_window['ds'].min().date()} to {train_window['ds'].max().date()}",
            f"{validation_window['ds'].min().date()} to {validation_window['ds'].max().date()}",
        ],
    }
)
display(split_summary)


# %%
show_section("7. Model Training and Validation", "Train Prophet and ARIMA, then compare them on the holdout window.")
validation_results = run_validation_models(train_window, validation_window)
metrics_df = save_metrics(validation_results)
display(metrics_df)


# %%
show_section("8. Validation Forecast Comparison", "Check how closely each model follows the actual validation trend.")
plt.figure(figsize=(14, 5))
plt.plot(train_window["ds"], train_window["y"], label="Train", color="#4c78a8", linewidth=1.2)
plt.plot(validation_window["ds"], validation_window["y"], label="Validation Actual", color="#111111", linewidth=2.2)
for result in validation_results:
    plt.plot(result.validation_forecast["ds"], result.validation_forecast["yhat"], label=f"{result.name} Forecast", linewidth=2)
plt.title("Validation Forecast Comparison")
plt.xlabel("Date")
plt.ylabel("Daily Sales")
plt.legend()
plt.tight_layout()
plt.show()


# %%
show_section("9. Refit on Full Data", "Use all historical data to forecast the next calendar month.")
full_results = retrain_and_forecast(daily_df, forecast_days=FORECAST_DAYS)
best_forecast_df, best_model_name = save_forecast_outputs(full_results, metrics_df)
print("Best model:", best_model_name)
display(best_forecast_df.head())


# %%
show_section("10. Next-Month Forecast", "This table is the main business deliverable for planning the next month.")
display(best_forecast_df)

plt.figure(figsize=(12, 5))
plt.plot(best_forecast_df["ds"], best_forecast_df["forecast_sales"], color="#d62728", linewidth=2.5)
plt.title(f"Next-Month Forecast from {best_model_name}")
plt.xlabel("Date")
plt.ylabel("Forecast Sales")
plt.tight_layout()
plt.show()


# %%
show_section("11. Exported Deliverables", "Show the output files that are ready for demos and recruiter reviews.")
write_business_summary(daily_df, metrics_df, best_forecast_df, best_model_name)
output_files = pd.DataFrame(
    {
        "file_name": [
            "reports/metrics_summary.csv",
            "reports/business_summary.md",
            "forecasts/next_month_forecast_comparison.csv",
            "forecasts/best_model_next_month_forecast.csv",
        ],
        "description": [
            "Validation metrics for Prophet and ARIMA",
            "Business-focused explanation of the forecast",
            "Forecasts from both models",
            "Selected next-month forecast from the best model",
        ],
    }
)
display(output_files)

print((REPORTS_DIR / "business_summary.md").read_text(encoding="utf-8"))


# %%
show_section("12. Final Summary", "Close the walkthrough with the recommended narrative for your portfolio demo.")
print("Analysis completed.")
print("Best model:", best_model_name)
print("Forecast file:", FORECASTS_DIR / "best_model_next_month_forecast.csv")
display(metrics_df)
