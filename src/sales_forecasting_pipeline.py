from __future__ import annotations

import json
import shutil
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FORECASTS_DIR = PROJECT_ROOT / "forecasts"

DATASET_SLUG = "pratyushakar/rossmann-store-sales"
DATASET_FILES = ["train.csv", "store.csv", "test.csv"]
VALIDATION_DAYS = 42
FORECAST_DAYS = 31


@dataclass
class ForecastResult:
    name: str
    validation_forecast: pd.DataFrame
    future_forecast: pd.DataFrame
    metrics: dict[str, float]


def ensure_directories() -> None:
    """Create the project directory structure if it does not already exist."""
    for folder in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, FORECASTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def download_dataset() -> dict[str, Path]:
    """Download the Rossmann dataset and copy the required CSV files locally."""
    dataset_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    copied_files: dict[str, Path] = {}

    for file_name in DATASET_FILES:
        source_path = dataset_path / file_name
        if not source_path.exists():
            matches = list(dataset_path.rglob(file_name))
            source_path = matches[0] if matches else source_path
        if not source_path.exists():
            raise FileNotFoundError(f"Expected dataset file was not found: {source_path}")

        destination = DATA_RAW_DIR / file_name
        shutil.copy2(source_path, destination)
        copied_files[file_name] = destination

    return copied_files


def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and store datasets from the local raw-data directory."""
    train_df = pd.read_csv(DATA_RAW_DIR / "train.csv", parse_dates=["Date"], low_memory=False)
    store_df = pd.read_csv(DATA_RAW_DIR / "store.csv", low_memory=False)
    return train_df, store_df


def prepare_daily_sales(train_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate store-level records into a daily sales time series."""
    merged_df = train_df.merge(store_df, on="Store", how="left")
    merged_df = merged_df.sort_values("Date").copy()

    merged_df["Promo"] = merged_df["Promo"].fillna(0)
    merged_df["SchoolHoliday"] = merged_df["SchoolHoliday"].fillna(0)
    merged_df["Open"] = merged_df["Open"].fillna(1)
    merged_df["StateHolidayFlag"] = (merged_df["StateHoliday"].astype(str) != "0").astype(int)

    daily_df = (
        merged_df.groupby("Date", as_index=False)
        .agg(
            sales=("Sales", "sum"),
            customers=("Customers", "sum"),
            open_store_count=("Open", "sum"),
            promo_store_share=("Promo", "mean"),
            school_holiday_share=("SchoolHoliday", "mean"),
            state_holiday_share=("StateHolidayFlag", "mean"),
        )
        .rename(columns={"Date": "ds", "sales": "y"})
    )

    daily_df["day_of_week"] = daily_df["ds"].dt.day_name()
    daily_df["month_name"] = daily_df["ds"].dt.month_name()
    daily_df["week_of_year"] = daily_df["ds"].dt.isocalendar().week.astype(int)
    daily_df["rolling_7d_sales"] = daily_df["y"].rolling(window=7, min_periods=1).mean()
    daily_df["rolling_30d_sales"] = daily_df["y"].rolling(window=30, min_periods=1).mean()

    daily_df.to_csv(DATA_PROCESSED_DIR / "daily_sales.csv", index=False)

    monthly_df = (
        daily_df.assign(month=daily_df["ds"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(monthly_sales=("y", "sum"), average_daily_sales=("y", "mean"))
    )
    monthly_df.to_csv(DATA_PROCESSED_DIR / "monthly_sales.csv", index=False)

    return daily_df


def split_train_validation(daily_df: pd.DataFrame, validation_days: int = VALIDATION_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the daily time series into train and validation windows."""
    if len(daily_df) <= validation_days:
        raise ValueError("The dataset is too short for the requested validation window.")

    train_df = daily_df.iloc[:-validation_days].copy()
    validation_df = daily_df.iloc[-validation_days:].copy()
    return train_df, validation_df


def smape(actual: pd.Series, forecast: pd.Series) -> float:
    """Calculate symmetric mean absolute percentage error."""
    denominator = (np.abs(actual) + np.abs(forecast)).replace(0, np.nan)
    values = 2 * np.abs(forecast - actual) / denominator
    return float(np.nanmean(values) * 100)


def evaluate_forecast(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    """Compute evaluation metrics for a forecast."""
    actual = actual.reset_index(drop=True)
    forecast = forecast.reset_index(drop=True).clip(lower=0)

    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    mape = np.nanmean(np.abs((actual - forecast) / actual.replace(0, np.nan))) * 100

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "smape": float(smape(actual, forecast)),
    }


def fit_prophet_model(train_df: pd.DataFrame) -> Prophet:
    """Train a Prophet model on the daily sales series."""
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
        seasonality_mode="multiplicative",
    )
    prophet_model.fit(train_df[["ds", "y"]])
    return prophet_model


def forecast_with_prophet(model: Prophet, periods: int) -> pd.DataFrame:
    """Generate daily forecasts with Prophet."""
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    return forecast


def fit_arima_model(train_df: pd.DataFrame):
    """Train an ARIMA-based model with weekly seasonality."""
    log_series = np.log1p(train_df["y"])
    arima_model = ARIMA(
        log_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return arima_model.fit()


def forecast_with_arima(model_fit, train_df: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Generate future predictions from the fitted ARIMA model."""
    forecast_values = pd.Series(np.expm1(model_fit.forecast(steps=periods))).clip(lower=0).reset_index(drop=True)
    future_dates = pd.date_range(train_df["ds"].max() + pd.Timedelta(days=1), periods=periods, freq="D")
    return pd.DataFrame({"ds": future_dates, "yhat": forecast_values})


def run_validation_models(train_df: pd.DataFrame, validation_df: pd.DataFrame) -> list[ForecastResult]:
    """Train both forecasting models, evaluate them, and return their validation results."""
    prophet_model = fit_prophet_model(train_df)
    prophet_forecast = forecast_with_prophet(prophet_model, periods=len(validation_df)).tail(len(validation_df)).reset_index(drop=True)
    prophet_metrics = evaluate_forecast(validation_df["y"], prophet_forecast["yhat"])

    arima_fit = fit_arima_model(train_df)
    arima_forecast = forecast_with_arima(arima_fit, train_df, periods=len(validation_df))
    arima_metrics = evaluate_forecast(validation_df["y"], arima_forecast["yhat"])

    return [
        ForecastResult(
            name="Prophet",
            validation_forecast=validation_df[["ds", "y"]].reset_index(drop=True).assign(
                yhat=prophet_forecast["yhat"],
                yhat_lower=prophet_forecast["yhat_lower"],
                yhat_upper=prophet_forecast["yhat_upper"],
            ),
            future_forecast=pd.DataFrame(),
            metrics=prophet_metrics,
        ),
        ForecastResult(
            name="ARIMA",
            validation_forecast=validation_df[["ds", "y"]].reset_index(drop=True).assign(
                yhat=arima_forecast["yhat"],
            ),
            future_forecast=pd.DataFrame(),
            metrics=arima_metrics,
        ),
    ]


def save_metrics(results: list[ForecastResult]) -> pd.DataFrame:
    """Save the validation metrics comparison to disk."""
    metrics_df = pd.DataFrame(
        [{"model": result.name, **result.metrics} for result in results]
    ).sort_values("rmse", ascending=True)
    metrics_df.to_csv(REPORTS_DIR / "metrics_summary.csv", index=False)
    return metrics_df


def retrain_and_forecast(daily_df: pd.DataFrame, forecast_days: int = FORECAST_DAYS) -> list[ForecastResult]:
    """Retrain the models on the full dataset and forecast the next month."""
    prophet_model = fit_prophet_model(daily_df)
    prophet_future = forecast_with_prophet(prophet_model, periods=forecast_days).tail(forecast_days).reset_index(drop=True)
    prophet_future = prophet_future.rename(columns={"yhat": "forecast_sales"})
    joblib.dump(prophet_model, MODELS_DIR / "prophet_model.joblib")

    arima_fit = fit_arima_model(daily_df)
    arima_future = forecast_with_arima(arima_fit, daily_df, periods=forecast_days).rename(columns={"yhat": "forecast_sales"})
    joblib.dump(arima_fit, MODELS_DIR / "arima_model.joblib")

    return [
        ForecastResult(name="Prophet", validation_forecast=pd.DataFrame(), future_forecast=prophet_future, metrics={}),
        ForecastResult(name="ARIMA", validation_forecast=pd.DataFrame(), future_forecast=arima_future, metrics={}),
    ]


def save_forecast_outputs(results: list[ForecastResult], metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Save the future forecasts and identify the best model."""
    best_model_name = metrics_df.iloc[0]["model"]
    output_frames = []

    for result in results:
        frame = result.future_forecast.copy()
        frame["model"] = result.name
        output_frames.append(frame)

    comparison_df = pd.concat(output_frames, ignore_index=True)
    comparison_df.to_csv(FORECASTS_DIR / "next_month_forecast_comparison.csv", index=False)

    best_forecast_df = comparison_df.loc[comparison_df["model"] == best_model_name].copy()
    best_forecast_df["forecast_sales"] = best_forecast_df["forecast_sales"].clip(lower=0)
    best_forecast_df["forecast_month"] = best_forecast_df["ds"].dt.to_period("M").astype(str)
    best_forecast_df.to_csv(FORECASTS_DIR / "best_model_next_month_forecast.csv", index=False)

    summary_df = (
        best_forecast_df.assign(week=lambda df: df["ds"].dt.isocalendar().week.astype(int))
        .groupby("week", as_index=False)["forecast_sales"]
        .sum()
        .rename(columns={"forecast_sales": "weekly_forecast_sales"})
    )
    summary_df.to_csv(FORECASTS_DIR / "weekly_forecast_summary.csv", index=False)

    return best_forecast_df, best_model_name


def save_visualizations(daily_df: pd.DataFrame, train_df: pd.DataFrame, validation_df: pd.DataFrame, validation_results: list[ForecastResult], best_forecast_df: pd.DataFrame, best_model_name: str) -> None:
    """Create presentation-ready charts for the README and demo sessions."""
    plt.figure(figsize=(14, 5))
    plt.plot(daily_df["ds"], daily_df["y"], color="#1f77b4", linewidth=1.5)
    plt.title("Daily Rossmann Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Total Daily Sales")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "daily_sales_trend.png", dpi=180)
    plt.close()

    monthly_sales = (
        daily_df.assign(month=daily_df["ds"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["y"]
        .sum()
        .rename(columns={"y": "monthly_sales"})
    )
    plt.figure(figsize=(14, 5))
    plt.plot(monthly_sales["month"], monthly_sales["monthly_sales"], color="#2ca02c", linewidth=2)
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Monthly Sales")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monthly_sales_trend.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(train_df["ds"], train_df["y"], label="Train", color="#4c78a8", linewidth=1.5)
    plt.plot(validation_df["ds"], validation_df["y"], label="Validation Actual", color="#222222", linewidth=2)

    for result in validation_results:
        plt.plot(result.validation_forecast["ds"], result.validation_forecast["yhat"], label=f"{result.name} Forecast", linewidth=2)

    plt.title("Validation Window Forecast Comparison")
    plt.xlabel("Date")
    plt.ylabel("Daily Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "validation_forecast_comparison.png", dpi=180)
    plt.close()

    metrics_df = pd.DataFrame([{"model": item.name, **item.metrics} for item in validation_results])
    metrics_long = metrics_df.melt(id_vars="model", var_name="metric", value_name="score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=metrics_long, x="metric", y="score", hue="model", palette="Set2")
    plt.title("Forecasting Metrics Comparison")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "metrics_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(best_forecast_df["ds"], best_forecast_df["forecast_sales"], color="#d62728", linewidth=2.5)
    plt.title(f"Next-Month Forecast from {best_model_name}")
    plt.xlabel("Date")
    plt.ylabel("Forecast Sales")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "next_month_forecast.png", dpi=180)
    plt.close()


def write_business_summary(daily_df: pd.DataFrame, metrics_df: pd.DataFrame, best_forecast_df: pd.DataFrame, best_model_name: str) -> None:
    """Write an easy-to-present business summary in Markdown."""
    last_actual_month = daily_df["ds"].max().to_period("M").to_timestamp()
    next_forecast_month = best_forecast_df["ds"].min().to_period("M").to_timestamp()

    recent_month_sales = (
        daily_df.loc[daily_df["ds"].dt.to_period("M").dt.to_timestamp() == last_actual_month, "y"].sum()
    )
    forecast_month_sales = best_forecast_df["forecast_sales"].sum()
    change_pct = ((forecast_month_sales - recent_month_sales) / recent_month_sales) * 100 if recent_month_sales else 0.0

    peak_day = best_forecast_df.loc[best_forecast_df["forecast_sales"].idxmax()]
    weakest_day = best_forecast_df.loc[best_forecast_df["forecast_sales"].idxmin()]

    summary_text = f"""# Sales Forecast Business Summary

## Forecast Goal

Predict the total Rossmann daily sales for the next calendar month and provide a business-friendly summary that is easy to present during a demo.

## Best Validation Model

- Model: `{best_model_name}`
- Selection rule: lowest validation `RMSE`

Validation metrics:

{metrics_df.to_markdown(index=False)}

## Next-Month Forecast

- Last actual month in the training data: `{last_actual_month.strftime("%Y-%m")}`
- Forecast month: `{next_forecast_month.strftime("%Y-%m")}`
- Forecast total sales: `{forecast_month_sales:,.0f}`
- Change vs previous month: `{change_pct:.2f}%`
- Strongest forecast day: `{peak_day["ds"].date()}` with `{peak_day["forecast_sales"]:,.0f}` sales
- Weakest forecast day: `{weakest_day["ds"].date()}` with `{weakest_day["forecast_sales"]:,.0f}` sales

## Business Interpretation

1. The project turns raw store-level sales records into a single daily business KPI that management can review quickly.
2. Prophet and ARIMA provide two distinct forecasting approaches, which makes the demo stronger because the final forecast is chosen from a measurable validation comparison.
3. The output files in the `forecasts/` folder can be used directly for presentation, dashboard prototyping, or follow-up analysis.

## Recommended Demo Narrative

1. Show the historical daily and monthly trend charts.
2. Explain that the last 42 days are kept as a validation window.
3. Compare Prophet vs ARIMA using RMSE, MAE, MAPE, and SMAPE.
4. Open `forecasts/best_model_next_month_forecast.csv` and explain the expected total sales for next month.
5. Close with how this helps planning for inventory, staffing, promotions, and budget allocation.
"""

    (REPORTS_DIR / "business_summary.md").write_text(summary_text, encoding="utf-8")


def save_project_metadata(best_model_name: str, metrics_df: pd.DataFrame, best_forecast_df: pd.DataFrame) -> None:
    """Store a compact JSON summary for quick inspection."""
    metadata = {
        "dataset": DATASET_SLUG,
        "best_model": best_model_name,
        "forecast_start": str(best_forecast_df["ds"].min().date()),
        "forecast_end": str(best_forecast_df["ds"].max().date()),
        "forecast_total_sales": float(best_forecast_df["forecast_sales"].sum()),
        "validation_metrics": metrics_df.to_dict(orient="records"),
    }
    (REPORTS_DIR / "project_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_pipeline() -> dict[str, str]:
    """Run the full sales forecasting workflow from download to reporting."""
    ensure_directories()
    copied_files = download_dataset()
    train_df, store_df = load_source_data()
    daily_df = prepare_daily_sales(train_df, store_df)
    train_window, validation_window = split_train_validation(daily_df)
    validation_results = run_validation_models(train_window, validation_window)
    metrics_df = save_metrics(validation_results)
    final_results = retrain_and_forecast(daily_df)
    best_forecast_df, best_model_name = save_forecast_outputs(final_results, metrics_df)
    save_visualizations(daily_df, train_window, validation_window, validation_results, best_forecast_df, best_model_name)
    write_business_summary(daily_df, metrics_df, best_forecast_df, best_model_name)
    save_project_metadata(best_model_name, metrics_df, best_forecast_df)

    print("Sales forecasting pipeline completed successfully.")
    print(f"Best model: {best_model_name}")
    print(f"Forecast file: {FORECASTS_DIR / 'best_model_next_month_forecast.csv'}")

    return {
        "train_csv": str(copied_files["train.csv"]),
        "best_model": best_model_name,
        "forecast_file": str(FORECASTS_DIR / "best_model_next_month_forecast.csv"),
    }
