# Sales Forecasting

This project predicts next-month sales using the Rossmann Store Sales dataset. It is designed as a portfolio-ready forecasting project that is easy to run, easy to explain, and comfortable to demo for recruiters, HR teams, hiring managers, or non-technical stakeholders.

## Repository Summary

- End-to-end next-month sales forecasting workflow using the Rossmann Store Sales dataset
- Two forecasting models compared with time-based validation: Prophet and ARIMA
- Forecast exports, evaluation metrics, business summary, and presentation-ready charts
- Local walkthrough and Google Colab notebook included for demo-friendly delivery

## Business Goal

Forecast the next month's sales so the business can plan inventory, staffing, promotion timing, and monthly revenue targets more confidently.

## Why This Project Matters

Sales forecasting is one of the most practical analytics use cases in business. Companies regularly need a forward-looking estimate to support:

- inventory planning
- workforce scheduling
- campaign timing
- budgeting and revenue planning
- performance monitoring against targets

That makes this project a strong portfolio example because the use case is immediately understandable and commercially relevant.

## Skills Demonstrated

- Time Series Analysis
- Forecasting
- Trend Analysis
- Model Validation
- Business Interpretation
- Presentation-Ready Reporting

## Tools Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Prophet
- ARIMA
- KaggleHub
- Joblib

## Dataset

This project uses the Kaggle dataset:

`pratyushakar/rossmann-store-sales`

The dataset is downloaded automatically in the pipeline with:

```python
import kagglehub

path = kagglehub.dataset_download("pratyushakar/rossmann-store-sales")
print("Path to dataset files:", path)
```

Main files used in this project:

- `train.csv`
- `store.csv`

## Project Outcome

The project:

1. Downloads and stages the raw dataset automatically.
2. Aggregates store-level rows into one daily sales time series.
3. Compares two forecasting approaches:
   - Prophet
   - ARIMA
4. Uses a holdout validation window to choose the stronger model.
5. Retrains on the full history.
6. Predicts sales for the next calendar month.
7. Exports charts, metrics, forecast files, and a business summary for presentation.

## Current Validated Result

From the latest successful local run:

- Best model by validation RMSE: `Prophet`
- Validation window: last `42 days`
- Forecast month: `2015-08`
- Forecast total sales: `200,949,869`
- Forecast change vs previous month: `-5.36%`

Latest validation metrics:

| Model | RMSE | MAE | MAPE | SMAPE |
|---|---:|---:|---:|---:|
| Prophet | 1,377,301 | 1,105,197 | 23.44% | 27.79% |
| ARIMA | 1,855,312 | 1,285,769 | 17.45% | 19.85% |

Interpretation:

- Prophet is selected because it produces the lowest RMSE on the holdout period.
- ARIMA remains a useful baseline and performs competitively on percentage-based error metrics.
- This gives you a stronger demo story because the final forecast comes from a measured model comparison rather than a single-model assumption.

Selection note:

- This project uses `RMSE` as the primary model selection metric.
- That is why `Prophet` is chosen as the final model even though `ARIMA` is lower on `MAPE` and `SMAPE`.
- The decision rule is explicit and documented, which makes the forecast selection easier to defend during demos and interviews.

## Project Structure

```text
Sales Forecasting/
|-- data/
|   |-- raw/
|   |-- processed/
|-- forecasts/
|-- models/
|-- notebooks/
|   |-- sales_forecasting_walkthrough.py
|   |-- sales_forecasting_colab.ipynb
|-- reports/
|   |-- figures/
|-- src/
|   |-- sales_forecasting_pipeline.py
|-- run_pipeline.py
|-- requirements.txt
|-- README.md
```

## Forecasting Approach

### 1. Data Preparation

The project merges the Rossmann training data with store metadata, then aggregates all store sales by date. This creates a clean daily business KPI that is easy to forecast and easy to explain.

The processed daily table includes:

- total daily sales
- total daily customers
- open store count
- average promo share
- school holiday share
- state holiday share
- 7-day rolling sales average
- 30-day rolling sales average

### 2. Validation Strategy

The latest 42 days are held out as a validation window. This is important because a forecasting model should be tested on future-like data instead of random splits.

### 3. Models Compared

#### Prophet

Prophet is useful because it handles:

- trend changes
- weekly seasonality
- yearly seasonality
- business-facing forecasting workflows

#### ARIMA

ARIMA is included as a classic time-series baseline. In this project it is configured with weekly seasonal behavior so it can capture recurring short-term patterns in daily sales.

### 4. Model Selection

The best model is selected using the lowest validation RMSE.

Additional metrics are also exported:

- RMSE
- MAE
- MAPE
- SMAPE

## Key Deliverables

After running the project, these outputs are generated automatically:

### Processed data

- `data/processed/daily_sales.csv`
- `data/processed/monthly_sales.csv`

### Validation and reporting

- `reports/metrics_summary.csv`
- `reports/business_summary.md`
- `reports/project_summary.json`

### Forecast outputs

- `forecasts/next_month_forecast_comparison.csv`
- `forecasts/best_model_next_month_forecast.csv`
- `forecasts/weekly_forecast_summary.csv`

### Charts

- `reports/figures/daily_sales_trend.png`
- `reports/figures/monthly_sales_trend.png`
- `reports/figures/validation_forecast_comparison.png`
- `reports/figures/metrics_comparison.png`
- `reports/figures/next_month_forecast.png`

### Saved models

- `models/prophet_model.joblib`
- `models/arima_model.joblib`

## How to Run Locally

### 1. Open the project folder

```powershell
cd Sales-Forecasting
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the full pipeline

```powershell
python run_pipeline.py
```

This command will:

- download the dataset
- build the daily time series
- train and compare Prophet and ARIMA
- create the next-month forecast
- export charts and summary files

## Quick Files to Open

If you want a fast portfolio walkthrough, start with these files:

1. `README.md`
2. `reports/figures/monthly_sales_trend.png`
3. `reports/metrics_summary.csv`
4. `forecasts/best_model_next_month_forecast.csv`
5. `reports/business_summary.md`

## Notebook-Style Walkthrough for Demo Use

The easiest file to present during a demo is:

- `notebooks/sales_forecasting_walkthrough.py`

This file uses the `# %%` format, which means it behaves like a notebook in VS Code.

### How to use it in VS Code

1. Open `notebooks/sales_forecasting_walkthrough.py`
2. Make sure the Python and Jupyter extensions are installed
3. Select the same Python environment used for the project
4. Click `Run Cell` step by step

### Why this is useful for a portfolio demo

It lets you explain the work in a simple narrative:

1. download the dataset
2. inspect the data
3. show the historical trend
4. explain the validation split
5. compare Prophet vs ARIMA
6. show the final next-month forecast
7. close with business recommendations

This makes the project much easier for non-technical viewers to follow.

## Google Colab Version

A Colab-ready notebook is included:

- `notebooks/sales_forecasting_colab.ipynb`

### How to use it in Google Colab

Option 1:

1. Upload the full project folder to Google Drive or GitHub first
2. Open `notebooks/sales_forecasting_colab.ipynb` in Colab
3. Run the install cell
4. Make sure the `src/` folder is available in the same working directory
5. Run the pipeline cells

Option 2:

1. Open Colab
2. Upload the notebook file
3. Upload the `src/` folder and any required project files
4. Run the notebook cells from top to bottom

### Colab note

Because the notebook imports the pipeline from `src/sales_forecasting_pipeline.py`, the `src/` folder must be present in the runtime. This keeps the Colab version aligned with the local project instead of duplicating logic.
The notebook now also supports a self-bootstrapping flow: if the project files are not present in the runtime, it will clone this repository and install the required dependencies automatically before running.

## Recommended Demo Flow for HR or Recruiters

If you want the project to feel smooth during a live demo, use this order:

1. Start with the business problem: predicting next-month sales.
2. Show the historical daily and monthly trend charts.
3. Explain that you compared two forecasting approaches instead of relying on only one model.
4. Open `reports/metrics_summary.csv` and explain how the best model was selected.
5. Open `forecasts/best_model_next_month_forecast.csv` and highlight the forecast total.
6. Open `reports/business_summary.md` to translate the numbers into business impact.

This flow helps even a non-technical audience understand the value quickly.

## What Makes This Portfolio Project Strong

- It solves a common business problem.
- It uses real forecasting methods instead of only descriptive analysis.
- It compares multiple models.
- It uses a proper time-based validation split.
- It exports clear outputs for presentation.
- It includes both local and Colab-friendly workflows.
- It is structured to be easy to explain during interviews.

## Technical Notes

- The project uses daily aggregated sales across all stores.
- Prophet is configured with weekly and yearly seasonality.
- ARIMA is configured with weekly seasonal behavior.
- Negative forecasts are clipped to zero before export.
- The final next-month forecast is produced after retraining the models on the full historical dataset.

## Possible Future Improvements

- add store-level forecasting by store segment
- include holiday or promo regressors for richer forecasting
- add cross-validation with multiple rolling windows
- build a Streamlit dashboard for interactive demo use
- compare with XGBoost or LightGBM on lag-based features

## Best Way to Present This Project

If your goal is to impress recruiters or hiring managers, focus on these three messages:

1. You understand a real business use case.
2. You know how to validate a forecasting model correctly.
3. You can translate technical output into business decisions.

That combination usually makes a project much stronger than just showing code.
