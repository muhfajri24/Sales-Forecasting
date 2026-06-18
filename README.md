# Sales Forecasting

This project forecasts next-month sales from Rossmann historical data and turns the forecast into planning-ready business insight.

## What This Project Does

- Downloads and prepares Rossmann sales data
- Compares `Prophet` and `ARIMA` with time-based validation
- Selects the final model using validation performance
- Exports forecast files, metrics, figures, and business summary outputs

## Why It Matters

This project shows how Python can be used to build a practical forecasting workflow for inventory planning, staffing, promotions, and revenue target setting.

## Primary Workflow

Python is the main way to run this project.

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Main entrypoints:

- `run_pipeline.py` for the full end-to-end run
- `src/sales_forecasting_pipeline.py` for the reusable forecasting logic

## Optional Notebook

Notebook files are included only as a secondary option for walkthroughs, recruiter demos, or quick testing.

- `notebooks/sales_forecasting_walkthrough.py`
- `notebooks/sales_forecasting_colab.ipynb`

## Dataset

Source dataset: Kaggle `pratyushakar/rossmann-store-sales`

The pipeline downloads the dataset automatically and stages it in `data/raw/`.

## Tools

- Python
- Pandas
- Matplotlib
- Seaborn
- Prophet
- ARIMA
- KaggleHub
- Joblib

## Project Structure

```text
Sales Forecasting/
|-- data/
|-- forecasts/
|-- models/
|-- notebooks/
|-- reports/
|-- src/
|   `-- sales_forecasting_pipeline.py
|-- run_pipeline.py
|-- requirements.txt
`-- README.md
```
