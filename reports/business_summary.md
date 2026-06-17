# Sales Forecast Business Summary

## Forecast Goal

Predict the total Rossmann daily sales for the next calendar month and provide a business-friendly summary that is easy to present during a demo.

## Best Validation Model

- Model: `Prophet`
- Selection rule: lowest validation `RMSE`

Validation metrics:

| model   |        rmse |         mae |    mape |   smape |
|:--------|------------:|------------:|--------:|--------:|
| Prophet | 1.3773e+06  | 1.1052e+06  | 23.4358 | 27.7932 |
| ARIMA   | 1.85531e+06 | 1.28577e+06 | 17.455  | 19.8463 |

## Next-Month Forecast

- Last actual month in the training data: `2015-07`
- Forecast month: `2015-08`
- Forecast total sales: `200,949,869`
- Change vs previous month: `-5.36%`
- Strongest forecast day: `2015-08-10` with `9,081,337` sales
- Weakest forecast day: `2015-08-23` with `0` sales

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
