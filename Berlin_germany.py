"""
Berlin_germany.py - Solar and wind data analysis for Berlin, Germany.

Tasks performed:
    1. Data overview and cleaning
        2. Time series visualisation
            3. ARIMA forecasting
                4. SARIMA forecasting

                Data source: NASA POWER (Berlin_Germany.csv, skiprows=12)
                """

from utils import (
    clean_dataset,
    plot_annual_trend,
    plot_monthly_trends,
    plot_correlation_heatmap,
    arima_forecast,
    sarima_forecast,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_FILE = 'Berlin_Germany.csv'
CITY     = 'Berlin, Germany'

PARAMETERS = ['T2M', 'WS50M', 'WS50M_MAX', 'ALLSKY_SFC_SW_DWN']

ARIMA_ORDER         = (5, 1, 0)
SARIMA_ORDER        = (1, 1, 1)
SARIMA_SEASONAL     = (1, 1, 1, 12)
FORECAST_PERIODS    = 10

# ---------------------------------------------------------------------------
# Task 1: Data overview and cleaning
# ---------------------------------------------------------------------------

print("=" * 60)
print(f"Task 1: Data Overview and Cleaning – {CITY}")
print("=" * 60)

data = clean_dataset(CSV_FILE)

# ---------------------------------------------------------------------------
# Task 2: Time series visualisation
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"Task 2: Time Series Visualisation – {CITY}")
print("=" * 60)

# Annual trend for each parameter
for param in PARAMETERS:
        plot_annual_trend(data, param, CITY)

# Monthly breakdown (temperature)
plot_monthly_trends(data[data['PARAMETER'] == 'T2M'], CITY,
                                        value_col='Temperature', ylabel='Temperature (°C)')

# Correlation heatmap
plot_correlation_heatmap(data, CITY)

# ---------------------------------------------------------------------------
# Task 3: ARIMA forecasting
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"Task 3: ARIMA Forecasting – {CITY}")
print("=" * 60)

for param in PARAMETERS:
        arima_forecast(data, param, CITY,
                                          order=ARIMA_ORDER,
                                          forecast_periods=FORECAST_PERIODS)

# ---------------------------------------------------------------------------
# Task 4: SARIMA forecasting
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(f"Task 4: SARIMA Forecasting – {CITY}")
print("=" * 60)

for param in PARAMETERS:
        sarima_forecast(data, param, CITY,
                                            order=SARIMA_ORDER,
                                            seasonal_order=SARIMA_SEASONAL,
                                            forecast_periods=FORECAST_PERIODS)
