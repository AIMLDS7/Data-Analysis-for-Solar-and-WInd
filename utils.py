"""
utils.py - Shared utility functions for solar and wind data analysis.

All reusable logic for data loading, cleaning, visualisation, and
time-series forecasting lives here so that city-specific scripts stay
short and consistent.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

MONTH_COLS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                            'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

PARAM_LABELS = {
      'T2M':             'Temperature at 2 Meters (°C)',
      'WS50M':           'Wind Speed at 50 Meters (m/s)',
      'WS50M_MAX':       'Maximum Wind Speed at 50 Meters (m/s)',
      'ALLSKY_SFC_SW_DWN': 'Solar Irradiance (kW-hr/m²/day)',
}

# ---------------------------------------------------------------------------
# Task 1 – Data loading and cleaning
# ---------------------------------------------------------------------------

def load_dataset(filepath, skiprows=12):
      """Load a NASA POWER CSV, skipping the metadata header rows."""
      df = pd.read_csv(filepath, skiprows=skiprows)
      print(f"Loaded '{filepath}': {df.shape[0]} rows, {df.shape[1]} columns.")
      return df


def summarise_missing(df):
      """Print a missing-value report and return the counts."""
      missing = df.isna().sum()
      pct = (missing / len(df)) * 100
      report = pd.DataFrame({'missing_count': missing, 'missing_pct': pct})
      print("\nMissing value report:")
      print(report[report['missing_count'] > 0].to_string() or "  No missing values.")
      return missing


def drop_missing(df):
      """Drop rows with any NaN values and report how many were removed."""
      before = len(df)
      cleaned = df.dropna()
      removed = before - len(cleaned)
      print(f"Dropped {removed} rows with NaN values ({removed/before*100:.1f}%).")
      return cleaned


def identify_outliers_zscore(series, threshold=3):
      """Return a boolean mask where True marks outlier rows (|z| > threshold)."""
      z_scores = np.abs((series - series.mean()) / series.std())
      return z_scores > threshold


def remove_outliers(df, numeric_start_col=2, threshold=3):
      """
          Remove rows that are outliers in *any* numeric column.
              numeric_start_col: column index where numeric data begins (default 2,
                                     skipping YEAR and PARAMETER).
                                         """
      numeric_cols = df.columns[numeric_start_col:]
      outlier_flags = pd.DataFrame(False, index=df.index, columns=numeric_cols)
      for col in numeric_cols:
                outlier_flags[col] = identify_outliers_zscore(df[col], threshold)

      outliers_per_col = outlier_flags.sum()
      print("\nOutliers detected per column:")
      print(outliers_per_col[outliers_per_col > 0].to_string() or "  None.")

    mask = outlier_flags.any(axis=1)
    cleaned = df[~mask]
    print(f"Removed {mask.sum()} outlier rows. {len(cleaned)} rows remaining.")
    return cleaned


def clean_dataset(filepath, skiprows=12):
      """Full pipeline: load → report missing → drop NaN → remove outliers."""
      df = load_dataset(filepath, skiprows=skiprows)
      num_samples, num_features = df.shape
      print(f"Features: {num_features}, Samples: {num_samples}")
      print("\nColumn dtypes:")
      print(df.dtypes.to_string())
      summarise_missing(df)
      df = drop_missing(df)
      df = remove_outliers(df)
      return df

# ---------------------------------------------------------------------------
# Task 2 – Visualisation helpers
# ---------------------------------------------------------------------------

def plot_annual_trend(df, parameter, city, ylabel=None):
      """Line plot of the annual (ANN) average for a given PARAMETER."""
      subset = df[df['PARAMETER'] == parameter]
      if subset.empty:
                print(f"Warning: no data found for PARAMETER='{parameter}'.")
                return
            label = ylabel or PARAM_LABELS.get(parameter, parameter)
    plt.figure(figsize=(14, 8))
    sns.lineplot(x='YEAR', y='ANN', data=subset)
    plt.title(f'Yearly Average {label} – {city}')
    plt.xlabel('Year')
    plt.ylabel(label)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_monthly_trends(df, city, value_col='ANN', ylabel='Value'):
      """Multi-line plot with one line per month across all years."""
    present_months = [c for c in MONTH_COLS if c in df.columns]
    monthly_data = df.melt(
              id_vars=['YEAR'],
              value_vars=present_months,
              var_name='Month',
              value_name=value_col,
    )
    plt.figure(figsize=(16, 10))
    sns.lineplot(x='YEAR', y=value_col, hue='Month',
                                  data=monthly_data, palette='tab10')
    plt.title(f'Monthly Average {ylabel} Across Years – {city}')
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, city):
      """Heatmap of Pearson correlation between numeric columns."""
    numeric_data = df.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title(f'Correlation Matrix of Features – {city}')
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Task 3 – ARIMA forecasting
# ---------------------------------------------------------------------------

def arima_forecast(data, parameter, city, order=(5, 1, 0), forecast_periods=10):
      """
          Fit an ARIMA model on the ANN column for the given PARAMETER and plot
              a forecast.

                  Parameters
                      ----------
                          data : pd.DataFrame  cleaned DataFrame containing PARAMETER and YEAR columns
                              parameter : str      value in the PARAMETER column to model
                                  city : str           used in plot titles
                                      order : tuple        (p, d, q) for ARIMA
                                          forecast_periods : int  number of years to forecast ahead
                                              """
    param_data = data[data['PARAMETER'] == parameter].copy()
    if param_data.empty:
              print(f"Warning: no data for PARAMETER='{parameter}'. Skipping ARIMA.")
              return

    param_data.set_index('YEAR', inplace=True)
    label = PARAM_LABELS.get(parameter, parameter)

    model = ARIMA(param_data['ANN'], order=order)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=forecast_periods)
    forecast_index = np.arange(
              param_data.index[-1] + 1,
              param_data.index[-1] + 1 + forecast_periods,
    )

    plt.figure(figsize=(14, 8))
    plt.plot(param_data.index, param_data['ANN'], label='Historical Data')
    plt.plot(forecast_index, forecast, label='Forecast', linestyle='--', color='orange')
    plt.title(f'ARIMA Forecast – {label} – {city}')
    plt.xlabel('Year')
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"\nARIMA forecast values for {label} at {city}:")
    print(forecast.to_string())
    return forecast

# ---------------------------------------------------------------------------
# Task 4 – SARIMA forecasting
# ---------------------------------------------------------------------------

def sarima_forecast(data, parameter, city,
                                        order=(1, 1, 1),
                                        seasonal_order=(1, 1, 1, 12),
                                        forecast_periods=10):
                                              """
                                                  Fit a SARIMA model on the ANN column for the given PARAMETER and plot
                                                      a forecast with the historical data.

                                                          Parameters
                                                              ----------
                                                                  data : pd.DataFrame  cleaned DataFrame
                                                                      parameter : str      value in the PARAMETER column to model
                                                                          city : str           used in plot titles
                                                                              order : tuple        (p, d, q) for SARIMA non-seasonal component
                                                                                  seasonal_order : tuple  (P, D, Q, s) seasonal component
                                                                                      forecast_periods : int  years to forecast
                                                                                          """
                                              param_data = data[data['PARAMETER'] == parameter].copy()
                                              if param_data.empty:
                                                        print(f"Warning: no data for PARAMETER='{parameter}'. Skipping SARIMA.")
                                                        return

                                              param_data.set_index('YEAR', inplace=True)
                                              label = PARAM_LABELS.get(parameter, parameter)

    model = SARIMAX(param_data['ANN'], order=order,
                                        seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    forecast_result = model_fit.get_forecast(steps=forecast_periods)
    forecast_index = np.arange(
              param_data.index[-1] + 1,
              param_data.index[-1] + 1 + forecast_periods,
    )
    forecast_values = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    combined = pd.concat([
              param_data['ANN'],
              pd.Series(forecast_values.values, index=forecast_index),
    ])

    plt.figure(figsize=(14, 8))
    plt.plot(param_data.index, param_data['ANN'], label='Historical Data')
    plt.plot(forecast_index, forecast_values.values,
                          label=f'{forecast_periods}-Year Forecast', linestyle='--', color='orange')
    plt.fill_between(forecast_index,
                                          conf_int.iloc[:, 0].values,
                                          conf_int.iloc[:, 1].values,
                                          alpha=0.2, color='orange', label='95% CI')
    plt.title(f'SARIMA Forecast – {label} – {city}')
    plt.xlabel('Year')
    plt.ylabel(label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    forecast_df = pd.DataFrame({
              'Year': forecast_index,
              'Forecasted_Value': forecast_values.values,
    })
    print(f"\nSARIMA forecast values for {label} at {city}:")
    print(forecast_df.to_string(index=False))
    return forecast_df
