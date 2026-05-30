<div align="center">
  
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0a1628,50:1a3a5c,100:0d2137&height=200&section=header&text=%F0%9F%8C%9E%20Solar%20%26%20Wind%20Analysis&fontSize=38&fontColor=ffffff&fontAlignY=38&desc=NASA%20POWER%20Climate%20Data%20%7C%20ARIMA%20and%20SARIMA%20Forecasting%20%7C%20Berlin%20vs%20Honolulu&descAlignY=60&descSize=16" width="100%"/> </div>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-ARIMA%2FSARIMA-4C72B0?style=flat-square)](https://www.statsmodels.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-11557C?style=flat-square)](https://matplotlib.org/)
[![NASA POWER](https://img.shields.io/badge/Data-NASA%20POWER-FC3D21?style=flat-square&logo=nasa&logoColor=white)](https://power.larc.nasa.gov/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

> **Multi-decade meteorological analysis and time-series forecasting for two climatically contrasting cities.**
> Analyses NASA POWER reanalysis data (1981–2023) across temperature, wind speed, and solar irradiance using a
> modular Python pipeline — with ARIMA and SARIMA models generating 10-year forecasts with 95% confidence intervals.

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Why These Two Cities?](#-why-these-two-cities)
- [Data Source & Parameters](#-data-source--parameters)
- [Analysis Pipeline](#-analysis-pipeline)
- [Forecasting Models](#-forecasting-models)
- [Key Findings](#-key-findings)
- [Code Architecture](#-code-architecture)
- [API Reference — utils.py](#-api-reference--utilspy)
- [Quick Start](#-quick-start)
- [Technical Decisions](#️-technical-decisions)
- [Limitations & Future Work](#️-limitations--future-work)

---

## 🔭 Overview

This project analyses **multi-decade hourly climate reanalysis data** from NASA's POWER dataset to quantify long-term trends and forecast future climate parameters at two geographically contrasting locations:

- **Berlin, Germany** — continental European climate; cold winters, moderate summers, wind-rich
- **Honolulu, Hawaii, USA** — subtropical Pacific climate; consistent solar irradiance year-round

The analysis covers **4 meteorological parameters** across a ~40-year window, applies statistical time-series models to each, and surfaces cross-city comparisons via a dedicated `compare_cities.py` script — enabling direct renewable energy potential benchmarking between a mid-latitude European and a tropical Pacific site.

---

## 🌍 Why These Two Cities?

The city pairing is deliberate, not arbitrary:

| Dimension | Berlin 🇩🇪 | Honolulu 🇺🇸 |
|---|---|---|
| **Latitude** | 52.5°N | 21.3°N |
| **Climate type** | Cfb (oceanic/continental) | Af (tropical) |
| **Solar potential** | Moderate, highly seasonal | High, consistent year-round |
| **Wind potential** | Moderate–high (offshore influence) | Trade-wind driven, steady |
| **Renewable context** | Germany's Energiewende policy | Hawaii 100% RPS target by 2045 |
| **Temp range** | Wide seasonal swing (~25°C) | Narrow (~5°C variation) |

Comparing these two exposes how climate zone affects the **predictability and bankability** of solar and wind resources — a core challenge in renewable energy project finance.

---

## 📡 Data Source & Parameters

All data sourced from **[NASA POWER](https://power.larc.nasa.gov/)** (Prediction Of Worldwide Energy Resources) — a satellite-derived, globally gridded reanalysis product designed specifically for renewable energy assessment.

**Data characteristics:**
- Temporal resolution: monthly averages (annual summary column `ANN` used for modelling)
- Spatial resolution: 0.5° × 0.625° grid
- Coverage: 1981–2023 (~43 years)
- Format: CSV with 12-row metadata header (auto-skipped by `load_dataset()`)

**Parameters analysed:**

| Parameter | Full Name | Unit | Relevance |
|---|---|---|---|
| `T2M` | Temperature at 2 Meters | °C | Heating/cooling demand; wind turbine performance |
| `WS50M` | Wind Speed at 50 Meters | m/s | Hub-height wind resource (IEC standard reference height) |
| `WS50M_MAX` | Maximum Wind Speed at 50 Meters | m/s | Extreme event analysis; turbine structural loading |
| `ALLSKY_SFC_SW_DWN` | All-Sky Surface Shortwave Downward Irradiance | kW-hr/m²/day | Global Horizontal Irradiance proxy for PV yield |

> `WS50M` at 50m is the IEC 61400-1 standard reference height for small-to-medium wind turbines. Modern utility-scale turbines operate at 80–150m hub heights, so `WS50M` slightly underestimates available wind resource for large installations.

---

## 🔬 Analysis Pipeline

Each city script (`Berlin_germany.py`, `Honolulu_USA.py`) runs the same **5-stage pipeline** via shared utilities in `utils.py`:

---

### Stage 1 — 📥 Data Ingestion & Quality Control

| Function | Description |
|---|---|
| `load_dataset()` | Skip 12-row NASA POWER metadata header |
| `summarise_missing()` | Per-column NaN count + percentage report |
| `drop_missing()` | Drop rows with any NaN; log removal count |
| `remove_outliers()` | Z-score filter \|z\| > 3.0 across all numeric columns |

---

### Stage 2 — 📊 Exploratory Data Analysis

| Function | Description |
|---|---|
| `plot_annual_trend()` | Annual average (ANN) line plots per parameter |
| `plot_monthly_trends()` | 12-line panel: one series per month |
| `plot_correlation_heatmap()` | Pearson r matrix across all features |

---

### Stage 3 — 📈 ARIMA Forecasting

| Function | Description |
|---|---|
| `arima_forecast()` | `order=(5,1,0)` · 10-year horizon · Fits on ANN column per parameter |

---

### Stage 4 — 📉 SARIMA Forecasting

| Function | Description |
|---|---|
| `sarima_forecast()` | `order=(1,1,1)` `seasonal=(1,1,1,12)` · 10-year horizon + 95% confidence bands |

---

### Stage 5 — 🌍 Cross-City Comparison

| Script | Description |
|---|---|
| `compare_cities.py` | Side-by-side Berlin vs Honolulu comparison across all parameters |

---

> **Pipeline Flow:**
> ```
> Data Ingestion → EDA → ARIMA → SARIMA → Cross-City Comparison
>      Stage 1   →  2  →   3   →    4   →       Stage 5
> ```

## 📈 Forecasting Models

### ARIMA — Autoregressive Integrated Moving Average

ARIMA is used as the baseline time-series model. It captures linear temporal structure through three components:

```
ARIMA(p=5, d=1, q=0)
```

| Parameter | Value | Meaning |
|---|---|---|
| `p = 5` | 5 autoregressive lags | Model learns from the past 5 years of annual values |
| `d = 1` | 1st-order differencing | Removes linear trend to achieve stationarity |
| `q = 0` | No moving-average terms | Pure AR structure — appropriate for smooth climate series |

**Why `d=1`?** Climate parameters like temperature and irradiance often exhibit slow upward trends. First differencing removes this non-stationarity before fitting, preventing the model from extrapolating the trend as a random walk.

```python
model = ARIMA(param_data['ANN'], order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)   # 10-year horizon
```

---

### SARIMA — Seasonal ARIMA

SARIMA extends ARIMA with explicit seasonal components, making it better suited to monthly climate data:

```
SARIMA(p=1, d=1, q=1)(P=1, D=1, Q=1, s=12)
```

| Component | Value | Meaning |
|---|---|---|
| `p, d, q` | `1, 1, 1` | Non-seasonal AR(1), differencing, MA(1) |
| `P, D, Q` | `1, 1, 1` | Seasonal AR(1), seasonal differencing, seasonal MA(1) |
| `s = 12` | 12 months | Annual seasonal period |

**Output:** Point forecasts + **95% confidence intervals** computed via `get_forecast().conf_int()`, visualised as shaded bands around the forecast line.

```python
model = SARIMAX(param_data['ANN'], order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit(disp=False)
forecast_result = model_fit.get_forecast(steps=10)
conf_int = forecast_result.conf_int()   # lower and upper 95% CI bounds
```

**Why SARIMA over ARIMA?** Climate data has annual seasonal cycles embedded in monthly resolution. SARIMA with `s=12` captures how Jan–Dec patterns repeat across years — producing narrower, more calibrated confidence intervals than plain ARIMA on annual averages.

---

## 💡 Key Findings

**Solar Irradiance (`ALLSKY_SFC_SW_DWN`):**
- Honolulu receives consistently **2–3× higher** irradiance than Berlin year-round
- Berlin shows strong seasonal variance (winter ~0.5 kWh/m²/day, summer ~5+ kWh/m²/day)
- Honolulu's irradiance is stable within ±15% year-round — far more bankable for PV project finance

**Wind Speed (`WS50M`):**
- Berlin's 50m wind speeds are generally higher on an annual basis
- Honolulu's trade winds produce a more **consistent** speed profile with lower seasonal variance
- `WS50M_MAX` shows Berlin has more extreme wind events — relevant for turbine structural design

**Temperature (`T2M`):**
- Berlin's wider temperature range creates stronger heating/cooling demand seasonality
- This seasonality is visible in SARIMA's wider confidence intervals for Berlin vs Honolulu

**Correlation analysis:**
- Strong positive correlation between `T2M` and `ALLSKY_SFC_SW_DWN` at both locations — warmer months are sunnier months
- Negative correlation between `T2M` and `WS50M` at Berlin — wind speeds tend to be higher in cooler seasons

---

## 🏗 Code Architecture

The project uses a **shared utilities pattern** — all reusable logic is centralised in `utils.py`, keeping city-specific scripts thin and consistent. Adding a new city requires writing a single script of ~20 lines.

```
📦 Data-Analysis-for-Solar-and-WInd/
│
├── 📄 utils.py                    ← Core library (269 lines)
│                                     Data loading, cleaning, EDA, ARIMA, SARIMA
│
├── 📄 Berlin_germany.py           ← Runs full pipeline for Berlin
├── 📄 Honolulu_USA.py             ← Runs full pipeline for Honolulu
├── 📄 compare_cities.py           ← Cross-city comparison and benchmark plots
│
├── 📊 Berlin_Germany.csv          ← Raw NASA POWER data (Berlin, ~43 years)
├── 📊 Honolulu_USA.csv            ← Raw NASA POWER data (Honolulu, ~43 years)
├── 📊 bdata.csv                   ← Cleaned/processed data for Berlin
├── 📊 hdata.csv                   ← Cleaned/processed data for Honolulu
│
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 LICENSE
```

**Design rationale:** City scripts import from `utils.py` rather than copy-pasting logic. Any fix to `remove_outliers()` or `sarima_forecast()` propagates to all city analyses automatically — a basic software engineering principle often skipped in data science projects.

---

## 📚 API Reference — `utils.py`

### Data Loading & Cleaning

| Function | Signature | Description |
|---|---|---|
| `load_dataset` | `(filepath, skiprows=12)` | Load NASA POWER CSV, skipping 12-row metadata header |
| `clean_dataset` | `(filepath, skiprows=12)` | Full pipeline: load → missing report → drop NaN → remove outliers |
| `summarise_missing` | `(df)` | Per-column NaN count + percentage report, returns `pd.Series` |
| `drop_missing` | `(df)` | Drop rows with any NaN; logs count removed |
| `remove_outliers` | `(df, numeric_start_col=2, threshold=3)` | Z-score filter `\|z\| > 3.0` across all numeric columns simultaneously |
| `identify_outliers_zscore` | `(series, threshold=3)` | Returns boolean mask where `True` = outlier row |

### Visualisation

| Function | Signature | Description |
|---|---|---|
| `plot_annual_trend` | `(df, parameter, city, ylabel=None)` | Line plot of annual average (`ANN`) per parameter |
| `plot_monthly_trends` | `(df, city, value_col, ylabel)` | 12-series chart via `df.melt()`: one line per calendar month |
| `plot_correlation_heatmap` | `(df, city)` | Pearson r heatmap, annotated, `coolwarm` palette |

### Forecasting

| Function | Signature | Returns |
|---|---|---|
| `arima_forecast` | `(data, parameter, city, order=(5,1,0), forecast_periods=10)` | `pd.Series` of forecast values |
| `sarima_forecast` | `(data, parameter, city, order=(1,1,1), seasonal_order=(1,1,1,12), forecast_periods=10)` | `pd.DataFrame` with `Year` and `Forecasted_Value` columns |

---

## 🚀 Quick Start

### 1 — Clone

```bash
git clone https://github.com/AIMLDS7/Data-Analysis-for-Solar-and-WInd.git
cd Data-Analysis-for-Solar-and-WInd
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
```

### 3 — Run city analyses

```bash
# Full pipeline for Berlin
python Berlin_germany.py

# Full pipeline for Honolulu
python Honolulu_USA.py

# Cross-city comparison plots
python compare_cities.py
```

### 4 — Use `utils.py` directly in your own script

```python
from utils import clean_dataset, arima_forecast, sarima_forecast

# Load and clean
df = clean_dataset('Berlin_Germany.csv')

# ARIMA: forecast temperature 10 years ahead
arima_forecast(df, parameter='T2M', city='Berlin')

# SARIMA: forecast solar irradiance with confidence intervals
sarima_forecast(df, parameter='ALLSKY_SFC_SW_DWN', city='Berlin')
```

---

## ⚙️ Technical Decisions

**Why Z-score threshold of 3.0?**
The empirical rule states ~99.7% of normally distributed data falls within 3 standard deviations of the mean. Climate parameters are approximately normally distributed annually, so `|z| > 3` flags genuine anomalies (measurement errors, data artefacts) rather than natural extremes. A threshold of 2.0 would remove too many valid high/low years.

**Why `skiprows=12` for NASA POWER CSVs?**
NASA POWER exports include a 12-line metadata header (project info, coordinates, temporal coverage, units). `skiprows=12` drops exactly this header to land on the column row — without it, `read_csv()` misparses the metadata as data.

**Why ARIMA `(5,1,0)` specifically?**
`p=5` captures up to 5 years of autocorrelation in annual climate data — appropriate for slow-moving variables that exhibit multi-year persistence (e.g. ENSO cycles affect temperature for 2–5 years). `d=1` removes the linear warming trend. `q=0` was selected because residuals showed no significant moving-average structure after AR and differencing.

**Why annual averages (`ANN`) for modelling?**
Working on `ANN` targets the slower inter-annual trend — the most policy-relevant signal for renewable resource assessment. Monthly-resolution modelling would require more sophisticated seasonal decomposition before ARIMA could be applied cleanly; SARIMA with `s=12` addresses this for the seasonal component.

**Why a shared `utils.py`?**
DRY principle (Don't Repeat Yourself). Any change to the cleaning pipeline or model order applies to all cities in one edit. It also makes the code unit-testable independently of city-specific data files.

---

## ⚠️ Limitations & Future Work

**Current limitations:**

| Limitation | Impact |
|---|---|
| Annual `ANN` averages used for ARIMA | Masks intra-year variation; monthly-resolution models would be more granular |
| Fixed ARIMA/SARIMA orders | `(5,1,0)` and `(1,1,1)(1,1,1,12)` not auto-tuned per parameter |
| No stationarity testing (ADF/KPSS) | Differencing order `d=1` assumed rather than statistically verified |
| No backtesting / walk-forward validation | Forecast accuracy on held-out years not quantified |
| 50m wind speed vs modern turbine heights | `WS50M` understates resource for 100m+ hub-height turbines |

**Planned improvements:**

- [ ] ADF / KPSS stationarity test to validate differencing order before model fit
- [ ] AIC/BIC grid search over `(p,d,q)` to auto-select best ARIMA order per parameter
- [ ] Walk-forward validation: rolling-window test on last 10 years to quantify RMSE
- [ ] Wind speed extrapolation to 100m using the power law profile: `WS100 = WS50 × (100/50)^α`
- [ ] Interactive Plotly visualisations to replace static Matplotlib charts
- [ ] `compare_cities.py` documentation and usage examples in README

---

## 📦 Dependencies

| Package | Version | Role |
|---|---|---|
| `pandas` | ≥ 1.3 | Data loading, reshaping, `melt()` for long-format plotting |
| `numpy` | ≥ 1.21 | Z-score computation, array operations |
| `matplotlib` | ≥ 3.4 | All visualisations |
| `seaborn` | ≥ 0.11 | `lineplot()`, `heatmap()`, statistical styling |
| `statsmodels` | ≥ 0.13 | `ARIMA`, `SARIMAX`, `get_forecast()`, `conf_int()` |
| `scikit-learn` | ≥ 1.0 | `mean_squared_error` for model evaluation |

See `requirements.txt` for pinned versions.

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d2137,50:1a3a5c,100:0a1628&height=100&section=footer" width="100%"/>
**Built with 🌞 NASA POWER · 📈 statsmodels · 🐼 Pandas · 📊 Matplotlib · 🌍 Berlin & Honolulu**

*Understanding climate to build a cleaner energy future.*

[![GitHub](https://img.shields.io/badge/GitHub-AIMLDS7-181717?style=flat-square&logo=github)](https://github.com/AIMLDS7)

</div>
"
