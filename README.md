# Data Analysis for Solar and Wind

> Solar and Wind Data Visualization and Time Series Forecasting using NASA POWER climate data for Berlin, Germany and Honolulu, USA.
>
> ---
>
> ## Overview
>
> This project analyses multi-decade meteorological data to understand long-term trends in temperature, wind speed, and solar irradiance at two geographically distinct locations. It also builds ARIMA and SARIMA time-series models to forecast future values.
>
> **Key analyses performed:**
>
> - Data cleaning: missing-value detection and Z-score outlier removal
> - - Exploratory visualisation: annual trends, monthly breakdowns, and correlation heatmaps
>   - - ARIMA forecasting for temperature, wind speed, and solar irradiance
>     - - SARIMA forecasting with seasonal components and 95% confidence intervals
>      
>       - ---
>
> ## Data Source
>
> Data was obtained from the [NASA POWER](https://power.larc.nasa.gov/) (Prediction Of Worldwide Energy Resources) project.
>
> **Parameters analysed:**
>
> | Parameter | Description |
> |-----------|-------------|
> | `T2M` | Temperature at 2 Meters (°C) |
> | `WS50M` | Wind Speed at 50 Meters (m/s) |
> | `WS50M_MAX` | Maximum Wind Speed at 50 Meters (m/s) |
> | `ALLSKY_SFC_SW_DWN` | All-Sky Surface Shortwave Downward Irradiance (kW-hr/m²/day) |
>
> ---
>
> ## Project Structure
>
> ```
> .
> ├── Berlin_Germany.csv       # Raw NASA POWER data for Berlin, Germany
> ├── Berlin_germany.py        # Analysis script for Berlin
> ├── Honolulu_USA.csv         # Raw NASA POWER data for Honolulu, USA
> ├── Honolulu_USA.py          # Analysis script for Honolulu
> ├── utils.py                 # Shared utility functions (cleaning, plotting, forecasting)
> ├── requirements.txt         # Python package dependencies
> ├── bdata.csv                # Processed/intermediate data for Berlin
> ├── hdata.csv                # Processed/intermediate data for Honolulu
> └── README.md
> ```
>
> ---
>
> ## Getting Started
>
> ### 1. Clone the repository
>
> ```bash
> git clone https://github.com/AIMLDS7/Data-Analysis-for-Solar-and-WInd.git
> cd Data-Analysis-for-Solar-and-WInd
> ```
>
> ### 2. Install dependencies
>
> ```bash
> pip install -r requirements.txt
> ```
>
> ### 3. Run the analysis
>
> ```bash
> # Berlin analysis
> python Berlin_germany.py
>
> # Honolulu analysis
> python Honolulu_USA.py
> ```
>
> ---
>
> ## Shared Utilities (`utils.py`)
>
> All reusable logic lives in `utils.py` so city-specific scripts stay concise and consistent. Key functions:
>
> | Function | Purpose |
> |----------|---------|
> | `load_dataset(filepath)` | Load NASA POWER CSV, skipping metadata rows |
> | `clean_dataset(filepath)` | Full pipeline: load → drop NaN → remove outliers |
> | `plot_annual_trend(df, parameter, city)` | Line plot of annual averages |
> | `plot_monthly_trends(df, city)` | Monthly breakdown across years |
> | `plot_correlation_heatmap(df, city)` | Feature correlation heatmap |
> | `arima_forecast(data, parameter, city)` | ARIMA model + forecast plot |
> | `sarima_forecast(data, parameter, city)` | SARIMA model + forecast with confidence interval |
>
> ---
>
> ## Requirements
>
> - Python 3.8+
> - - See `requirements.txt` for all dependencies
>  
>   - ---
>
> ## License
>
> This project is open-source. Feel free to use and adapt the code with attribution.
