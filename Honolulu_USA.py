#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Task 1: Data Overview and Cleaning

# In[14]:


# Load the dataset
BD = pd.read_csv('Honolulu_USA.csv', skiprows=12)

# Overview of the dataset: number of features and samples
num_samples, num_features = BD.shape

# Datatypes of the features
datatypes = BD.dtypes

# Checking for missing values
missing_values = BD.isna().sum()
missing_values_percentage = (missing_values / num_samples) * 100

# Removing samples with NaN values
cleaned_data = BD.dropna()

# Metrics of the dataset after removing NaN values
cleaned_num_samples, cleaned_num_features = cleaned_data.shape

# Displaying the overview information
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")
print("\nDatatypes of features:")
print(datatypes)
print("\nMissing values (total):")
print(missing_values)
print("\nMissing values (percentage):")
print(missing_values_percentage)
print(f"\nNumber of samples after removing NaN: {cleaned_num_samples}")
print(f"Number of features after removing NaN: {cleaned_num_features}")


# In[15]:


# Function to identify outliers using Z-score
def identify_outliers_zscore(data, threshold=3):
    z_scores = np.abs((data - data.mean()) / data.std())
    return (z_scores > threshold)

# Identify outliers in each numerical column
outliers = pd.DataFrame()
for column in cleaned_data.columns[2:]:
    outliers[column] = identify_outliers_zscore(cleaned_data[column])

# Sum of outliers in each column
outliers_sum = outliers.sum()

# Remove rows that have outliers in any column
cleaned_data_no_outliers = cleaned_data[~outliers.any(axis=1)]

# Metrics after removing outliers
final_num_samples, final_num_features = cleaned_data_no_outliers.shape

# Displaying the summary of outliers removal
print("\nOutliers per column:")
print(outliers_sum)
print(f"\nNumber of samples after removing outliers: {final_num_samples}")
print(f"Number of features after removing outliers: {final_num_features}")


# In[16]:


# Visual analysis
plt.figure(figsize=(14, 8))
sns.lineplot(x='YEAR', y='ANN', data=cleaned_data_no_outliers)
plt.title('Yearly Average Temperature Trend')
plt.xlabel('Year')
plt.ylabel('Annual Average Temperature (°C)')
plt.grid(True)
plt.show()


# In[17]:


monthly_columns = cleaned_data_no_outliers.columns[2:14]
monthly_data = cleaned_data_no_outliers.melt(id_vars=['YEAR'], value_vars=monthly_columns, var_name='Month', value_name='Temperature')

plt.figure(figsize=(16, 10))
sns.lineplot(x='YEAR', y='Temperature', hue='Month', data=monthly_data, palette='tab10')
plt.title('Monthly Average Temperatures Across Years')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()


# # Task 2: Time Series Processing

# In[20]:


# Visual analysis for T2M parameter
plt.figure(figsize=(14, 8))
sns.lineplot(x='YEAR', y='ANN', data=cleaned_data_no_outliers[cleaned_data_no_outliers['PARAMETER'] == 'T2M'])
plt.title('Yearly Average Temperature Trend')
plt.xlabel('Year')
plt.ylabel('Annual Average Temperature (°C)')
plt.grid(True)
plt.show()


# In[21]:


# Visual analysis for WS50M parameter
plt.figure(figsize=(14, 8))
sns.lineplot(x='YEAR', y='ANN', data=cleaned_data_no_outliers[cleaned_data_no_outliers['PARAMETER'] == 'WS50M'])
plt.title('Yearly Average Wind Speed')
plt.xlabel('Year')
plt.ylabel('Annual Average Wind Speed (m/s)')
plt.grid(True)
plt.show()


# In[9]:


# Visual analysis for WS50M_MAX parameter
plt.figure(figsize=(14, 8))
sns.lineplot(x='YEAR', y='ANN', data=cleaned_data_no_outliers[cleaned_data_no_outliers['PARAMETER'] == 'WS50M_MAX'])
plt.title('Yearly Maximum Wind Speed at 50 Meters (WS50M_MAX)')
plt.xlabel('Year')
plt.ylabel('Annual Maximum Wind Speed (m/s)')
plt.grid(True)
plt.show()


# In[10]:


# Visual analysis for ALLSKY_SFC_SW_DWN parameter
plt.figure(figsize=(14, 8))
sns.lineplot(x='YEAR', y='ANN', data=cleaned_data_no_outliers[cleaned_data_no_outliers['PARAMETER'] == 'ALLSKY_SFC_SW_DWN'])
plt.title('Yearly All Sky Surface Shortwave Downward Irradiance (ALLSKY_SFC_SW_DWN)')
plt.xlabel('Year')
plt.ylabel('Annual Irradiance (kW-hr/m^2/day)')
plt.grid(True)
plt.show()


# In[11]:


# Correlation matrix for numerical columns
numeric_data = cleaned_data_no_outliers.select_dtypes(include=[np.number])

correlation_matrix = numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()


# # Task 3: SARIMA Forecasting

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# In[50]:


# Function to fit ARIMA model and forecast
def arima_forecast(data, parameter, order=(5,1,0), forecast_periods=10):
    # Filter data for the specific parameter
    param_data = data[data['PARAMETER'] == parameter].copy()
    param_data.set_index('YEAR', inplace=True)
    
    # Fit ARIMA model
    model = ARIMA(param_data['ANN'], order=order)
    model_fit = model.fit()
    
    # Forecast
    forecast = model_fit.forecast(steps=forecast_periods)
    
    # Plot the results
    plt.figure(figsize=(14, 8))
    plt.plot(param_data.index, param_data['ANN'], label='Historical Data')
    plt.plot(np.arange(param_data.index[-1] + 1, param_data.index[-1] + 1 + forecast_periods), forecast, label='10 year Forecast', linestyle='--')
    plt.title(f'ARIMA Forecast for Temperature at Honolulu')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print the forecast values
    print(f'Forecast values for Temperature of Honolulu')
    print(forecast)

# ARIMA Forecast for T2M
arima_forecast(cleaned_data_no_outliers, 'T2M', forecast_periods=10)


# In[52]:


# Function to fit ARIMA model and forecast
def arima_forecast(data, parameter, order=(5,1,0), forecast_periods=10):
    # Filter data for the specific parameter
    param_data = data[data['PARAMETER'] == parameter].copy()
    param_data.set_index('YEAR', inplace=True)
    
    # Fit ARIMA model
    model = ARIMA(param_data['ANN'], order=order)
    model_fit = model.fit()
    
    # Forecast
    forecast = model_fit.forecast(steps=forecast_periods)
    
    # Plot the results
    plt.figure(figsize=(14, 8))
    plt.plot(param_data.index, param_data['ANN'], label='Historical Data')
    plt.plot(np.arange(param_data.index[-1] + 1, param_data.index[-1] + 1 + forecast_periods), forecast, label='10 year Forecast', linestyle='--')
    plt.title(f'ARIMA Forecast for Windspeed at Honolulu')
    plt.xlabel('Year')
    plt.ylabel('Wind Speed (m/s)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print the forecast values
    print(f'Forecast values for Windspeed at Honolulu')
    print(forecast)

# ARIMA Forecast for WS50M
arima_forecast(cleaned_data_no_outliers, 'WS50M', forecast_periods=10)


# In[53]:


# Function to fit ARIMA model and forecast
def arima_forecast(data, parameter, order=(5,1,0), forecast_periods=10):
    # Filter data for the specific parameter
    param_data = data[data['PARAMETER'] == parameter].copy()
    param_data.set_index('YEAR', inplace=True)
    
    # Fit ARIMA model
    model = ARIMA(param_data['ANN'], order=order)
    model_fit = model.fit()
    
    # Forecast
    forecast = model_fit.forecast(steps=forecast_periods)
    
    # Plot the results
    plt.figure(figsize=(14, 8))
    plt.plot(param_data.index, param_data['ANN'], label='Historical Data')
    plt.plot(np.arange(param_data.index[-1] + 1, param_data.index[-1] + 1 + forecast_periods), forecast, label='10 year Forecast', linestyle='--')
    plt.title(f'ARIMA Forecast for Maximum Windspeed at Honolulu')
    plt.xlabel('Year')
    plt.ylabel('Maximum Wind Speed (m/s)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print the forecast values
    print(f'Forecast values for Maximum Windspeed at Honolulu')
    print(forecast)

# ARIMA Forecast for WS50M_MAX
arima_forecast(cleaned_data_no_outliers, 'WS50M_MAX', forecast_periods=10)


# In[54]:


# Function to fit ARIMA model and forecast
def arima_forecast(data, parameter, order=(5,1,0), forecast_periods=10):
    # Filter data for the specific parameter
    param_data = data[data['PARAMETER'] == parameter].copy()
    param_data.set_index('YEAR', inplace=True)
    
    # Fit ARIMA model
    model = ARIMA(param_data['ANN'], order=order)
    model_fit = model.fit()
    
    # Forecast
    forecast = model_fit.forecast(steps=forecast_periods)
    
    # Plot the results
    plt.figure(figsize=(14, 8))
    plt.plot(param_data.index, param_data['ANN'], label='Historical Data')
    plt.plot(np.arange(param_data.index[-1] + 1, param_data.index[-1] + 1 + forecast_periods), forecast, label='10 year Forecast', linestyle='--')
    plt.title(f'ARIMA Forecast for Irradiance at Honolulu')
    plt.xlabel('Year')
    plt.ylabel('Irradiance (kw-hr/m2/day)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print the forecast values
    print(f'Forecast values for Irradiance at Honolulu')
    print(forecast)

# ARIMA Forecast for ALLSKY_SFC_SW_DWN
arima_forecast(cleaned_data_no_outliers, 'ALLSKY_SFC_SW_DWN', forecast_periods=10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Task 4: SARIMA Forecasting

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# In[41]:


def sarima_forecast(data, parameter, order=(1,1,1), seasonal_order=(1,1,1,12), forecast_periods=10):
    # Filter data for the specific parameter
    param_data = data[data['PARAMETER'] == parameter].copy()
    param_data.set_index('YEAR', inplace=True)
    
    # Fit SARIMA model
    model = SARIMAX(param_data['ANN'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    
    # Forecast
    forecast = model_fit.get_forecast(steps=forecast_periods)
    forecast_index = np.arange(param_data.index[-1] + 1, param_data.index[-1] + 1 + forecast_periods)
    forecast_values = forecast.predicted_mean
    
    # Combine historical data with forecasted data
    combined_data = pd.concat([param_data['ANN'], pd.Series(forecast_values, index=forecast_index)])
    
    # Plot the results
    plt.figure(figsize=(14, 8))
    plt.plot(param_data.index, param_data['ANN'], label='Historical Data')
    plt.plot(forecast_index, forecast_values, label='10 Year Forecasted Data', linestyle='--')
    plt.title(f'SARIMA Forecast for Windspeed at Honolulu')
    plt.xlabel('Year')
    plt.ylabel('Wind Speed (m/s)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print the forecast values
    forecast_df = pd.DataFrame({'Year': forecast_index, 'Forecasted_Value': forecast_values})
    print(f'Forecasted values for the next {forecast_periods} years for Windspeed at Honolulu:')
    print(forecast_df)

# Load the cleaned data from Task 1-3
# Here, I assume `cleaned_data_no_outliers` is already available from previous tasks

# Example of what cleaned_data_no_outliers might look like
# cleaned_data_no_outliers = pd.read_csv('cleaned_berlin_germany.csv')

# SARIMA Forecast for WS50M
sarima_forecast(cleaned_data_no_outliers, 'WS50M', order=(1,1,1), seasonal_order=(1,1,1,12), forecast_periods=10)


# In[43]:


def sarima_forecast(data, parameter, order=(1,1,1), seasonal_order=(1,1,1,12), forecast_periods=10):
    # Filter data for the specific parameter
    param_data = data[data['PARAMETER'] == parameter].copy()
    param_data.set_index('YEAR', inplace=True)
    
    # Fit SARIMA model
    model = SARIMAX(param_data['ANN'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    
    # Forecast
    forecast = model_fit.get_forecast(steps=forecast_periods)
    forecast_index = np.arange(param_data.index[-1] + 1, param_data.index[-1] + 1 + forecast_periods)
    forecast_values = forecast.predicted_mean
    
    # Combine historical data with forecasted data
    combined_data = pd.concat([param_data['ANN'], pd.Series(forecast_values, index=forecast_index)])
    
    # Plot the results
    plt.figure(figsize=(14, 8))
    plt.plot(param_data.index, param_data['ANN'], label='Historical Data')
    plt.plot(forecast_index, forecast_values, label='10 Year Forecasted Data', linestyle='--')
    plt.title(f'SARIMA Forecast for Maximum Windspeed at Honolulu')
    plt.xlabel('Year')
    plt.ylabel('Maximum Wind Speed (m/s)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print the forecast values
    forecast_df = pd.DataFrame({'Year': forecast_index, 'Forecasted_Value': forecast_values})
    print(f'Forecasted values for the next {forecast_periods} years for Maximum Windspeed at Honolulu:')
    print(forecast_df)

# Load the cleaned data from Task 1-3
# Here, I assume `cleaned_data_no_outliers` is already available from previous tasks

# Example of what cleaned_data_no_outliers might look like
# cleaned_data_no_outliers = pd.read_csv('cleaned_berlin_germany.csv')

# SARIMA Forecast for WS50M_MAX
sarima_forecast(cleaned_data_no_outliers, 'WS50M_MAX' , order=(1,1,1), seasonal_order=(1,1,1,12), forecast_periods=10)


# In[44]:


def sarima_forecast(data, parameter, order=(1,1,1), seasonal_order=(1,1,1,12), forecast_periods=10):
    # Filter data for the specific parameter
    param_data = data[data['PARAMETER'] == parameter].copy()
    param_data.set_index('YEAR', inplace=True)
    
    # Fit SARIMA model
    model = SARIMAX(param_data['ANN'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    
    # Forecast
    forecast = model_fit.get_forecast(steps=forecast_periods)
    forecast_index = np.arange(param_data.index[-1] + 1, param_data.index[-1] + 1 + forecast_periods)
    forecast_values = forecast.predicted_mean
    
    # Combine historical data with forecasted data
    combined_data = pd.concat([param_data['ANN'], pd.Series(forecast_values, index=forecast_index)])
    
    # Plot the results
    plt.figure(figsize=(14, 8))
    plt.plot(param_data.index, param_data['ANN'], label='Historical Data')
    plt.plot(forecast_index, forecast_values, label='10 Year Forecasted Data', linestyle='--')
    plt.title(f'SARIMA Forecast for Temperature at Honolulu')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print the forecast values
    forecast_df = pd.DataFrame({'Year': forecast_index, 'Forecasted_Value': forecast_values})
    print(f'Forecasted values for the next {forecast_periods} years for Temperature at Honolulu:')
    print(forecast_df)

# Load the cleaned data from Task 1-3
# Here, I assume `cleaned_data_no_outliers` is already available from previous tasks

# Example of what cleaned_data_no_outliers might look like
# cleaned_data_no_outliers = pd.read_csv('cleaned_berlin_germany.csv')

# SARIMA Forecast for T2M
sarima_forecast(cleaned_data_no_outliers, 'T2M' , order=(1,1,1), seasonal_order=(1,1,1,12), forecast_periods=10)


# In[45]:


def sarima_forecast(data, parameter, order=(1,1,1), seasonal_order=(1,1,1,12), forecast_periods=10):
    # Filter data for the specific parameter
    param_data = data[data['PARAMETER'] == parameter].copy()
    param_data.set_index('YEAR', inplace=True)
    
    # Fit SARIMA model
    model = SARIMAX(param_data['ANN'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    
    # Forecast
    forecast = model_fit.get_forecast(steps=forecast_periods)
    forecast_index = np.arange(param_data.index[-1] + 1, param_data.index[-1] + 1 + forecast_periods)
    forecast_values = forecast.predicted_mean
    
    # Combine historical data with forecasted data
    combined_data = pd.concat([param_data['ANN'], pd.Series(forecast_values, index=forecast_index)])
    
    # Plot the results
    plt.figure(figsize=(14, 8))
    plt.plot(param_data.index, param_data['ANN'], label='Historical Data')
    plt.plot(forecast_index, forecast_values, label='10 Year Forecasted Data', linestyle='--')
    plt.title(f'SARIMA Forecast for Irradiance at Honolulu')
    plt.xlabel('Year')
    plt.ylabel('Irradiance (kw-hr/m2/day)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print the forecast values
    forecast_df = pd.DataFrame({'Year': forecast_index, 'Forecasted_Value': forecast_values})
    print(f'Forecasted values for the next {forecast_periods} years for Irradiance at Honolulu:')
    print(forecast_df)

# Load the cleaned data from Task 1-3
# Here, I assume `cleaned_data_no_outliers` is already available from previous tasks

# Example of what cleaned_data_no_outliers might look like
# cleaned_data_no_outliers = pd.read_csv('cleaned_berlin_germany.csv')

# SARIMA Forecast for ALLSKY_SFC_SW_DWN
sarima_forecast(cleaned_data_no_outliers, 'ALLSKY_SFC_SW_DWN' , order=(1,1,1), seasonal_order=(1,1,1,12), forecast_periods=10)


# In[ ]:





# In[47]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the two CSV files
file_path1 = 'bdata.csv'
file_path2 = 'hdata.csv'

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Filter data to only include 'T2M' parameter
df1 = df1[df1['PARAMETER'] == 'T2M']
df2 = df2[df2['PARAMETER'] == 'T2M']

# Set the 'YEAR' as index
df1.set_index('YEAR', inplace=True)
df2.set_index('YEAR', inplace=True)

# Calculate the mean of each month and annual for both datasets
monthly_mean1 = df1.loc[:, 'JAN':'ANN'].mean()
monthly_mean2 = df2.loc[:, 'JAN':'ANN'].mean()

# Calculate the annual mean temperature for both datasets
annual_mean1 = df1['ANN'].mean()
annual_mean2 = df2['ANN'].mean()

# Plotting annual trends
years1 = df1.index
years2 = df2.index

plt.figure(figsize=(14, 7))
plt.plot(years1, df1['ANN'], label='Berlin', marker='o')
plt.plot(years2, df2['ANN'], label='Honolulu', marker='o')

plt.xlabel('Year')
plt.ylabel('Annual Mean Temperature (°C)')
plt.title('Comparison of Annual Mean Temperature Trends')
plt.legend()
plt.grid(True)
plt.show()

# Print the calculated mean values
print(f'Mean Annual Temperature for Berlin: {annual_mean1:.2f} °C')
print(f'Mean Annual Temperature for Honolulu: {annual_mean2:.2f} °C')


# In[10]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the two CSV files
file_path1 = 'bdata.csv'
file_path2 = 'hdata.csv'

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Filter data to only include 'ALLSKY_SFC_SW_DWN' parameter
df1 = df1[df1['PARAMETER'] == 'ALLSKY_SFC_SW_DWN']
df2 = df2[df2['PARAMETER'] == 'ALLSKY_SFC_SW_DWN']

# Set the 'YEAR' as index
df1.set_index('YEAR', inplace=True)
df2.set_index('YEAR', inplace=True)

# Calculate the mean of each month and annual for both datasets
monthly_mean1 = df1.loc[:, 'JAN':'ANN'].mean()
monthly_mean2 = df2.loc[:, 'JAN':'ANN'].mean()

# Calculate the annual mean irradiance for both datasets
annual_mean1 = df1['ANN'].mean()
annual_mean2 = df2['ANN'].mean()

# Plotting annual trends
years1 = df1.index
years2 = df2.index

plt.figure(figsize=(14, 7))
plt.plot(years1, df1['ANN'], label='Berlin', marker='o')
plt.plot(years2, df2['ANN'], label='Honolulu', marker='o')

plt.xlabel('Year')
plt.ylabel('Annual Mean Irradiance (kWh/m²/day)')
plt.title('Comparison of Annual Mean Irradiance Trends')
plt.legend()
plt.grid(True)
plt.show()

# Print the calculated mean values
print(f'Mean Annual Irradiance for Berlin: {annual_mean1:.2f} kWh/m²/day')
print(f'Mean Annual Irradiance for Honolulu: {annual_mean2:.2f} kWh/m²/day')


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the two CSV files
file_path1 = 'bdata.csv'
file_path2 = 'hdata.csv'

df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Filter data to only include 'WS50M' parameter
df1 = df1[df1['PARAMETER'] == 'WS50M']
df2 = df2[df2['PARAMETER'] == 'WS50M']

# Set the 'YEAR' as index
df1.set_index('YEAR', inplace=True)
df2.set_index('YEAR', inplace=True)

# Calculate the mean of each month and annual for both datasets
monthly_mean1 = df1.loc[:, 'JAN':'ANN'].mean()
monthly_mean2 = df2.loc[:, 'JAN':'ANN'].mean()

# Calculate the annual mean wind speed for both datasets
annual_mean1 = df1['ANN'].mean()
annual_mean2 = df2['ANN'].mean()

# Plotting annual trends
years1 = df1.index
years2 = df2.index

plt.figure(figsize=(14, 7))
plt.plot(years1, df1['ANN'], label='Berlin', marker='o')
plt.plot(years2, df2['ANN'], label='Honolulu', marker='o')

plt.xlabel('Year')
plt.ylabel('Annual Mean Wind Speed (m/s)')
plt.title('Comparison of Annual Mean Wind Speed Trends')
plt.legend()
plt.grid(True)
plt.show()

# Print the calculated mean values
print(f'Mean Annual Wind Speed for Berlin: {annual_mean1:.2f} m/s')
print(f'Mean Annual Wind Speed for Honolulu: {annual_mean2:.2f} m/s')


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
file_path = 'hdata.csv'
df = pd.read_csv(file_path)

# Filter data to only include 'T2M' parameter
df = df[df['PARAMETER'] == 'T2M']

# Set the 'YEAR' as index
df.set_index('YEAR', inplace=True)

# Calculate the mean of all years for each month and annual
monthly_mean = df.loc[:, 'JAN':'ANN'].mean()

# Plotting annual trends
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['ANN'], marker='o', label='Annual Temperature', color='blue')

# Plot the mean temperature for each month and annual
plt.figure(figsize=(12, 6))
monthly_mean.plot(kind='bar', color='skyblue')
plt.title('Mean Monthly and Annual Temperature (1981-2022)')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.grid()
plt.tight_layout()

# Customize the plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['ANN'], marker='o', label='Annual Average Temperature', color='blue')
plt.title('Annual Temperature Trends (1981-2022)')
plt.xlabel('Year')
plt.ylabel('Annual Temperature (°C)')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plots
plt.show()


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
file_path = 'bdata.csv'
df = pd.read_csv(file_path)

# Filter data to only include 'T2M' parameter
df = df[df['PARAMETER'] == 'T2M']

# Set the 'YEAR' as index
df.set_index('YEAR', inplace=True)

# Calculate the mean of all years for each month and annual
monthly_mean = df.loc[:, 'JAN':'ANN'].mean()

# Plotting annual trends
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['ANN'], marker='o', label='Annual Temperature', color='blue')

# Plot the mean temperature for each month and annual
plt.figure(figsize=(12, 6))
monthly_mean.plot(kind='bar', color='skyblue')
plt.title('Mean Monthly and Annual Temperature (1981-2022)')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.grid()
plt.tight_layout()

# Customize the plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['ANN'], marker='o', label='Annual Average Temperature', color='blue')
plt.title('Annual Temperature Trends (1981-2022)')
plt.xlabel('Year')
plt.ylabel('Annual Temperature (°C)')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plots
plt.show()


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
file_path = 'hdata.csv'
df = pd.read_csv(file_path)

# Filter data to only include 'ALLSKY_SFC_SW_DWN' parameter
df = df[df['PARAMETER'] == 'ALLSKY_SFC_SW_DWN']

# Set the 'YEAR' as index
df.set_index('YEAR', inplace=True)

# Calculate the mean of all years for each month and annual
monthly_mean = df.loc[:, 'JAN':'ANN'].mean()

# Plotting annual trends
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['ANN'], marker='o', label='Annual Irradiance', color='blue')

# Plot the mean irradiance for each month and annual
plt.figure(figsize=(12, 6))
monthly_mean.plot(kind='bar', color='skyblue')
plt.title('Mean Monthly and Annual Irradiance (1981-2022)')
plt.xlabel('Month')
plt.ylabel('Irradiance (kWh/m²/day)')
plt.grid()
plt.tight_layout()

# Customize the plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['ANN'], marker='o', label='Annual Average Irradiance', color='blue')
plt.title('Annual Irradiance Trends (1981-2022)')
plt.xlabel('Year')
plt.ylabel('Annual Irradiance (kWh/m²/day)')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plots
plt.show()


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
file_path = 'bdata.csv'
df = pd.read_csv(file_path)

# Filter data to only include 'ALLSKY_SFC_SW_DWN' parameter
df = df[df['PARAMETER'] == 'ALLSKY_SFC_SW_DWN']

# Set the 'YEAR' as index
df.set_index('YEAR', inplace=True)

# Calculate the mean of all years for each month and annual
monthly_mean = df.loc[:, 'JAN':'ANN'].mean()

# Plotting annual trends
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['ANN'], marker='o', label='Annual Irradiance', color='blue')

# Plot the mean irradiance for each month and annual
plt.figure(figsize=(12, 6))
monthly_mean.plot(kind='bar', color='skyblue')
plt.title('Mean Monthly and Annual Irradiance (1981-2022)')
plt.xlabel('Month')
plt.ylabel('Irradiance (kWh/m²/day)')
plt.grid()
plt.tight_layout()

# Customize the plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['ANN'], marker='o', label='Annual Average Irradiance', color='blue')
plt.title('Annual Irradiance Trends (1981-2022)')
plt.xlabel('Year')
plt.ylabel('Annual Irradiance (kWh/m²/day)')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plots
plt.show()


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
file_path = 'bdata.csv'
df = pd.read_csv(file_path)

# Filter data to only include 'WS50M' parameter
df = df[df['PARAMETER'] == 'WS50M']

# Set the 'YEAR' as index
df.set_index('YEAR', inplace=True)

# Calculate the mean of all years for each month and annual
monthly_mean = df.loc[:, 'JAN':'ANN'].mean()

# Plotting annual trends
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['ANN'], marker='o', label='Annual Wind Speed', color='blue')
plt.title('Berlin Annual Wind Speed Trends (1981-2022)')
plt.xlabel('Year')
plt.ylabel('Annual Wind Speed (m/s)')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()

# Plot the mean wind speed for each month and annual
plt.figure(figsize=(12, 6))
monthly_mean.plot(kind='bar', color='skyblue')
plt.title('Berlin Mean Monthly and Annual Wind Speed (1981-2022)')
plt.xlabel('Month')
plt.ylabel('Wind Speed (m/s)')
plt.grid()
plt.tight_layout()

# Show the plots
plt.show()


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
file_path = 'bdata.csv'
df = pd.read_csv(file_path)

# Filter data to only include 'WS50M' parameter
df = df[df['PARAMETER'] == 'WS50M']

# Set the 'YEAR' as index
df.set_index('YEAR', inplace=True)

# Calculate the mean of all years for each month and annual
monthly_mean = df.loc[:, 'JAN':'ANN'].mean()

# Plotting annual trends
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['ANN'], marker='o', label='Annual Wind Speed', color='blue')
plt.title('Honolulu Annual Wind Speed Trends (1981-2022)')
plt.xlabel('Year')
plt.ylabel('Annual Wind Speed (m/s)')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()

# Plot the mean wind speed for each month and annual
plt.figure(figsize=(12, 6))
monthly_mean.plot(kind='bar', color='skyblue')
plt.title('Honolulu Mean Monthly and Annual Wind Speed (1981-2022)')
plt.xlabel('Month')
plt.ylabel('Wind Speed (m/s)')
plt.grid()
plt.tight_layout()

# Show the plots
plt.show()


# In[ ]:




