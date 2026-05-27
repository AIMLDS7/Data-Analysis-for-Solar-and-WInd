"""
  compare_cities.py - Side-by-side comparison of Berlin and Honolulu climate data.

    Run this script after the individual city analyses to see how the two
locations differ across temperature, wind speed, and solar irradiance.
    """

    import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import clean_dataset, PARAM_LABELS, MONTH_COLS

  # ---------------------------------------------------------------------------
  # Configuration
  # ---------------------------------------------------------------------------

  BERLIN_FILE   = 'Berlin_Germany.csv'
  HONOLULU_FILE = 'Honolulu_USA.csv'
  CITY_BERLIN   = 'Berlin, Germany'
  CITY_HONOLULU = 'Honolulu, USA'
  PARAMETERS    = ['T2M', 'WS50M', 'ALLSKY_SFC_SW_DWN']

  # ---------------------------------------------------------------------------
  # Load and clean both datasets
  # ---------------------------------------------------------------------------

  print("=" * 60)
  print(f"Loading and cleaning: {CITY_BERLIN}")
  print("=" * 60)
  berlin = clean_dataset(BERLIN_FILE)

  print("\n" + "=" * 60)
  print(f"Loading and cleaning: {CITY_HONOLULU}")
  print("=" * 60)
  honolulu = clean_dataset(HONOLULU_FILE)

  # ---------------------------------------------------------------------------
  # Helper: filter by PARAMETER
  # ---------------------------------------------------------------------------

  def get_param(df, parameter):
      subset = df[df['PARAMETER'] == parameter].copy()
      subset.set_index('YEAR', inplace=True)
      return subset

  # ---------------------------------------------------------------------------
  # Plot 1: Annual trend comparison for each parameter
  # ---------------------------------------------------------------------------

  def plot_annual_comparison(param):
      """Overlay both cities on a single annual-trend line plot."""
      b_data = get_param(berlin,   param)
      h_data = get_param(honolulu, param)
      label  = PARAM_LABELS.get(param, param)

      if b_data.empty and h_data.empty:
          print(f"No data for parameter '{param}' in either city. Skipping.")
          return

      plt.figure(figsize=(14, 6))
      if not b_data.empty:
          plt.plot(b_data.index, b_data['ANN'], label=CITY_BERLIN,  color='steelblue')
      if not h_data.empty:
          plt.plot(h_data.index, h_data['ANN'], label=CITY_HONOLULU, color='coral')

      plt.title(f'Annual {label}: {CITY_BERLIN} vs {CITY_HONOLULU}')
      plt.xlabel('Year')
      plt.ylabel(label)
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.show()

  for param in PARAMETERS:
      plot_annual_comparison(param)

  # ---------------------------------------------------------------------------
  # Plot 2: Monthly average comparison (box plots per month)
  # ---------------------------------------------------------------------------

  def plot_monthly_boxplot_comparison(param):
      """Box plots of monthly values for each city, side by side."""
      label = PARAM_LABELS.get(param, param)
      present_months = [c for c in MONTH_COLS if c in berlin.columns]

      b_sub = berlin[berlin['PARAMETER'] == param][present_months].copy()
      h_sub = honolulu[honolulu['PARAMETER'] == param][present_months].copy()

      if b_sub.empty or h_sub.empty:
          return

    b_melted = b_sub.melt(var_name='Month', value_name=label)
      b_melted['City'] = CITY_BERLIN
    h_melted = h_sub.melt(var_name='Month', value_name=label)
    h_melted['City'] = CITY_HONOLULU

    combined = pd.concat([b_melted, h_melted], ignore_index=True)
    combined['Month'] = pd.Categorical(combined['Month'],
                                       categories=present_months, ordered=True)

    plt.figure(figsize=(16, 7))
    sns.boxplot(data=combined, x='Month', y=label, hue='City',
                palette={'Berlin, Germany': 'steelblue', 'Honolulu, USA': 'coral'})
    plt.title(f'Monthly {label} Distribution: {CITY_BERLIN} vs {CITY_HONOLULU}')
      plt.xlabel('Month')
      plt.ylabel(label)
      plt.legend(title='City')
      plt.grid(True, axis='y')
      plt.tight_layout()
      plt.show()

  for param in PARAMETERS:
      plot_monthly_boxplot_comparison(param)

  # ---------------------------------------------------------------------------
  # Plot 3: Summary statistics table
  # ---------------------------------------------------------------------------

  print("\n" + "=" * 60)
  print("SUMMARY STATISTICS COMPARISON")
  print("=" * 60)

  rows = []
  for param in PARAMETERS:
      label = PARAM_LABELS.get(param, param)
      b_ann = get_param(berlin,   param).get('ANN', pd.Series(dtype=float))
      h_ann = get_param(honolulu, param).get('ANN', pd.Series(dtype=float))
      rows.append({
          'Parameter': label,
          f'{CITY_BERLIN} Mean':   round(b_ann.mean(), 3) if not b_ann.empty else 'N/A',
          f'{CITY_BERLIN} Std':    round(b_ann.std(),  3) if not b_ann.empty else 'N/A',
          f'{CITY_HONOLULU} Mean': round(h_ann.mean(), 3) if not h_ann.empty else 'N/A',
          f'{CITY_HONOLULU} Std':  round(h_ann.std(),  3) if not h_ann.empty else 'N/A',
  })

  summary_df = pd.DataFrame(rows)
  print(summary_df.to_string(index=False))

  # ---------------------------------------------------------------------------
  # Plot 4: Heatmap of mean monthly values per city and parameter
  # ---------------------------------------------------------------------------

  def monthly_mean_heatmap(df, city, param):
      """Return a Series of mean monthly values for a given parameter."""
      present = [c for c in MONTH_COLS if c in df.columns]
      subset = df[df['PARAMETER'] == param][present]
      if subset.empty:
          return pd.Series(dtype=float)
      return subset.mean()

  fig, axes = plt.subplots(len(PARAMETERS), 2, figsize=(16, 4 * len(PARAMETERS)))
  for i, param in enumerate(PARAMETERS):
      label = PARAM_LABELS.get(param, param)
      for j, (df, city) in enumerate([(berlin, CITY_BERLIN), (honolulu, CITY_HONOLULU)]):
        means = monthly_mean_heatmap(df, city, param)
          if means.empty:
            axes[i, j].set_visible(False)
            continue
          sns.heatmap(
              means.values.reshape(1, -1),
              ax=axes[i, j],
              xticklabels=means.index,
              yticklabels=[label],
              cmap='YlOrRd',
              annot=True,
              fmt='.1f',
              linewidths=0.5,
              cbar_kws={'shrink': 0.8},
          )
          axes[i, j].set_title(f'{city}\n{label}')
          axes[i, j].set_xlabel('Month')

  plt.suptitle('Mean Monthly Values by Parameter and City', fontsize=14, y=1.02)
  plt.tight_layout()
  plt.show()

  print("\nComparison complete.")
