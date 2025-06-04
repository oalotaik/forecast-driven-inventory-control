import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import forecast_periodic_review_inventory

# Path to the sample data file
# file_path = "../data/raw/sample.csv"
file_path = "../data/raw/sample_future_forecasts.csv"
forecast_df = pd.read_csv(file_path)

# Prepare the sample data
df_sample = forecast_df[["period", "demand", "forecast"]].copy()
df_sample.columns = ["period", "demand", "forecast"]
# df_sample = df_sample.dropna(subset=["demand", "forecast"]).reset_index(drop=True)
df_sample


# Run the simulation function
result_df = forecast_periodic_review_inventory(
    df=df_sample,
    lead_time=1,
    review_period=3,
    safety_factor=1.645,
    initial_inventory=0,
    use_rolling_ss=False,
    include_review_period_in_ss=True,
    plot=True,
)
result_df.to_csv(
    "../data/processed/simulation_results_future_forecasts.csv",
    index=False,
)
