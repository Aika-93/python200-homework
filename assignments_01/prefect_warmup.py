# --- Pipelines ---

# Pipeline Q2

import numpy as np
import pandas as pd
from prefect import task, flow
from prefect.logging import get_run_logger
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    return pd.Series(arr, name="values")

@task
def clean_data(series):
    return series.dropna()

@task
def summarize_data(series):
    return {
        "mean": np.mean(series),
        "median": np.median(series),
        "std": np.std(series),
        "mode": series.mode()[0]
    }

@flow
def data_pipeline(arr):
    series = create_series(arr)
    clean_series = clean_data(series)
    summary = summarize_data(clean_series)
    return summary
    
if __name__ == "__main__":
    result = data_pipeline(arr)

    for key, value in result.items():
        print(f"{key}: {value}")

"""
1. Prefect is not needed here because the pipeline is very simple and can be done with basic Python functions.
2. A simple pipeline like this can still be useful in Prefect when it needs to run frequently, process incoming data regularly, or be part of a scheduled workflow. 

"""