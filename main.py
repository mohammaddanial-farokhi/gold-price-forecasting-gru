import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import os
import seaborn as sns

# ===========================================
# 1- EDA
# ===========================================
def create_twentieth_dataset(datasetname):
    dataset = pd.read_csv(datasetname)
    dataset["Date"] = pd.to_datetime(dataset["Date"])
    twentieth_dataset = dataset[dataset["Date"].dt.year >= 2000]
    return twentieth_dataset

def time_series_plot(dataset):

    plt.figure(figsize=(14, 7))  

    plt.plot(dataset["Date"], dataset["Price"], linewidth=2, color="blue")

    plt.title("Figure 1", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Price", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.7)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator(1))  
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  

    ax.yaxis.set_major_locator(MultipleLocator(500))  

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    #1. EDA
    twentieth_dataset=create_twentieth_dataset("monthly.csv")
    time_series_plot(twentieth_dataset)