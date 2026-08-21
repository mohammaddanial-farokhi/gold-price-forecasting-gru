import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler


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


def check_missing(dataset):
    main = pd.read_csv(dataset)
    column_names = list(main.columns)
    for i in column_names:
        total = len(main[i])
        missing = main[i].isna().sum()
        existing = total - missing
        print(f"there is {missing} missing {i} in {existing} values (Total: {total})")


def decomposition(dataset):
    decomposition = seasonal_decompose(dataset.set_index("Date")["Price"].dropna(), model="additive", period=12)
    decomposition.plot()
    plt.show()


def ADF_test(dataset):
    result = adfuller(dataset["Price"])
    p_value = result[1]
    print(f"ADF-value: {result[0]}")
    print(f"p-value: {result[1]}")

    if p_value > 0.05:
        print("this dataset is Non-Stationary")


# ===========================================
# 2- PreProcess
# ===========================================
def make_stationary_dataset(dataset):
    dataset["Log_Return"] = np.log(dataset["Price"]).diff()
    dataset.dropna(inplace=True)
    print(dataset.head())


def create_x_y(log_return):
    series = log_return.dropna().values.reshape(-1, 1)

    scaler = StandardScaler()
    scaled_series = scaler.fit_transform(series)

    def create_sequences(data, look_back=12):
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i - look_back : i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_series, look_back=12)

    print(f"input shape(X): {X.shape}")
    print(f"output shape(y): {y.shape}")

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"number of training data: {len(X_train)}")
    print(f"number of testng data: {len(X_test)}")


if __name__ == "__main__":

    dataset = pd.read_csv("monthly.csv")
    dataset["Date"] = pd.to_datetime(dataset["Date"])

    # 1. EDA
    # twentieth_dataset=create_twentieth_dataset("monthly.csv")
    # time_series_plot(twentieth_dataset)
    # check_missing(dataset)
    # decomposition(dataset)
    # ADF_test(dataset)

    # 2. PreProcess
    make_stationary_dataset(dataset)
    # create_x_y(dataset["Log_Return"])
