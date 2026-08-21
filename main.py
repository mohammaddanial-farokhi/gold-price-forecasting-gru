import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler


# ===========================================
# 1- EDA
# ===========================================
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


def autocorrelation(dataset):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(dataset["Price"], lags=40, ax=ax1)
    plot_pacf(dataset["Price"], lags=40, ax=ax2)
    plt.show()


def monthly_volatility(dataset):
    dataset['Returns'] = dataset['Price'].pct_change() * 100  
    
    top_returns = dataset.nlargest(5, 'Returns')
    print(top_returns[['Returns']])  

    plt.figure(figsize=(14, 5))
    plt.plot(dataset.index, dataset['Returns'], color='red', linewidth=0.8)
    plt.title('monthly Volatility persent')
    plt.grid(True)
    plt.show()


def plot_long_term_trend(dataset):
    dataset['MA_12'] = dataset['Price'].rolling(window=12).mean()
    dataset['MA_60'] = dataset['Price'].rolling(window=60).mean()

    plt.figure(figsize=(14, 6))
    plt.plot(dataset.index, dataset['Price'], label='main price', linewidth=1, alpha=0.5)
    plt.plot(dataset.index, dataset['MA_12'], label='year AVG', linewidth=2)
    plt.plot(dataset.index, dataset['MA_60'], label='5 years AVG', linewidth=2)
    plt.legend()
    plt.show()


# ===========================================
# 2- PreProcess
# ===========================================
def make_stationary_dataset(dataset):
    dataset["Log_Return"] = np.log(dataset["Price"]).diff()
    dataset.dropna(inplace=True)
    # print(dataset.head())

    return dataset


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
    dataset.set_index("Date", inplace=True)

    # 1. EDA
    # time_series_plot(dataset)
    # check_missing(dataset)
    # decomposition(dataset)
    # ADF_test(dataset)
    # autocorrelation(dataset)
    # monthly_volatility(dataset)
    # plot_long_term_trend(dataset)

    # 2. PreProcess
    # stationary_dataset = make_stationary_dataset(dataset)
    # create_x_y(dataset["Log_Return"])
