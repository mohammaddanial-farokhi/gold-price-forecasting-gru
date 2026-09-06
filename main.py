import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


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
    decomposition = seasonal_decompose(
        dataset.set_index("Date")["Price"].dropna(),
        model="additive",
        period=12,
    )
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
    dataset["Returns"] = dataset["Price"].pct_change() * 100

    top_returns = dataset.nlargest(5, "Returns")
    print(top_returns[["Returns"]])

    largest_negative = dataset.nsmallest(5, "Returns")
    print(largest_negative[["Returns"]])

    plt.figure(figsize=(14, 5))
    plt.plot(dataset.index, dataset["Returns"], color="red", linewidth=0.8)
    plt.title("monthly Volatility persent")
    plt.grid(True)
    plt.show()


def plot_long_term_trend(dataset):
    dataset["MA_12"] = dataset["Price"].rolling(window=12).mean()
    dataset["MA_60"] = dataset["Price"].rolling(window=60).mean()

    plt.figure(figsize=(14, 6))
    plt.plot(
        dataset.index, dataset["Price"], label="main price", linewidth=1, alpha=0.5
    )
    plt.plot(dataset.index, dataset["MA_12"], label="year AVG", linewidth=2)
    plt.plot(dataset.index, dataset["MA_60"], label="5 years AVG", linewidth=2)
    plt.legend()
    plt.show()


# ===========================================
# 2- PreProcess
# ===========================================
def make_stationary_dataset(dataset):

    dataset = dataset.copy()
    dataset = dataset.sort_index()
    dataset = dataset[dataset.index.year >= 1880].copy()

    if (dataset["Price"] <= 0).any():
        raise ValueError("Price contains zero or negative values.")

    dataset["Log_Return"] = np.log(dataset["Price"] / dataset["Price"].shift(1))

    dataset.dropna(subset=["Log_Return"], inplace=True)

    return dataset


def RNN_sequence_creation(log_return, look_back=12):

    series = log_return.dropna().values.reshape(-1, 1)

    # ==========================================
    # 1. Time Series Split
    # ==========================================

    total_size = len(series)

    train_size = int(total_size * 0.70)
    validation_size = int(total_size * 0.15)

    train_series = series[:train_size]

    validation_series = series[train_size : train_size + validation_size]

    test_series = series[train_size + validation_size :]

    print("Dataset split:")
    print(f"Total:      {len(series)}")
    print(f"Train:      {len(train_series)}")
    print(f"Validation: {len(validation_series)}")
    print(f"Test:       {len(test_series)}")

    # ==========================================
    # 2. Scaling
    # ==========================================

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_series)

    validation_scaled = scaler.transform(validation_series)

    test_scaled = scaler.transform(test_series)

    # ==========================================
    # 3. Sequence Creation
    # ==========================================

    def create_sequences(data, look_back):

        X, y = [], []

        for i in range(look_back, len(data)):
            X.append(data[i - look_back : i])
            y.append(data[i, 0])

        return np.array(X), np.array(y)

    # ==========================================
    # 4. Training Sequences
    # ==========================================

    X_train, y_train = create_sequences(train_scaled, look_back)

    # ==========================================
    # 5. Validation Sequences
    # ==========================================

    validation_input = np.concatenate([train_scaled[-look_back:], validation_scaled])

    X_validation, y_validation = create_sequences(validation_input, look_back)

    # ==========================================
    # 6. Test Sequences
    # ==========================================

    test_input = np.concatenate([validation_scaled[-look_back:], test_scaled])

    X_test, y_test = create_sequences(test_input, look_back)

    # ==========================================
    # 7. Print Shapes
    # ==========================================

    print("\nSequence shapes:")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_validation:", X_validation.shape)
    print("y_validation:", y_validation.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    return (X_train, X_validation, X_test, y_train, y_validation, y_test, scaler)


def log_return_ADF(dataset):
    result = adfuller(dataset["Log_Return"].dropna())

    print("log return ADF Statistic:", result[0])
    print("log return p-value:", result[1])


def log_return_autocorrelation(dataset):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(dataset["Log_Return"], lags=40, ax=ax1)
    plot_pacf(dataset["Log_Return"], lags=40, ax=ax2)
    plt.show()


# ===========================================
# 3- Modeling
# ===========================================
def build_gru_model(input_shape):
    model = Sequential(
        [
            GRU(64, activation="tanh", input_shape=input_shape),
            Dense(1024, activation="relu"),
            Dropout(0.4),
            Dense(1),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ===========================================
# 4- train or load save model
# ===========================================
def train_rnn(model, X_train, y_train, X_validation, y_validation):
    MODEL_PATH = "./saved_models/gold_gru_RNN.keras"

    if os.path.exists(MODEL_PATH):
        print("Model found. Loading the trained model...")
        model = load_model(MODEL_PATH)
        history = None

    else:
        print("No saved model found. Training a new model...")

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            min_delta=0.001,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        )

        history = model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_validation, y_validation),
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )

        model.save(MODEL_PATH)
        print("Model saved successfully.")

    return model, history


# ===========================================
# 5- Evaluate
# ===========================================
def evaluate(model, X_test, y_test, scaler):

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nTest MSE: {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")

    # Predict scaled values
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Convert back to original Log-Return scale
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    return y_pred, y_actual


# ===========================================
# 6- show results
# ===========================================


def results(y_pred, y_actual, history):
    # ==========================================
    # Regression Metrics
    # ==========================================

    mae = mean_absolute_error(y_actual, y_pred)

    mse = mean_squared_error(y_actual, y_pred)

    rmse = np.sqrt(mse)

    print("\n========== Test Results ==========")

    print(f"MAE:  {mae:.6f}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")

    # ==========================================
    # Direction Accuracy
    # ==========================================

    actual_direction = np.sign(y_actual)
    predicted_direction = np.sign(y_pred)

    direction_accuracy = np.mean(actual_direction == predicted_direction)

    print(f"Direction Accuracy: {direction_accuracy * 100:.2f}%")

    # ==========================================
    # Actual vs Prediction
    # ==========================================

    plt.figure(figsize=(14, 6))

    plt.plot(y_actual, label="Actual", linewidth=1)

    plt.plot(y_pred, label="Prediction", linestyle="--", linewidth=1.5)

    plt.title("GRU Prediction vs Actual Log Return")

    plt.xlabel("Test Samples")
    plt.ylabel("Log Return")

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ==========================================
    # Learning Curve
    # ==========================================

    if history is not None:
        plt.figure(figsize=(12, 5))

        plt.plot(history.history["loss"], label="Training Loss")

        plt.plot(history.history["val_loss"], label="Validation Loss")

        plt.title("Model Learning Curve")

        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")

        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    else:
        print("\nTraining history is not available (model was loaded from disk).")


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
    stationary_dataset = make_stationary_dataset(dataset)
    X_train, X_validation, X_test, y_train, y_validation, y_test, scaler = (
        RNN_sequence_creation(stationary_dataset["Log_Return"], look_back=12)
    )

    # log_return_ADF(stationary_dataset)
    # log_return_autocorrelation(stationary_dataset)

    # 3. Modeling
    model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    # 4. train or load save model
    model, history = train_rnn(model, X_train, y_train, X_validation, y_validation)

    # 5. evaluate
    y_pred, y_actual = evaluate(model, X_test, y_test, scaler)

    # 6. show results
    results(y_pred, y_actual, history)
