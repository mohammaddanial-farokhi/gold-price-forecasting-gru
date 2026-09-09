# gold-price-forecasting-gru

<br>

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg)
![Keras](https://img.shields.io/badge/Keras-API-red.svg)
![GRU](https://img.shields.io/badge/Model-GRU-blueviolet.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-purple.svg)
![NumPy](https://img.shields.io/badge/NumPy-Computation-navy.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-lightblue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Preprocessing-yellow.svg)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Time%20Series-green.svg)
![Task](https://img.shields.io/badge/Task-Time%20Series%20Forecasting-success.svg)
![Dataset](https://img.shields.io/badge/Dataset-Monthly%20Gold%20Prices-purple.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

# 📈 Monthly Gold Price Forecasting Using GRU

This project implements a **Gated Recurrent Unit (GRU)** neural network for **monthly gold price time-series forecasting**.

The project focuses on the complete time-series machine learning workflow, starting from **Exploratory Data Analysis (EDA)** and statistical testing, followed by **stationarity transformation, preprocessing, sequence generation, GRU modeling, training, evaluation, and visualization**.

Instead of directly modeling the raw gold price, the project transforms the price series into **log returns** to obtain a more suitable representation for time-series modeling.

The model uses the previous **12 months of log-return data** to predict the next monthly log return.

> **Important:** This project is primarily an educational and experimental time-series forecasting project. It is **not intended to provide financial advice or trading signals**.

---

## 🎯 Project Objectives

The main objectives of this project are:

* Analyze long-term monthly gold price behavior.
* Perform exploratory time-series analysis.
* Investigate stationarity using the **Augmented Dickey-Fuller (ADF) test**.
* Analyze autocorrelation and partial autocorrelation.
* Examine long-term trends and monthly volatility.
* Transform the non-stationary price series into **log returns**.
* Prepare sequential data for recurrent neural network training.
* Build a GRU-based forecasting model.
* Evaluate prediction performance using regression metrics.
* Measure the model's ability to predict the **direction of monthly returns**.
* Visualize actual vs predicted values and the model learning curve.
* Automatically save and reload the trained model.

---

# 📂 Project Structure

```text
.
├── monthly.csv                         # Monthly gold price dataset
│
├── saved_models/
│   └── gold_gru_RNN.keras              # Saved trained GRU model
│
├── main.py                             # Main project script
│
├── requirements.txt                    # Python dependencies
│
└── README.md                           # Project documentation
```



---

## 📊 Dataset

The project uses a monthly gold price dataset stored in:

```text
monthly.csv
```

The dataset is expected to contain at least the following columns:

```text
Date,Price
```

Example:

```text
Date,Price
1833-01-01,19.30
1833-02-01,19.30
1833-03-01,19.30
...
```

The original dataset contains historical monthly gold prices beginning in the **19th century**.

For the modeling pipeline, observations before **1880** are excluded:

```python
dataset = dataset[dataset.index.year >= 1880]
```

This leaves the historical period from 1880 onward for preprocessing and modeling.

---
# 🔍 Exploratory Data Analysis

The project includes several EDA functions for understanding the historical gold price series before preprocessing and modeling.

### Price Series Analysis

The `time_series_plot()` function visualizes the historical gold price over time and helps identify long-term trends and major changes in the price series.

The project also uses moving averages through `plot_long_term_trend()` to examine long-term behavior:

* 12-month moving average
* 60-month moving average

### Stationarity Testing

Financial price series are often non-stationary. The project uses the **Augmented Dickey-Fuller (ADF) test** to examine the stationarity of the original gold price series.

```python
ADF_test(dataset)
```

If the ADF test produces a p-value greater than `0.05`, the function reports the series as non-stationary.

This analysis motivates the transformation of the raw price series into **log returns**, which are used as the input target for the GRU model.

### Additional EDA Functions

| Function                 | Purpose                              |
| ------------------------ | ------------------------------------ |
| `time_series_plot()`     | Visualizes the historical gold price |
| `check_missing()`        | Checks for missing values            |
| `decomposition()`        | Performs seasonal decomposition      |
| `ADF_test()`             | Tests price-series stationarity      |
| `autocorrelation()`      | Displays ACF and PACF plots          |
| `monthly_volatility()`   | Analyzes monthly percentage returns  |
| `plot_long_term_trend()` | Displays long-term moving averages   |


---

# 🔄 Data Preprocessing

The preprocessing pipeline converts the raw gold price series into a more suitable representation for the GRU model.

## 1. Filtering the Dataset

Only observations from 1880 onward are used:

```python
dataset = dataset[dataset.index.year >= 1880]
```

---

## 2. Log Return Transformation

The model does not directly predict the raw gold price.

Instead, the project calculates monthly log returns:

```python
dataset["Log_Return"] = np.log(
    dataset["Price"] / dataset["Price"].shift(1)
)
```

The log-return transformation is commonly used in financial time-series analysis because it represents the relative change between consecutive observations.

The first resulting observation is removed because it has no previous month:

```python
dataset.dropna(subset=["Log_Return"], inplace=True)
```

---

## 3. Train / Validation / Test Split

The resulting log-return series is divided chronologically into:

| Dataset    | Percentage |
| ---------- | ---------- |
| Training   | 70%        |
| Validation | 15%        |
| Testing    | 15%        |

The split is performed **chronologically**, rather than randomly.

This is important for time-series forecasting because future observations should not be used to train the model.

```text
Historical Data
│
├─────────────── 70% ───────────────┤
│             Training              │
│                                   │
├──────── 15% ────────┤
│     Validation      │
│                     │
├──────── 15% ────────┤
│        Test         │
└─────────────────────┘
                       → Time
```

---

# 🧩 Sequence Creation

The GRU model requires sequential input.

The project uses:

```python
look_back = 12
```

This means the model receives the previous **12 monthly log returns** as input when predicting the next value.

Conceptually:

```text
Month 1 ─┐
Month 2  │
Month 3  │
   ...   ├──→ GRU ──→ Month 13 prediction
Month 11 │
Month 12 ┘
```

The resulting input shape is:

```text
(samples, 12, 1)
```

where:

* `samples` = number of generated sequences
* `12` = look-back window
* `1` = single feature (`Log_Return`)

---

# 🧠 Model Architecture

The forecasting model is based on a **Gated Recurrent Unit (GRU)**.

GRUs are recurrent neural network architectures designed to capture patterns and dependencies in sequential data while being computationally simpler than some other recurrent architectures.

The model architecture is:

| Layer     | Configuration      | Purpose                         |
| --------- | ------------------ | ------------------------------- |
| `GRU`     | 64 units, `tanh`   | Learn temporal dependencies     |
| `Dense`   | 1024 units, `ReLU` | Learn nonlinear representations |
| `Dropout` | 0.4                | Reduce overfitting              |
| `Dense`   | 1 unit             | Predict next log return         |

The architecture is implemented as:

```python
model = Sequential(
    [
        GRU(64, activation="tanh", input_shape=input_shape),
        Dense(1024, activation="relu"),
        Dropout(0.4),
        Dense(1),
    ]
)
```

---

## ⚙️ Model Configuration

The model is compiled using:

| Parameter     | Value                       |
| ------------- | --------------------------- |
| Optimizer     | Adam                        |
| Learning Rate | `0.0005`                    |
| Loss Function | Mean Squared Error (`MSE`)  |
| Metric        | Mean Absolute Error (`MAE`) |

```python
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="mse",
    metrics=["mae"],
)
```

---

# 🚂 Training

The model is trained with:

| Parameter               | Value   |
| ----------------------- | ------- |
| Maximum Epochs          | `50`    |
| Batch Size              | `32`    |
| Early Stopping Patience | `10`    |
| Reduce LR Patience      | `3`     |
| LR Reduction Factor     | `0.5`   |
| Minimum Learning Rate   | `1e-6`  |
| Minimum Improvement     | `0.001` |

### Early Stopping

Early stopping monitors validation loss:

```python
EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    min_delta=0.001,
)
```

If the validation loss does not improve sufficiently for 10 consecutive epochs, training stops and the best weights are restored.

### Learning Rate Scheduling

The project also uses `ReduceLROnPlateau`:

```python
ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
)
```

When validation loss stops improving, the learning rate is reduced by a factor of 2.

---

# 💾 Model Saving and Loading

The trained model is automatically saved to:

```text
./saved_models/gold_gru_RNN.keras
```

The training function checks whether this file already exists.

If the model exists:

```text
Model found. Loading the trained model...
```

the saved model is loaded and training is skipped.

If the model does not exist, a new model is trained and saved.

This makes it possible to reuse the trained model without retraining it every time the script is executed.

---

# 📊 Evaluation

After training, predictions are generated on the test set.

Because the model operates on standardized data, predictions are converted back to the original **log-return scale** using the fitted scaler.

The project evaluates the model using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Direction Accuracy

---

## 📈 Regression Results

The final test results are:

| Metric                 |     Result |
| ---------------------- | ---------: |
| **MAE**                | `0.029686` |
| **MSE**                | `0.001429` |
| **RMSE**               | `0.037798` |
| **Direction Accuracy** | **58.87%** |

### Direction Accuracy

In addition to traditional regression metrics, the project evaluates whether the model correctly predicts the **direction of the next log return**.

The calculation is based on the sign of the actual and predicted returns:

```python
actual_direction = np.sign(y_actual)
predicted_direction = np.sign(y_pred)

direction_accuracy = np.mean(
    actual_direction == predicted_direction
)
```

The final Direction Accuracy is:

```text
58.87%
```

This means the model correctly identified the positive/negative direction of the test-set log return in approximately **58.87% of cases**.

> **Note:** Direction Accuracy above 50% does not automatically imply a profitable trading strategy. Transaction costs, slippage, risk management, position sizing, and out-of-sample validation would also need to be considered before drawing any trading conclusions.

---

# 📉 Results Visualization

The project generates an **Actual vs Prediction** plot:

```text
Actual Log Return
        vs
Predicted Log Return
```

This visualization allows the model's predictions to be compared directly against the actual test-set log returns.

The project also generates a **Learning Curve** showing:

* Training loss
* Validation loss

This can be used to inspect model convergence and potential overfitting.

---

# 🛠️ Utility Functions

The project is organized into several functional sections.

## EDA

```text
time_series_plot()
check_missing()
decomposition()
ADF_test()
autocorrelation()
monthly_volatility()
plot_long_term_trend()
```

## Preprocessing

```text
make_stationary_dataset()
RNN_sequence_creation()
log_return_ADF()
log_return_autocorrelation()
```

## Modeling

```text
build_gru_model()
```

## Training

```text
train_rnn()
```

## Evaluation

```text
evaluate()
results()
```

This structure keeps the workflow modular and makes individual stages easier to inspect or modify.

---

# 🚀 Usage

## 1. Install Dependencies

Create a Python environment and install the required libraries:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn statsmodels
```

Or use:

```bash
pip install -r requirements.txt
```

---

## 2. Prepare the Dataset

Place the dataset in the project root:

```text
monthly.csv
```

Make sure it contains:

```text
Date,Price
```

---

## 3. Run the Project

Execute:

```bash
python main.py
```

The script will:

```text
Load dataset
     ↓
Convert Date column
     ↓
Filter data from 1880 onward
     ↓
Calculate Log Returns
     ↓
Split data chronologically
     ↓
Scale the data
     ↓
Create sequences
     ↓
Build GRU model
     ↓
Train or load saved model
     ↓
Evaluate on test data
     ↓
Generate predictions
     ↓
Display results and plots
```

---

# 🔧 Running EDA Functions

The EDA functions are available but are commented out in the `__main__` section.

For example:

```python
# 1. EDA
# time_series_plot(dataset)
# check_missing(dataset)
# decomposition(dataset)
# ADF_test(dataset)
# autocorrelation(dataset)
# monthly_volatility(dataset)
# plot_long_term_trend(dataset)
```

Uncomment the function you want to run.

For example:

```python
ADF_test(dataset)
```

or:

```python
autocorrelation(dataset)
```

The same approach can be used for the log-return analysis:

```python
# log_return_ADF(stationary_dataset)
# log_return_autocorrelation(stationary_dataset)
```


---

# 🔧 Customization

Several important parameters can easily be modified.

### Look-back Window

Inside `RNN_sequence_creation()`:

```python
look_back = 12
```

This controls how many previous observations are used to predict the next return.

For example:

```python
look_back = 24
```

would use the previous 24 months.

---

### Train / Validation / Test Split

The current split is:

```python
train_size = int(total_size * 0.70)
validation_size = int(total_size * 0.15)
```

which produces:

```text
70% Training
15% Validation
15% Testing
```

---

### GRU Architecture

The model can be modified inside:

```python
build_gru_model()
```

For example, the number of GRU units, Dense units, and Dropout rate can be adjusted depending on the experiment.

---

### Training Parameters

The following parameters can also be changed:

```python
epochs = 50
batch_size = 32
learning_rate = 0.0005
```

along with the Early Stopping and learning-rate scheduling parameters.

---


### 2. Monthly Frequency

The dataset contains monthly observations.

Therefore, this model is designed for **monthly time-series forecasting**, not intraday or high-frequency prediction.

---

### 3. Forecasting Returns Rather Than Prices

The model predicts the next **log return**, not the exact future gold price.

This makes the task different from direct price forecasting.

---

### 4. Limited Predictive Signal

The final Direction Accuracy of:

```text
58.87%
```

shows that the model has some ability to capture directional behavior in this test set, but the result should **not** be interpreted as proof of a reliable trading edge.

Financial markets are highly noisy and influenced by many variables that are not represented in this dataset.

---

# 🔮 Possible Future Improvements

Several extensions could improve the project and provide more meaningful experiments:


* Add additional financial and economic features such as interest rates, inflation, and USD strength.
* Compare the GRU model with LSTM and Transformer-based architectures.
* Experiment with different look-back windows and model hyperparameters.
* Use walk-forward validation and proper backtesting for a more realistic evaluation.


---
## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

If you have ideas for improving the forecasting pipeline, feel free to open an **Issue** or submit a **Pull Request**.

---

## 📜 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project in accordance with the terms of the license.
