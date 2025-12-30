# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.random import set_seed

# Reproducibility
set_seed(455)
np.random.seed(455)

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
dataset = pd.read_csv(
    url,
    index_col="Month",
    parse_dates=["Month"],
)

print(dataset.head())
print(dataset.describe())
print(dataset.isna().sum())

# Train-test split years
tstart = "1949"
tend = "1958"

# --- Plot train/test split ---
def train_test_plot(dataset, tstart, tend):
    plt.figure(figsize=(16, 4))
    dataset.loc[f"{tstart}":f"{tend}", "Passengers"].plot(label=f"Train ({tstart}-{tend})")
    dataset.loc[f"{tend}":, "Passengers"].plot(label=f"Test ({tend} and beyond)")
    plt.legend()
    plt.title("Pasajeros en aerolineas")
    plt.xlabel("Fecha")
    plt.ylabel("Pasajeros")
    plt.show()

train_test_plot(dataset, tstart, tend)

# --- Split dataset ---
def train_test_split_custom(dataset, tstart, tend):
    train = dataset.loc[f"{tstart}":f"{tend}", "Passengers"].values
    test = dataset.loc[f"{tend}":, "Passengers"].values
    return train, test

training_set, test_set = train_test_split_custom(dataset, tstart, tend)

# --- Normalize ---
sc = MinMaxScaler(feature_range=(0, 1))
training_set = training_set.reshape(-1, 1)
training_set_scaled = sc.fit_transform(training_set)

# --- Helper to split sequence into samples ---
def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence) - n_steps):
        X.append(sequence[i:i+n_steps])
        y.append(sequence[i+n_steps])
    return np.array(X), np.array(y)

# Parameters
n_steps = 60
features = 1

# Prepare training data
X_train, y_train = split_sequence(training_set_scaled, n_steps)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], features)

# --- LSTM Model ---
model_lstm = Sequential()
model_lstm.add(LSTM(units=125, activation="tanh", input_shape=(n_steps, features)))
model_lstm.add(Dense(units=1))
model_lstm.compile(optimizer="RMSprop", loss="mse")
model_lstm.summary()

# Train
model_lstm.fit(X_train, y_train, epochs=50, batch_size=32)

# --- Prepare test data ---
dataset_total = dataset["Passengers"]
inputs = dataset_total[len(dataset_total) - len(test_set) - n_steps:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test, y_test = split_sequence(inputs, n_steps)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], features)

# Predict
predicted_stock_price = model_lstm.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# --- Plot predictions ---
def plot_predictions(real, predicted):
    plt.figure(figsize=(10, 6))
    plt.plot(real, color="gray", label="Pasajeros Reales")
    plt.plot(predicted, color="red", label="Pasajeros Predichos")
    plt.title("Predicción de Pasajeros en Aerolíneas (LTSM)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

plot_predictions(test_set, predicted_stock_price)

# --- RMSE ---
def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is: {:.2f}".format(rmse))
    return rmse

return_rmse(test_set, predicted_stock_price)

# --- GRU Model ---
model_gru = Sequential()
model_gru.add(GRU(units=125, activation="tanh", input_shape=(n_steps, features)))
model_gru.add(Dense(units=1))
model_gru.compile(optimizer="RMSprop", loss="mse")
model_gru.summary()

# Train GRU
model_gru.fit(X_train, y_train, epochs=50, batch_size=32)

# GRU predictions
GRU_predicted_stock_price = model_gru.predict(X_test)
GRU_predicted_stock_price = sc.inverse_transform(GRU_predicted_stock_price)

plot_predictions(test_set, GRU_predicted_stock_price)
return_rmse(test_set, GRU_predicted_stock_price)

