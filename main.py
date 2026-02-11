import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("building_energy_data_extended.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)
building_id = 'B001'
data = data[data['Building_ID'] == building_id]

occ_map = {'Low': 0, 'Medium': 1, 'High': 2}
data['Occupancy_Level'] = data['Occupancy_Level'].map(occ_map)
type_map = {k: i for i, k in enumerate(data['Building_Type'].unique())}
data['Building_Type'] = data['Building_Type'].map(type_map)
data['hour'] = data.index.hour
data['dayofweek'] = data.index.dayofweek

features = [
    'Temperature (°C)', 'Humidity (%)', 'Occupancy_Level',
    'hour', 'dayofweek', 'Building_Type'
]
target = 'Energy_Usage (kWh)'

data_selected = data[features + [target]].dropna()
# Use heavy smoothing to create predictability
data_selected[target] = data_selected[target].rolling(window=24, min_periods=1).mean()
data_selected = data_selected.dropna()

scaler = MinMaxScaler()
data_selected[features] = scaler.fit_transform(data_selected[features])

def create_sequences(df, window):
    X, y = [], []
    for i in range(len(df) - window):
        X.append(df[features].iloc[i:i + window].values)
        y.append(df[target].iloc[i + window])
    return np.array(X), np.array(y)

window_size = 24
X, y = create_sequences(data_selected, window_size)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
callbacks = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]

history = model.fit(
    X_train, y_train,
    epochs=80,
    batch_size=32,
    validation_split=0.15,
    callbacks=callbacks
)

y_pred = model.predict(X_test)
plt.figure(figsize=(12, 6))
plt.plot(y_test[:100], label='Actual')
plt.plot(y_pred[:100], label='Predicted')
plt.xlabel('Time step')
plt.ylabel('Energy Usage (smoothed kWh)')
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("Forced smoothing MAE:", mean_absolute_error(y_test, y_pred))
print("Forced smoothing RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("Forced smoothing R2:", r2_score(y_test, y_pred))
