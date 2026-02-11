import pandas as pd

data = pd.read_csv("building_energy_data_extended.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)
building_id = 'B001'
type_map = {k: i for i, k in enumerate(data['Building_Type'].unique())}
data['Building_Type'] = data['Building_Type'].map(type_map)
data = data[data['Building_ID'] == building_id]
data['hour'] = data.index.hour
data['dayofweek'] = data.index.dayofweek
data['Energy_Usage (kWh)_smoothed'] = data['Energy_Usage (kWh)'].rolling(window=6, min_periods=1).mean()
occ_map = {'Low': 0, 'Medium': 1, 'High': 2}
data['Occupancy_Level'] = data['Occupancy_Level'].map(occ_map)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Use your data and already-smoothed target
df = data.copy()

# Create lag features
for lag in range(1, 25):  # use previous 24 hours
    df[f'energy_lag_{lag}'] = df['Energy_Usage (kWh)_smoothed'].shift(lag)

# Drop rows with NA (from shifting)
df = df.dropna()

# FEATURES for RF: All original plus lags
features_rf = [
    'Temperature (°C)',
    'Humidity (%)',
    'Occupancy_Level',
    'hour',
    'dayofweek',
    'Building_Type'
] + [f'energy_lag_{lag}' for lag in range(1, 25)]

X = df[features_rf]
y = df['Energy_Usage (kWh)_smoothed']

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred_rf = rf.predict(X_test)

# Plot result
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(y_test.values[:100], label='Actual')
plt.plot(y_pred_rf[:100], label='RF Predicted')
plt.legend()
plt.xlabel('Time step')
plt.ylabel('Energy (smoothed)')
plt.title('Random Forest Energy Prediction')
plt.show()

# Metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)
print("Random Forest MAE:", mae_rf)
print("Random Forest RMSE:", rmse_rf)
print("Random Forest R2:", r2_rf)
