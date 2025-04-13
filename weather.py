import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO)

class AdvancedWeatherPredictor:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()

    def load_or_generate_data(self, data_path=None):
        if data_path:
            try:
                df = pd.read_csv(data_path)
                logging.info("Data loaded from file.")
                return df
            except Exception as e:
                logging.warning(f"Error loading data: {e}")

        # Generate synthetic weather data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1500)
        day_of_year = np.array([d.dayofyear for d in dates])
        day_sin = np.sin(2 * np.pi * day_of_year / 365)
        day_cos = np.cos(2 * np.pi * day_of_year / 365)

        temp = 15 + 10 * day_sin + np.random.normal(0, 2, len(dates))
        humidity = 60 + 20 * day_cos + np.random.normal(0, 5, len(dates))
        wind_speed = 10 + 3 * np.random.random(len(dates))
        pressure = 1013 + np.random.normal(0, 5, len(dates))

        df = pd.DataFrame({
            'date': dates,
            'temperature': temp,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'pressure': pressure,
            'day_sin': day_sin,
            'day_cos': day_cos
        })

        for i in range(1, 4):
            df[f'temp_lag_{i}'] = df['temperature'].shift(i)
            df[f'humidity_lag_{i}'] = df['humidity'].shift(i)

        df.dropna(inplace=True)
        logging.info("Synthetic data generated.")
        return df

    def prepare_data(self, df, target='temperature', future_days=1):
        df[f'{target}_future'] = df[target].shift(-future_days)
        df.dropna(inplace=True)

        feature_cols = [col for col in df.columns if col not in ['date', target, f'{target}_future']]
        X = df[feature_cols].values
        y = df[f'{target}_future'].values
        return X, y, feature_cols

    def train_model(self, X, y):
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'lstm':
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

            model = Sequential()
            model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
            self.model = model
        else:
            self.model = RandomForestRegressor(n_estimators=150, random_state=42)

        if self.model_type != 'lstm':
            self.model.fit(X_train, y_train)

        # Prediction & Evaluation
        if self.model_type == 'lstm':
            y_pred = self.model.predict(X_test).flatten()
        else:
            y_pred = self.model.predict(X_test)

        self.evaluate(y_test, y_pred)
        return self.model

    def evaluate(self, y_true, y_pred):
        print(f"Mean Squared Error: {mean_squared_error(y_true, y_pred):.3f}")
        print(f"Mean Absolute Error: {mean_absolute_error(y_true, y_pred):.3f}")
        print(f"R2 Score: {r2_score(y_true, y_pred):.3f}")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.xlabel("Actual Temp")
        plt.ylabel("Predicted Temp")
        plt.title("Actual vs Predicted Temperature")
        plt.show()

    def predict(self, X_new):
        X_scaled = self.scaler.transform(X_new)
        if self.model_type == 'lstm':
            X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        return self.model.predict(X_scaled)

    def save_model(self, path='weather_model.pkl'):
        if self.model_type != 'lstm':
            joblib.dump((self.model, self.scaler), path)
            logging.info("Model saved successfully.")

# Main Execution
if __name__ == "__main__":
    predictor = AdvancedWeatherPredictor(model_type='lstm')  # 'linear', 'random_forest', 'lstm'
    data = predictor.load_or_generate_data()
    print(data.head())

    X, y, features = predictor.prepare_data(data)
    print(f"Feature columns: {features}")

    predictor.train_model(X, y)

    prediction = predictor.predict(X[:1])
    print(f"Predicted Temp: {prediction[0]}")

    predictor.save_model()
