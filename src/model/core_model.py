import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import joblib
from src.utils.config import LOOK_BACK, EPOCHS, BATCH_SIZE, MODEL_PATH, DATA_PATH

class CoreModel:
    """
    Role: Quantitative Model (LSTM)
    Responsibilities: Train on historical data, Predict future price.
    """
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.look_back = LOOK_BACK
        # Ensure directories exist
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    def prepare_data(self, data: pd.DataFrame):
        """Prepares data for LSTM (Scaling + Sliding Window)."""
        dataset = data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(dataset)
        
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y), scaled_data

    def build_model(self, input_shape):
        """Builds the LSTM architecture."""
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model

    def train(self, X_train, y_train):
        """Trains the model."""
        if self.model is None:
            self.build_model((X_train.shape[1], 1))
        
        print("Starting training...")
        self.model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        self.save()
        print("Training complete and model saved.")

    def predict(self, recent_data):
        """
        Predicts the next price based on recent data.
        recent_data: List or Array of last LOOK_BACK prices (raw).
        """
        if self.model is None:
            if not self.load():
                raise Exception("Model not trained or loaded.")

        # recent_data should be exactly look_back length
        recent_data = np.array(recent_data).reshape(-1, 1)
        scaled_input = self.scaler.transform(recent_data)
        
        # Reshape for LSTM [1, look_back, 1]
        X_input = np.array([scaled_input])
        X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))
        
        predicted_scaled = self.model.predict(X_input)
        predicted_price = self.scaler.inverse_transform(predicted_scaled)
        
        return float(predicted_price[0][0])

    def save(self):
        """Saves model and scaler."""
        if self.model:
            self.model.save(MODEL_PATH)
            # Save scaler alongside
            scaler_path = MODEL_PATH.replace(".h5", "_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)

    def load(self):
        """Loads model and scaler."""
        if os.path.exists(MODEL_PATH):
            try:
                self.model = load_model(MODEL_PATH)
                scaler_path = MODEL_PATH.replace(".h5", "_scaler.pkl")
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False
