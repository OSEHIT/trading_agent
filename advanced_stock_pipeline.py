import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    # 0.7.20 structure: Report is top-level, DataDriftPreset is likely in presets
    from evidently import Report
    try:
        from evidently.presets import DataDriftPreset
    except ImportError:
        # Last resort check
        from evidently.metric_preset import DataDriftPreset
import datetime

# --- Configuration ---
LOOK_BACK = 60
TEST_SIZE = 0.2
EPOCHS = 10  # Reduced for demonstration speed
BATCH_SIZE = 32

class InputAgent:
    """
    Role: Junior Analyst
    Responsibilities: Validate ticker, fetch macro context.
    """
    def categorize_market_sentiment(self, ticker):
        # Mocking an API call to a news sentiment service
        # In a real app, this would query NewsAPI or AlphaVantage
        print(f"[{self.__class__.__name__}] Analyzing macro context for {ticker}...")
        return {
            "sentiment": "Neutral-Optimistic",
            "key_factors": ["Tech sector recovery", "Interest rate pauses"],
            "ticker_valid": True
        }

class CoreModel:
    """
    Role: Quantitative Model (LSTM)
    Responsibilities: Train on historical data, Predict future price.
    """
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def download_data(self, ticker):
        print(f"[{self.__class__.__name__}] Downloading data for {ticker}...")
        # Get 2 years of data
        data = yf.download(ticker, period="2y", interval="1d", progress=False)
        if len(data) == 0:
            raise ValueError(f"No data found for {ticker}")
        return data

    def prepare_data(self, data):
        # Focus on 'Close' price
        dataset = data[['Close']].values
        
        # Scaling
        self.scaler.fit(dataset)
        scaled_data = self.scaler.transform(dataset)
        
        X, y = [], []
        for i in range(LOOK_BACK, len(scaled_data)):
            X.append(scaled_data[i-LOOK_BACK:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y, dataset

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train(self, X_train, y_train):
        print(f"[{self.__class__.__name__}] Training LSTM model...")
        self.model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    def predict(self, X_last_sequence):
        # Helper to predict next day
        X_input = X_last_sequence.reshape(1, LOOK_BACK, 1)
        prediction_scaled = self.model.predict(X_input, verbose=0)
        prediction = self.scaler.inverse_transform(prediction_scaled)
        return prediction[0][0]

class DriftGuardian:
    """
    Role: Reliability Monitor (Evidently AI)
    Responsibilities: Check for Data Drift between Train set and recent inference data.
    """
    def check_drift(self, reference_data, current_data):
        print(f"[{self.__class__.__name__}] Running Data Drift Check...")
        
        # Convert to DataFrame for Evidently
        ref_df = pd.DataFrame(reference_data, columns=['close'])
        curr_df = pd.DataFrame(current_data, columns=['close'])
        
        # Create a Report using DataDriftPreset
        # We are checking if the distribution of 'close' prices has shifted significantly
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=ref_df, current_data=curr_df)
        
        # Extract a simplified result
        # Note: In production, checking 'volatility' specifically often involves custom metrics 
        # or checking the 'std' drift. DataDriftPreset covers general distribution drift.
        # debug: print available methods
        print("DEBUG: drift_report methods:", dir(drift_report))
        # as_dict() might be missing in newer versions, use json()
        import json
        # Try to find a valid method to get data
        try:
            json_output = drift_report.json()
            drift_result = json.loads(json_output)
        except AttributeError:
             print("DEBUG: json() failed, trying as_dict again or other?")
             # Fallback if neither exist?
             return {"drift_detected": False, "full_report": "Error getting report"} 
        dataset_drift = drift_result['metrics'][0]['result']['dataset_drift']
        
        return {
            "drift_detected": dataset_drift,
            "full_report": drift_result
        }

class OutputAgent:
    """
    Role: Financial Expert
    Responsibilities: Synthesize inputs and check reliability.
    """
    def synthesize(self, context, prediction, drift_info, current_price):
        print(f"\n[{self.__class__.__name__}] Synthesizing Final Report...")
        
        drift_detected = drift_info['drift_detected']
        trust_model = not drift_detected
        
        expected_change = ((prediction - current_price) / current_price) * 100
        
        # --- Prompt Simulation for Synthesis ---
        system_prompt = f"""
        ACT AS: Senior Financial Advisor.
        
        CONTEXT:
        - Ticker Macro Context: {context}
        - Current Price: ${current_price:.2f}
        - LSTM Prediction (Next Day): ${prediction:.2f} ({expected_change:+.2f}%)
        
        RELIABILITY CHECK (The Guardian):
        - Data Drift Detected: {drift_detected}
        - Trust Model? {'YES' if trust_model else 'NO - EXTREME CAUTION'}
        """
        
        print("\n--- INTERNAL AGENT THOUGHT PROCESS ---")
        print(system_prompt)
        print("--------------------------------------")
        
        # Final Decision Logic
        if not trust_model:
            return f"[CAUTION]: Significant market shift detected (Data Drift). The model's training data may no longer be relevant. We recommend IGNORING the technical prediction of ${prediction:.2f} and waiting for market stability. Rely on fundamental analysis only: {context['sentiment']}."
        else:
            action = "BUY" if expected_change > 1 else "SELL" if expected_change < -1 else "HOLD"
            return f"[RECOMMENDATION]: {action}. Model is operating within normal parameters. Predicted move: {expected_change:.2f}%. Fundamentals ({context['sentiment']}) align/diverge. Target: ${prediction:.2f}."

def main_pipeline(ticker):
    print(f"--- Starting Agent-Sandwich Pipeline for {ticker} ---")
    
    # 1. Input Agent
    input_agent = InputAgent()
    context = input_agent.categorize_market_sentiment(ticker)
    if not context['ticker_valid']:
        print("Invalid Ticker. Aborting.")
        return

    # 2. Core Model (Training & Prediction)
    model = CoreModel()
    raw_data = model.download_data(ticker)
    if raw_data.empty: return

    X, y, full_dataset = model.prepare_data(raw_data)
    
    # Split for standard training
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build & Train
    model.build_model((X_train.shape[1], 1))
    model.train(X_train, y_train)
    
    # Predict next day using the very last sequence available
    # Note: real-time would use the latest window from today.
    last_sequence = X[-1]
    prediction = model.predict(last_sequence)
    
    # 3. The Guardian (Drift Check)
    # We compare Training Data Distribution vs Recent Test Data Distribution
    # to see if the market regime has changed.
    guardian = DriftGuardian()
    
    # Use raw price data for drift check to be interpretable
    train_prices = full_dataset[:split_idx]
    recent_prices = full_dataset[split_idx:]
    
    drift_output = guardian.check_drift(train_prices, recent_prices)
    
    # 4. Output Agent
    current_price = float(raw_data['Close'].iloc[-1])
    expert = OutputAgent()
    final_report = expert.synthesize(context, prediction, drift_output, current_price)
    
    print("\n" + "="*50)
    print("FINAL CLIENT REPORT:")
    print(final_report)
    print("="*50)

if __name__ == "__main__":
    # Test with a known ticker
    main_pipeline("MSFT")
