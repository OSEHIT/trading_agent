import sys
import os
import pandas as pd
sys.path.append(os.getcwd())

from src.agents.input_agent import InputAgent
from src.model.core_model import CoreModel
from src.agents.output_agent import OutputAgent
from src.monitoring.guardian import DriftGuardian

def test_pipeline():
    print("--- Starting Pipeline Verification ---")
    
    # 1. Initialize Components
    print("[1] Initializing Agents...")
    input_agent = InputAgent()
    core_model = CoreModel()
    output_agent = OutputAgent()
    guardian = DriftGuardian()
    
    ticker = "MSFT" # Use MSFT for test

    # 2. Input Agent
    print(f"[2] Input Agent: Validating {ticker}...")
    if not input_agent.validate_ticker(ticker):
        print("❌ Invalid Ticker")
        return
    
    sentiment = input_agent.categorize_market_sentiment(ticker)
    print(f"    Sentiment: {sentiment}")

    # 3. Core Model
    print("[3] Core Model: Fetching Data & predicting...")
    try:
        data = input_agent.get_data_for_training(ticker)
        if data.empty:
            print("❌ No data fetched")
            return
            
        print(f"    Fetched {len(data)} rows.")
        
        # Train small model if not exists to ensure predict works
        # In real flow, we might load, but here we force a quick train for verification
        print("    Triggering micro-training for verification...")
        X, y, _ = core_model.prepare_data(data)
        # Train only on small subset for speed
        core_model.train(X[-100:], y[-100:])
        
        # Predict
        recent_data = data['Close'].values[-core_model.look_back:]
        prediction = core_model.predict(recent_data)
        print(f"    Prediction: {prediction}")
        
    except Exception as e:
        print(f"❌ Model Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Drift Guardian
    print("[4] Drift Guardian: Checking Drift...")
    # Mock some current data
    current_data = data.tail(30)
    drift_result = guardian.check_drift(data, current_data)
    print(f"    Drift Result: {drift_result['drift_detected']}")

    # 5. Output Agent
    print("[5] Output Agent: Synthesizing...")
    current_price = data['Close'].iloc[-1].item()
    result = output_agent.synthesize(sentiment, prediction, drift_result, current_price)
    
    print("\n--- Final Report ---")
    print(f"Signal: {result['signal']}")
    print(f"Target: {result['target_price']}")
    print(f"Rationale: {result['rationale']}")
    
    if result['signal'] in ["BUY", "SELL", "HOLD"]:
         print("\n✅ VERIFICATION PASSED")
    else:
         print("\n❌ VERIFICATION FAILED (Invalid Signal)")

if __name__ == "__main__":
    test_pipeline()
