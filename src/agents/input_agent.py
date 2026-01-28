import yfinance as yf
import pandas as pd

class InputAgent:
    """
    Role: Junior Analyst
    Responsibilities: Validate ticker, fetch macro context.
    """
    
    def validate_ticker(self, ticker: str) -> bool:
        """Checks if ticker exists by attempting a quick fetch."""
        if not ticker:
            return False
        try:
            ticker_obj = yf.Ticker(ticker)
            # Fast check: history for 1d
            hist = ticker_obj.history(period="1d")
            return not hist.empty
        except Exception:
            return False

    def categorize_market_sentiment(self, ticker: str):
        """
        Simulates retrieving market sentiment for the ticker.
        In a real app, this might scrape news or use an LLM.
        """
        # Mock logic based on recent price movement
        try:
            data = yf.download(ticker, period="5d", progress=False)
            if data.empty:
                return "Neutral (No Data)"
            
            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            
            # Simple scalar comparison if series, else direct
            if hasattr(start_price, "item"):
                start_price = start_price.item()
            if hasattr(end_price, "item"):
                end_price = end_price.item()

            if end_price > start_price * 1.02:
                return "Bullish (Strong Uptrend)"
            elif end_price < start_price * 0.98:
                return "Bearish (Selling Pressure)"
            else:
                return "Neutral (Choppy)"
        except Exception as e:
            return f"Error fetching sentiment: {str(e)}"

    def get_data_for_training(self, ticker: str):
        """Fetches long-term data for training."""
        # Using 2y data for decent training size
        data = yf.download(ticker, period="2y", interval="1d", progress=False)
        return data
