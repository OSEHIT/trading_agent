import httpx
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type

class ModelInspectorInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol to inspect (e.g., 'AAPL')")

class ModelInspector(BaseTool):
    name = "model_inspector"
    description = "Queries the LSTM model to get technical price predictions."
    args_schema: Type[BaseModel] = ModelInspectorInput

    def _run(self, ticker: str) -> str:
        try:
            # Synchronous request to the serving API
            # In a real async agent, we might use _arun
            with httpx.Client() as client:
                response = client.post(
                    "http://stock-api:8080/predict",
                    json={"ticker": ticker},
                    timeout=30.0
                )
                if response.status_code == 200:
                    data = response.json()
                    return (
                        f"Prediction for {ticker}:\n"
                        f"- Signal: {data.get('signal')}\n"
                        f"- Target Price: ${data.get('target_price', 0):.2f}\n"
                        f"- Confidence: {data.get('confidence', 0)}\n"
                        f"- Rationale: {data.get('rationale')}"
                    )
                else:
                    return f"Error querying model: {response.text}"
        except Exception as e:
            return f"Failed to connect to model service: {str(e)}"

    def _arun(self, ticker: str):
        raise NotImplementedError("ModelInspector does not support async")
