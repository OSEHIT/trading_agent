from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class NewsScraperInput(BaseModel):
    query: str = Field(description="The search query for financial news (e.g., 'AAPL stock news')")

class NewsScraper(BaseTool):
    name = "news_scraper"
    description = "Searches for real-time financial news and market sentiment."
    args_schema: Type[BaseModel] = NewsScraperInput
    
    # Instance of the actual search runner
    search = DuckDuckGoSearchRun()

    def _run(self, query: str) -> str:
        try:
            return self.search.run(query)
        except Exception as e:
            return f"Failed to search news: {str(e)}"

    def _arun(self, query: str):
        raise NotImplementedError("NewsScraper does not support async yet")
