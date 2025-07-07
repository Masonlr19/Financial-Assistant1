import requests

class NewsApiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"

    def get_news(self, query: str, page_size: int = 20):
        params = {
            "q": query,
            "pageSize": page_size,
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()
