import requests

class TradierClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tradier.com/v1/markets"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

    def get_historical_data(self, symbol: str, interval: str = 'daily', start_date: str = '2024-01-01', end_date: str = '2024-12-31'):
        params = {
            "symbol": symbol,
            "interval": interval,
            "start": start_date,
            "end": end_date
        }
        response = requests.get(f"{self.base_url}/history", headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
