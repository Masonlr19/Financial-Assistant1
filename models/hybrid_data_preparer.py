# models/hybrid_data_preparer.py

import numpy as np
from models.sentiment_embedder import SentimentEmbedder

class HybridDataPreparer:
    def __init__(self, seq_len=10):
        self.seq_len = seq_len
        self.sentiment_embedder = SentimentEmbedder()

    def prepare_price_sequences(self, historical_data):
        # Example assumes 'close', 'volume', etc. in historical_data
        # Returns shape: (num_samples, seq_len, num_price_features)
        df = historical_data.copy()
        features = ['close', 'volume']  # Add more features as needed
        data = df[features].values
        price_sequences = []

        for i in range(len(data) - self.seq_len):
            price_sequences.append(data[i:i + self.seq_len])

        return np.array(price_sequences)

    def prepare_sentiment_sequences(self, headlines_per_week):
        # headlines_per_week: List of list of headlines for each week
        embeddings = []
        for headlines in headlines_per_week:
            week_embeds = self.sentiment_embedder.embed_headlines(headlines)
            embeddings.append(week_embeds)
        return np.array(embeddings)

    def prepare_labels(self, historical_data):
        # Binary label: 1 if next week's close > this week's close
        closes = historical_data['close'].values
        labels = (closes[self.seq_len:] > closes[self.seq_len - 1:-1]).astype(int)
        return labels
