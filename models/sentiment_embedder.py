# models/sentiment_embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np

class SentimentEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_headlines(self, headlines):
        if not headlines:
            return np.zeros((1, 384))  # Fallback zero vector if no headlines
        embeddings = self.model.encode(headlines)
        avg_embedding = np.mean(embeddings, axis=0)  # average over headlines
        return avg_embedding.reshape(1, -1)
