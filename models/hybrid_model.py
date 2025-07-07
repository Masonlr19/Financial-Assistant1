# models/hybrid_model.py

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model

class HybridPricePredictor:
    def __init__(self, seq_len, num_price_features, sentiment_embedding_dim, model_path="hybrid_model.keras"):
        self.seq_len = seq_len
        self.num_price_features = num_price_features
        self.sentiment_embedding_dim = sentiment_embedding_dim
        self.model_path = model_path
        self.model = self.build_model()

    def build_model(self):
        price_input = Input(shape=(self.seq_len, self.num_price_features), name="price_input")
        price_lstm = LSTM(64, return_sequences=False)(price_input)
        price_dropout = Dropout(0.2)(price_lstm)

        sentiment_input = Input(shape=(self.seq_len, self.sentiment_embedding_dim), name="sentiment_input")
        sentiment_lstm = LSTM(32, return_sequences=False)(sentiment_input)
        sentiment_dropout = Dropout(0.2)(sentiment_lstm)

        concat = Concatenate()([price_dropout, sentiment_dropout])
        dense1 = Dense(64, activation="relu")(concat)
        dense2 = Dense(32, activation="relu")(dense1)
        output = Dense(1, activation="sigmoid")(dense2)

        model = Model(inputs=[price_input, sentiment_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, price_seq, sentiment_seq, labels, epochs=10, batch_size=32):
        self.model.fit([price_seq, sentiment_seq], labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        self.model.save(self.model_path)

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)

    def predict(self, price_seq, sentiment_seq):
        probas = self.model.predict([price_seq, sentiment_seq]).flatten()
        preds = (probas >= 0.5).astype(int)
        confs = (probas * 100).round(1)
        return preds.tolist(), confs.tolist()
