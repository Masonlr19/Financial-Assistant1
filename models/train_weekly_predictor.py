# models/train_weekly_predictor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from services.feature_engineering import prepare_features
from datetime import datetime, timedelta

class WeeklyPredictorTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary labels: 1 if next week's close > this week's close, else 0.
        Assumes data sorted by date ascending.
        """
        df = df.copy()
        df['next_close'] = df['close'].shift(-1)
        df['target'] = (df['next_close'] > df['close']).astype(int)
        return df['target'][:-1]  # last row has no next close, so drop

    def prepare_training_data(self, raw_data: dict):
        df = prepare_features(raw_data)
        labels = self.generate_labels(df)
        
        features = df.drop(columns=['date', 'close']).iloc[:-1]  # drop last row with no label

        # Scale features
        X_scaled = self.scaler.fit_transform(features)

        return X_scaled, labels.values

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_next_5_weeks(self, last_data: pd.DataFrame):
        """
        Predict the next 5 Fridays.
        last_data: DataFrame with latest features, including date, scaled.
        Returns predictions and confidence scores.
        """
        preds = []
        confs = []
        current_features = last_data.drop(columns=['date', 'close']).values.reshape(1, -1)

        for _ in range(5):
            pred_prob = self.model.predict_proba(current_features)[0][1]
            pred = 1 if pred_prob >= 0.5 else 0
            preds.append(pred)
            confs.append(round(pred_prob * 100, 2))

            # For simplicity, simulate next week by updating date + features here
            # Ideally, you would recalculate features with new projected prices and dates
            # For now, just reuse current_features (placeholder)

        return preds, confs

