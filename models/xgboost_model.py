from xgboost import XGBClassifier
import numpy as np

class XGBoostPricePredictor:
    def __init__(self, scaler, prepared_data, preloaded_model=None):
        self.scaler = scaler
        self.prepared_data = prepared_data
        self.model = preloaded_model or XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_next_5_weeks(self):
        preds = []
        confs = []
        for i in range(5):
            X_pred = self.scaler.transform([self.prepared_data[-1]])
            prob = self.model.predict_proba(X_pred)[0]
            prediction = int(prob[1] > 0.5)
            confidence = round(prob[1] * 100, 2)
            preds.append(prediction)
            confs.append(confidence)
        return preds, confs


