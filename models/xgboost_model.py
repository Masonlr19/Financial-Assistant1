from xgboost import XGBClassifier
import numpy as np

class XGBoostPricePredictor:
    def __init__(self, scaler, data):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.scaler = scaler
        self.data = data

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_next_5_weeks(self):
        preds = []
        confs = []

        last_features = self.data[['return_1w', 'return_2w', 'ma_3w', 'ma_5w']].iloc[-1].values.reshape(1, -1)
        last_close = self.data['close'].iloc[-1]

        for _ in range(5):
            last_features_scaled = self.scaler.transform(last_features)
            prob = self.model.predict_proba(last_features_scaled)[0][1]
            preds.append(1 if prob > 0.5 else 0)
            confs.append(int(prob * 100))

            avg_return = np.mean(self.data['return_1w'])
            next_return_1w = avg_return if preds[-1] == 1 else -avg_return
            next_return_2w = last_features[0][0]
            next_ma_3w = (last_features[0][2] * 2 + last_close * (1 + next_return_1w)) / 3
            next_ma_5w = (last_features[0][3] * 4 + last_close * (1 + next_return_1w)) / 5

            last_features = np.array([[next_return_1w, next_return_2w, next_ma_3w, next_ma_5w]])

        return preds, confs

def __init__(self, scaler, prepared_data, preloaded_model=None):
    self.scaler = scaler
    self.prepared_data = prepared_data
    self.model = preloaded_model  # Use loaded model if provided

