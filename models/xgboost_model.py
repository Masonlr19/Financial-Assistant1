import os
import joblib
import xgboost as xgb

class XGBoostPricePredictor:
    def __init__(self, scaler, prepared_data, model_path="models/xgboost_model.pkl"):
        self.scaler = scaler
        self.prepared_data = prepared_data
        self.model_path = model_path
        self.model = None

    def train(self, X, y):
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            n_estimators=100,
            random_state=42
        )
        self.model.fit(X, y)
        # Save the model after training
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No saved model found at {self.model_path}")
        self.model = joblib.load(self.model_path)

    def predict_next_5_weeks(self):
        if self.model is None:
            self.load_model()

        # Prepare features for the next 5 Fridays (this is a placeholder)
        # You should implement feature prep matching your model's training
        X_next_5 = self.prepared_data.get_next_5_weeks_features()

        preds_proba = self.model.predict_proba(X_next_5)[:, 1]
        preds = (preds_proba >= 0.5).astype(int)  # 1 if predicted increase, else 0

        # Confidence from probability (scale 0-100)
        confs = (preds_proba * 100).round(2)

        return preds.tolist(), confs.tolist()




