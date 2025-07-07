from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

class XGBoostPricePredictor:
    def __init__(self, scaler, prepared_data):
        self.scaler = scaler
        self.prepared_data = prepared_data
        self.model = None
        self.metrics = {}

    def train(self, X, y):
        # Split train/val (80/20)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_val)

        # Compute metrics
        self.metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1_score": f1_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_pred),
            "confusion_matrix": confusion_matrix(y_val, y_pred).tolist()
        }

    def predict_next_5_weeks(self):
        # Example dummy implementation - replace with actual prediction logic
        # Should return predictions (0/1) and confidence scores (0-100)
        import random
        preds = [random.choice([0, 1]) for _ in range(5)]
        confs = [random.randint(50, 100) for _ in range(5)]
        return preds, confs



