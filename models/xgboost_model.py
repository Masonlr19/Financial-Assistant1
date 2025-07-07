import os
import joblib
import numpy as np
from xgboost import XGBClassifier

class XGBoostPricePredictor:
    def __init__(self, scaler, prepared_data, model_path="xgboost_model.joblib", scaler_path="scaler.joblib"):
        """
        Args:
            scaler: A fitted scaler object (e.g., StandardScaler) for feature scaling
            prepared_data: Data used for preparing features (optional, depending on your code)
            model_path: File path to save/load the trained model
            scaler_path: File path to save/load the scaler
        """
        self.scaler = scaler
        self.prepared_data = prepared_data
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None

    def train(self, X, y):
        """
        Train the XGBoost classifier on the given features and labels.
        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
            y: Target labels (numpy array or pandas Series)
        """
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X, y)

        # Save model and scaler after training
        self.save_model()

    def save_model(self):
        """
        Save the trained model and scaler to disk.
        """
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
        if self.scaler is not None:
            joblib.dump(self.scaler, self.scaler_path)

    def load_model(self):
        """
        Load the model and scaler from disk.
        Raises FileNotFoundError if files not found.
        """
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError("Model or scaler file not found.")

        self.model = joblib.load(self.model_path)
        s





