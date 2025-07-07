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
        self.scaler = joblib.load(self.scaler_path)

    def predict_next_5_weeks(self):
        """
        Predict price movement for the next 5 weeks and return confidence scores.
        Assumes prepared_data includes the features for these 5 weeks.
        
        Returns:
            preds: List of 0 or 1 indicating predicted decrease or increase
            confs: List of confidence percentages for each prediction
        """
        if self.model is None:
            raise ValueError("Model is not loaded or trained.")

        # Extract features for next 5 weeks from prepared_data
        # This depends on how prepared_data is structured.
        # Here's an example assuming prepared_data['future_features'] exists and is a DataFrame:
        future_features = self.prepared_data.get('future_features', None)
        if future_features is None or len(future_features) < 5:
            raise ValueError("Prepared data missing future features for 5 weeks.")

        X_future = future_features.iloc[:5]

        # Scale features
        X_scaled = self.scaler.transform(X_future)

        # Predict probabilities (probability of class 1)
        probas = self.model.predict_proba(X_scaled)[:, 1]

        # Threshold at 0.5 to get binary prediction
        preds = (probas >= 0.5).astype(int)

        # Convert probabilities to confidence percentages (0-100%)
        confs = (probas * 100).round(1)

        return preds.tolist(), confs.tolist()






