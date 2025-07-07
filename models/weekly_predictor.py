import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class WeeklyPredictorDataPreparer:
    def __init__(self):
        self.scaler = StandardScaler()

    def prepare_data(self, historical_data):
        df = pd.DataFrame(historical_data['history']['day'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        weekly = df['close'].resample('W-FRI').last().dropna()

        data = pd.DataFrame()
        data['close'] = weekly
        data['return_1w'] = weekly.pct_change(1)
        data['return_2w'] = weekly.pct_change(2)
        data['ma_3w'] = weekly.rolling(window=3).mean()
        data['ma_5w'] = weekly.rolling(window=5).mean()
        data = data.dropna()

        data['label'] = (data['close'].shift(-1) > data['close']).astype(int)
        data = data.dropna()

        X = data[['return_1w', 'return_2w', 'ma_3w', 'ma_5w']]
        y = data['label']

        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y, data
