# services/feature_engineering.py

import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds common technical indicators to the DataFrame.
    Assumes df has columns: 'date', 'close', 'open', 'high', 'low', 'volume'
    """

    df = df.copy()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()

    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    df['BB_Std'] = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

    # Fill NaN values created by rolling calculations
    df.fillna(method='bfill', inplace=True)
    
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    return df

def prepare_features(raw_data: dict) -> pd.DataFrame:
    """
    Prepare features from raw Tradier API response (assumed format).
    Returns a DataFrame ready for model training/prediction.
    """

    df = pd.DataFrame(raw_data['history']['day'])
    df = df.sort_values('date')

    df = add_technical_indicators(df)
    df = add_time_features(df)

    # Select relevant columns for modeling
    feature_cols = [
        'close', 'open', 'high', 'low', 'volume',
        'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
        'MACD', 'Signal_Line', 'RSI',
        'BB_Upper', 'BB_Lower',
        'week_of_year', 'month', 'day_of_week'
    ]

    return df[feature_cols + ['date']]

