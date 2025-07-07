import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import joblib
import os

from services.tradier_client import TradierClient
from services.newsapi_client import NewsApiClient
from services.sentiment_analyzer import SentimentAnalyzer

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Hybrid model imports
from models.hybrid_data_preparer import HybridDataPreparer
from models.hybrid_model import HybridPricePredictor

# === Predictor class with fixes and future features generation ===
class XGBoostPricePredictor:
    def __init__(self, scaler, prepared_data=None, model_path="xgboost_model.joblib", scaler_path="scaler.joblib"):
        self.scaler = scaler
        self.prepared_data = prepared_data or {}
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None

    def train(self, X, y):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X, y)
        self.save_model()

    def save_model(self):
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
        if self.scaler is not None:
            joblib.dump(self.scaler, self.scaler_path)

    def load_model(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError("Model or scaler file not found.")
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def generate_future_features(self, n_weeks=5):
        hist_df = self.prepared_data.get('historical_features', None)
        if hist_df is None or hist_df.empty:
            raise ValueError("No historical features found to generate future features.")
        last_date = hist_df['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=n_weeks, freq='W-FRI')
        template = hist_df.drop(columns=['date']).iloc[-1]
        future_features = pd.DataFrame([template.values] * n_weeks, columns=template.index)
        future_features['date'] = future_dates
        cols = ['date'] + [c for c in future_features.columns if c != 'date']
        future_features = future_features[cols]
        return future_features

    def predict_next_5_weeks(self):
        if self.model is None:
            raise ValueError("Model is not loaded or trained.")
        future_features = self.prepared_data.get('future_features', None)
        if future_features is None or len(future_features) < 5:
            future_features = self.generate_future_features(n_weeks=5)
        if 'date' in future_features.columns:
            X_future = future_features.drop(columns=['date'])
        else:
            X_future = future_features
        X_scaled = self.scaler.transform(X_future)
        probas = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probas >= 0.5).astype(int)
        confs = (probas * 100).round(1)
        return preds.tolist(), confs.tolist()

# === Data Preparer with feature engineering and future features ===
class WeeklyPredictorDataPreparer:
    def __init__(self):
        self.scaler = StandardScaler()

    def prepare_data(self, raw_data):
        df = pd.DataFrame(raw_data['history']['day'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['return'] = df['close'].pct_change()
        df['ma_3'] = df['close'].rolling(window=3).mean()
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df = df.dropna().reset_index(drop=True)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna().reset_index(drop=True)
        feature_cols = ['return', 'ma_3', 'ma_5']
        X = df[feature_cols]
        y = df['target']
        X_scaled = self.scaler.fit_transform(X)
        historical_features = df[['date'] + feature_cols].copy()
        last_features = historical_features.iloc[-1][feature_cols]
        last_date = historical_features.iloc[-1]['date']
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=5, freq='W-FRI')
        future_features = pd.DataFrame([last_features.values] * 5, columns=feature_cols)
        future_features['date'] = future_dates
        future_features = future_features[['date'] + feature_cols]
        prepared_data = {
            'historical_features': historical_features,
            'future_features': future_features
        }
        return X_scaled, y.values, prepared_data

# === Plot helpers ===
def plot_historical_prices(data):
    df = pd.DataFrame(data['history']['day'])
    df['date'] = pd.to_datetime(df['date'])
    chart = alt.Chart(df).mark_line().encode(
        x='date:T',
        y='close:Q'
    ).properties(title="📊 Historical Close Prices")
    st.altair_chart(chart, use_container_width=True)

def plot_prediction_confidence(confs):
    fig, ax = plt.subplots()
    ax.bar(range(1, 6), confs, color='green')
    ax.set_title("📈 Confidence for Next 5 Fridays")
    ax.set_xlabel("Week")
    ax.set_ylabel("Confidence (%)")
    st.pyplot(fig)

# === Streamlit App ===
def main():
    st.set_page_config(page_title="Financial Assistant", layout="wide")
    st.title("💰 ML Financial Assistant")

    # Sidebar Inputs
    st.sidebar.header("🔧 Settings")
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")

    tradier_api_key = st.secrets["TRADIER_API_KEY"]
    tradier_client = TradierClient(tradier_api_key)

    newsapi_api_key = st.secrets.get("NEWSAPI_API_KEY", None)
    newsapi_client = NewsApiClient(newsapi_api_key) if newsapi_api_key else None

    # Add new tab
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Predictions", "📊 Historical Data", "📰 News Sentiment", "🤖 Hybrid Model"])

    with tab2:
        if st.button("Show Historical Data"):
            with st.spinner("Loading historical data..."):
                try:
                    data = tradier_client.get_historical_data(symbol)
                    st.success("Data fetched successfully!")
                    plot_historical_prices(data)
                    st.subheader("Raw JSON Response")
                    st.json(data)
                except Exception as e:
                    st.error(f"Error fetching data: {e}")

    with tab1:
        if st.button("Run Weekly Price Predictions"):
            with st.spinner("Running ML model..."):
                try:
                    data = tradier_client.get_historical_data(symbol)
                    preparer = WeeklyPredictorDataPreparer()
                    X, y, prepared_data = preparer.prepare_data(data)
                    predictor = XGBoostPricePredictor(preparer.scaler, prepared_data)

                    try:
                        predictor.load_model()
                        st.info("Loaded pre-trained model.")
                    except FileNotFoundError:
                        st.info("No pre-trained model found, training new model...")
                        predictor.train(X, y)

                    preds, confs = predictor.predict_next_5_weeks()

                    st.success("Prediction complete!")

                    st.subheader("Prediction Summary")
                    for i, (p, c) in enumerate(zip(preds, confs), 1):
                        direction = "Increase 📈" if p == 1 else "Decrease 📉"
                        st.metric(label=f"Week {i}", value=direction, delta=f"{c}% confidence")

                    plot_prediction_confidence(confs)

                except Exception as e:
                    st.error(f"Prediction error: {e}")

    with tab3:
        st.header(f"📰 News Sentiment for {symbol}")

        if st.button("Fetch and Analyze News Sentiment (Tradier + NewsAPI)"):
            with st.spinner("Fetching news and analyzing sentiment..."):
                try:
                    tradier_news_data = tradier_client.get_news(symbol)
                    analyzer = SentimentAnalyzer()

                    def analyze_news_list(news_list, key_headline):
                        sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
                        compound_scores = []
                        headlines_sentiments = []
                        for article in news_list:
                            headline = article[key_headline]
                            sentiment, score = analyzer.analyze_sentiment(headline)
                            sentiments[sentiment] += 1
                            compound_scores.append(score)
                            headlines_sentiments.append((headline, sentiment))
                        avg_score = sum(compound_scores) / len(compound_scores) if compound_scores else 0
                        return sentiments, avg_score, headlines_sentiments

                    tradier_sentiments, tradier_avg, tradier_headlines = analyze_news_list(
                        tradier_news_data.get('news', []), 'headline'
                    )

                    if newsapi_client:
                        newsapi_news_data = newsapi_client.get_news(symbol)
                        newsapi_sentiments, newsapi_avg, newsapi_headlines = analyze_news_list(
                            newsapi_news_data.get('articles', []), 'title'
                        )
                    else:
                        newsapi_sentiments, newsapi_avg, newsapi_headlines = {}, 0, []

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📈 Tradier News Sentiment")
                        st.write(f"Average Sentiment Score: {tradier_avg:.3f}")
                        st.bar_chart(tradier_sentiments)
                        st.write("Headlines:")
                        for headline, sentiment in tradier_headlines:
                            st.write(f"**[{sentiment.upper()}]** {headline}")

                    with col2:
                        st.subheader("📰 NewsAPI News Sentiment")
                        if newsapi_client:
                            st.write(f"Average Sentiment Score: {newsapi_avg:.3f}")
                            st.bar_chart(newsapi_sentiments)
                            st.write("Headlines:")
                            for headline, sentiment in newsapi_headlines:
                                st.write(f"**[{sentiment.upper()}]** {headline}")
                        else:
                            st.write("NewsAPI API key not configured.")

                except Exception as e:
                    st.error(f"Error fetching or analyzing news: {e}")

    # === NEW Hybrid Model Tab ===
    with tab4:
        st.header("🤖 Hybrid Model Predictions")

        if st.button("Run Hybrid Model Predictions"):
            with st.spinner("Running Hybrid ML model..."):
                try:
                    # Get historical price data
                    data = tradier_client.get_historical_data(symbol)
                    historical_df = pd.DataFrame(data['history']['day'])
                    historical_df['date'] = pd.to_datetime(historical_df['date'])

                    # Example: Mock function, replace with your real implementation
                    headlines_per_week = tradier_client.get_weekly_news_headlines(symbol, weeks=20)

                    # Prepare data
                    seq_len = 10
                    price_features = ['close', 'volume']
                    preparer = HybridDataPreparer(seq_len=seq_len)

                    price_seq = preparer.prepare_price_sequences(historical_df[price_features])
                    sentiment_seq = preparer.prepare_sentiment_sequences(headlines_per_week)
                    labels = preparer.prepare_labels(historical_df)

                    # Initialize and train/load model
                    predictor = HybridPricePredictor(seq_len=seq_len, num_price_features=len(price_features), sentiment_embedding_dim=384)

                    try:
                        predictor.load_model()
                        st.success("Loaded pre-trained Hybrid model.")
                    except Exception:
                        st.warning("No pre-trained Hybrid model found, training now...")
                        predictor.train(price_seq, sentiment_seq, labels)

                    # Predict on the last available 5 weeks
                    preds, confs = predictor.predict(price_seq[-5:], sentiment_seq[-5:])

                    st.subheader("Prediction Summary")
                    for i, (p, c) in enumerate(zip(preds, confs), 1):
                        direction = "Increase 📈" if p == 1 else "Decrease 📉"
                        st.metric(label=f"Week {i}", value=direction, delta=f"{c}% confidence")

                except Exception as e:
                    st.error(f"Hybrid model error: {e}")

if __name__ == "__main__":
    main()
