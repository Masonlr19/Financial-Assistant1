import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import os
import joblib

from services.tradier_client import TradierClient
from services.newsapi_client import NewsApiClient
from services.sentiment_analyzer import SentimentAnalyzer
from models.weekly_predictor import WeeklyPredictorDataPreparer
from models.xgboost_model import XGBoostPricePredictor


def load_trained_model(symbol):
    model_path = f"models/{symbol}_xgboost_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model for {symbol} not found. Please train it first.")
    return joblib.load(model_path)

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

def main():
    st.set_page_config(page_title="Financial Assistant", layout="wide")
    st.title("💰 ML Financial Assistant")

    # Sidebar Inputs
    st.sidebar.header("🔧 Settings")
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")

    # Tradier Client
    tradier_api_key = st.secrets["TRADIER_API_KEY"]
    tradier_client = TradierClient(tradier_api_key)

    # NewsAPI Client
    newsapi_api_key = st.secrets.get("NEWSAPI_API_KEY", None)
    newsapi_client = NewsApiClient(newsapi_api_key) if newsapi_api_key else None

    tab1, tab2, tab3 = st.tabs(["📈 Predictions", "📊 Historical Data", "📰 News Sentiment"])

    with tab2:
        if st.button("Show Historical Data"):
            with st.spinner("Loading historical data..."):
                try:
                    data = tradier_client.get_historical_data(symbol)
                    st.success("Data fetched successfully!")
                    plot_historical_prices(data)
                except Exception as e:
                    st.error(f"Error fetching data: {e}")

    with tab1:
        if st.button("Run Weekly Price Predictions"):
            with st.spinner("Loading model and predicting..."):
                try:
                    data = tradier_client.get_historical_data(symbol)
                    preparer = WeeklyPredictorDataPreparer()
                    _, _, prepared_data = preparer.prepare_data(data)

                    # Load pre-trained model
                    model = load_trained_model(symbol)

                    # Create predictor with loaded model
                    predictor = XGBoostPricePredictor(preparer.scaler, prepared_data, preloaded_model=model)
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


if __name__ == "__main__":
    main()
