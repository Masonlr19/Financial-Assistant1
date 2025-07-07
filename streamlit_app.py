import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

from services.tradier_client import TradierClient
from services.sentiment_analyzer import SentimentAnalyzer
from models.weekly_predictor import WeeklyPredictorDataPreparer
from models.xgboost_model import XGBoostPricePredictor


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
    api_key = st.secrets["TRADIER_API_KEY"]
    tradier_client = TradierClient(api_key)

    tab1, tab2, tab3 = st.tabs(["📈 Predictions", "📊 Historical Data", "📰 News Sentiment"])

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
                    predictor.train(X, y)

                    preds, confs = predictor.predict_next_5_weeks()

                    st.success("Prediction complete!")

                    # Show metrics
                    st.subheader("Prediction Summary")
                    for i, (p, c) in enumerate(zip(preds, confs), 1):
                        direction = "Increase 📈" if p == 1 else "Decrease 📉"
                        st.metric(label=f"Week {i}", value=direction, delta=f"{c}% confidence")

                    # Show confidence chart
                    plot_prediction_confidence(confs)

                except Exception as e:
                    st.error(f"Prediction error: {e}")

    with tab3:
        st.header(f"📰 News Sentiment for {symbol}")

        if st.button("Fetch and Analyze News Sentiment"):
            with st.spinner("Fetching news and analyzing sentiment..."):
                try:
                    news_data = tradier_client.get_news(symbol)
                    analyzer = SentimentAnalyzer()

                    sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
                    compound_scores = []

                    for article in news_data.get('news', []):
                        headline = article['headline']
                        sentiment, score = analyzer.analyze_sentiment(headline)
                        sentiments[sentiment] += 1
                        compound_scores.append(score)

                    avg_score = sum(compound_scores) / len(compound_scores) if compound_scores else 0
                    st.write(f"### Average Sentiment Score: {avg_score:.3f}")

                    st.bar_chart(sentiments)

                    st.subheader("News Headlines with Sentiment")
                    for article in news_data.get('news', []):
                        headline = article['headline']
                        sentiment, _ = analyzer.analyze_sentiment(headline)
                        st.write(f"**[{sentiment.upper()}]** {headline}")

                except Exception as e:
                    st.error(f"Error fetching or analyzing news: {e}")


if __name__ == "__main__":
    main()
