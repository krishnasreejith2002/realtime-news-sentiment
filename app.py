import streamlit as st
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------------------
# Streamlit UI
# ------------------------
st.title("Real-Time News Sentiment Dashboard")

# ------------------------
# GNews API Key
# ------------------------
API_KEY = "f46d6c567e17c4981634467e431e3721"  # <-- Replace with your actual key
URL = f"https://gnews.io/api/v4/top-headlines?country=in&max=10&apikey={API_KEY}"

# Fetch latest news
response = requests.get(URL)
data = response.json()
articles = data.get('articles', [])

if not articles:
    st.warning("No news fetched. Check your API key or internet connection.")
else:
    df = pd.DataFrame(articles)

    # Handle missing description
    if 'description' not in df.columns:
        df['description'] = ""
    df = df[['title', 'description']]

    # ------------------------
    # Sentiment Analysis
    # ------------------------
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        score = analyzer.polarity_scores(text)['compound']
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    df['sentiment'] = df['title'].apply(get_sentiment)

    # ------------------------
    # Display in Streamlit
    # ------------------------
    st.subheader("Latest News with Sentiment")
    st.dataframe(df[['title', 'sentiment']])

    st.subheader("Sentiment Distribution")
    st.bar_chart(df['sentiment'].value_counts())
