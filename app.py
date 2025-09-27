import streamlit as st
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

# ------------------------
# Streamlit UI
# ------------------------
st.title("Real-Time News Sentiment Dashboard")

# Set refresh interval (in seconds)
REFRESH_INTERVAL = 30

# ------------------------
# GNews API Key
# ------------------------
API_KEY = "f46d6c567e17c4981634467e431e3721"  # <-- Replace with your actual key
URL = f"https://gnews.io/api/v4/top-headlines?country=in&max=10&apikey={API_KEY}"

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# ------------------------
# Sentiment function
# ------------------------
def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.0:       # Positive if >= 0.0
        return "Positive"
    else:                  # Negative if < 0.0
        return "Negative"

# ------------------------
# Main loop for auto-refresh
# ------------------------
while True:
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

        # Combine title + description for better sentiment
        df['full_text'] = df['title'] + " " + df['description']

        # Get sentiment
        df['sentiment'] = df['full_text'].apply(get_sentiment)

        # Display news
        st.subheader("Latest News with Sentiment")
        st.dataframe(df[['title', 'sentiment']])

        # Display sentiment distribution
        st.subheader("Sentiment Distribution")
        st.bar_chart(df['sentiment'].value_counts())

    # Wait before next refresh
    time.sleep(REFRESH_INTERVAL)
