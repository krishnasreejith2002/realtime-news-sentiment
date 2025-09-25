import streamlit as st
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------------
# Config
# -------------------------
API_KEY = st.secrets["f46d6c567e17c4981634467e431e3721"]  # put API key in Streamlit secrets
BASE_URL = "https://gnews.io/api/v4/top-headlines"
CATEGORIES = ["business", "technology", "sports", "entertainment"]

analyzer = SentimentIntensityAnalyzer()

def fetch_gnews(category="business", max_results=5):
    params = {
        "token": API_KEY,
        "category": category,
        "lang": "en",
        "max": max_results
    }
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    articles = r.json().get("articles", [])
    return [(a["title"], category) for a in articles]

def vader_sentiment(title):
    score = analyzer.polarity_scores(title)["compound"]
    return "Positive" if score >= 0 else "Negative"

# -------------------------
# Streamlit UI
# -------------------------
st.title("📰 Real-Time News Sentiment Dashboard")

selected_category = st.selectbox("Select category", CATEGORIES)
if st.button("Fetch Latest News"):
    news = fetch_gnews(selected_category, max_results=10)
    if not news:
        st.warning("No news found.")
    else:
        df = pd.DataFrame(news, columns=["title", "category"])
        df["sentiment"] = df["title"].apply(vader_sentiment)
        st.dataframe(df)

        # Visualization
        st.bar_chart(df["sentiment"].value_counts())
