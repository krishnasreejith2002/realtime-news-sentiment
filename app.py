import streamlit as st
import requests
import pandas as pd
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# -------------------------
# App Title
# -------------------------
st.set_page_config(page_title="Real-Time News Sentiment", layout="wide")
st.title("📰 Real-Time News Sentiment Dashboard")

# -------------------------
# API Config
# -------------------------
API_KEY = st.secrets["f46d6c567e17c4981634467e431e3721"]  # Secure way (set in Streamlit secrets)
BASE_URL = "https://gnews.io/api/v4/top-headlines"

# -------------------------
# Sentiment Analyzer
# -------------------------
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text: str) -> str:
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# -------------------------
# Fetch GNews
# -------------------------
def fetch_gnews(category="business", max_results=5):
    params = {
        "token": API_KEY,
        "category": category,
        "lang": "en",
        "max": max_results
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        return [(a["title"], category) for a in articles]
    except Exception as e:
        st.error(f"⚠️ Error fetching news: {e}")
        return []

# -------------------------
# Sidebar Controls
# -------------------------
categories = ["business", "technology", "sports", "entertainment"]
selected_categories = st.sidebar.multiselect("Select Categories", categories, default=categories)
refresh_time = st.sidebar.slider("Refresh interval (seconds)", 10, 120, 30)

# -------------------------
# Streaming Loop
# -------------------------
placeholder = st.empty()
seen_titles = set()

while True:
    batch_titles = []
    for cat in selected_categories:
        news = fetch_gnews(category=cat, max_results=5)
        for t in news:
            if t[0] not in seen_titles:
                seen_titles.add(t[0])
                batch_titles.append(t)

    if batch_titles:
        df = pd.DataFrame(batch_titles, columns=["title", "category"])
        df["sentiment"] = df["title"].apply(get_sentiment)

        with placeholder.container():
            st.subheader("📊 Latest News Sentiment")
            st.dataframe(df, use_container_width=True)

            # Sentiment counts
            sentiment_counts = df["sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

    time.sleep(refresh_time)
