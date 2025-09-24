import streamlit as st
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import pandas as pd

# -----------------------------
# GNews API Setup
# -----------------------------
# Use Streamlit secrets for security
API_KEY = st.secrets[f46d6c567e17c4981634467e431e3721]  # Set this in Streamlit secrets
BASE_URL = "https://gnews.io/api/v4/top-headlines"

categories = ["business", "technology", "sports", "entertainment"]
seen_titles = set()
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
    titles = [(a['title'], a['source']['name']) for a in articles]
    return titles

def get_sentiment(title):
    score = analyzer.polarity_scores(title)['compound']
    return "Positive" if score >= 0 else "Negative"

# -----------------------------
# Streamlit Dashboard
# -----------------------------
st.set_page_config(page_title="Real-Time News Sentiment", layout="wide")
st.title("📰 Real-Time News Sentiment Dashboard")
st.markdown("Live GNews headlines with sentiment analysis.")

# Last update timestamp
st.write("Last updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# -----------------------------
# Fetch latest headlines
# -----------------------------
batch_titles = []
for cat in categories:
    news = fetch_gnews(category=cat, max_results=5)
    for title, source in news:
        if title not in seen_titles:
            seen_titles.add(title)
            batch_titles.append({
                "Title": title,
                "Source": source,
                "Sentiment": get_sentiment(title)
            })

if batch_titles:
    df = pd.DataFrame(batch_titles)

    # -----------------------------
    # Metrics
    # -----------------------------
    pos_count = df[df['Sentiment'] == "Positive"].shape[0]
    neg_count = df[df['Sentiment'] == "Negative"].shape[0]

    col1, col2 = st.columns(2)
    col1.metric("Positive Headlines", pos_count)
    col2.metric("Negative Headlines", neg_count)

    # -----------------------------
    # Color-coded Table
    # -----------------------------
    def highlight_sentiment(row):
        color = 'background-color: #d4edda' if row['Sentiment'] == 'Positive' else 'background-color: #f8d7da'
        return [color]*len(row)

    st.subheader("Latest Headlines")
    st.dataframe(df.style.apply(highlight_sentiment, axis=1), height=400)

    # -----------------------------
    # Bar Chart for Sentiment Distribution
    # -----------------------------
    st.subheader("Sentiment Distribution")
    st.bar_chart(df['Sentiment'].value_counts())

else:
    st.write("No new headlines at this moment. Refreshing shortly...")

# -----------------------------
# Auto-refresh every 30 seconds
# -----------------------------
st.experimental_rerun()
