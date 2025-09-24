import streamlit as st
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from time import sleep

# ----------------------------
# CONFIG
# ----------------------------
GNEWS_API_KEY = "f46d6c567e17c4981634467e431e3721"  # <-- replace with your API key
GNEWS_URL = "https://gnews.io/api/v4/top-headlines"

analyzer = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Real-Time News Sentiment", layout="wide")
st.title("📰 Real-Time News Sentiment Dashboard")

# ----------------------------
# FUNCTIONS
# ----------------------------
def fetch_news(query="India", max_results=5):
    params = {
        "q": query,
        "token": GNEWS_API_KEY,
        "lang": "en",
        "max": max_results
    }
    response = requests.get(GNEWS_URL, params=params)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return [{"title": art["title"]} for art in articles]
    else:
        st.error(f"Failed to fetch news: {response.status_code}")
        return []

def classify_sentiment(title):
    score = analyzer.polarity_scores(title)["compound"]
    return "Positive" if score >= 0 else "Negative"

def get_dataframe(news_list):
    df = pd.DataFrame(news_list)
    df["sentiment"] = df["title"].apply(classify_sentiment)
    return df

# ----------------------------
# STREAMING LOOP
# ----------------------------
query = st.text_input("Search news for keyword:", value="India")
batch_size = st.number_input("Number of headlines per batch:", min_value=1, max_value=20, value=5)

placeholder = st.empty()

while True:
    news_list = fetch_news(query=query, max_results=batch_size)
    if news_list:
        df = get_dataframe(news_list)
        placeholder.dataframe(df)
    else:
        placeholder.write("No news available.")
    sleep(10)  # refresh every 10 seconds
