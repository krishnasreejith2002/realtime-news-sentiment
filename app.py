import streamlit as st
import pandas as pd
import requests
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# --- Streamlit UI ---
st.title("Real-Time News Sentiment Dashboard")

# --- GNews API Key ---
API_KEY = "f46d6c567e17c4981634467e431e3721"  # Replace with your actual key
URL = f"https://gnews.io/api/v4/top-headlines?country=in&max=10&apikey={API_KEY}"

# Fetch news
response = requests.get(URL)
data = response.json()
articles = data.get('articles', [])
df = pd.DataFrame(articles)

# Handle missing description
if 'description' not in df.columns:
    df['description'] = ""
df = df[['title', 'description']]

# --- Spark Setup ---
spark = SparkSession.builder.appName("NewsSentiment").getOrCreate()
spark_df = spark.createDataFrame(df)

# --- Dummy sentiment labels ---
spark_df = spark_df.withColumn(
    "label",
    col("title").rlike("Modi|success|good|happy|announces").cast("double")
)

# --- ML Pipeline ---
tokenizer = Tokenizer(inputCol="title", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

model = pipeline.fit(spark_df)
predictions = model.transform(spark_df)

# Convert to Pandas for Streamlit
pred_df = predictions.select("title", "prediction").toPandas()
pred_df['sentiment'] = pred_df['prediction'].apply(lambda x: "Positive" if x == 1.0 else "Negative")

# --- Display ---
st.subheader("Latest News with Sentiment")
st.dataframe(pred_df[['title', 'sentiment']])

st.subheader("Sentiment Distribution")
st.bar_chart(pred_df['sentiment'].value_counts())
