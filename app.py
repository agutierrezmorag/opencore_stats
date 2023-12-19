import os
from datetime import date

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import streamlit as st
import torch
from nltk.corpus import stopwords
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from transformers import BertTokenizerFast, EncoderDecoderModel, pipeline
from wordcloud import WordCloud


@st.cache_resource
def db_connection():
    """
    Establishes a connection to the MongoDB database.

    Returns:
        pymongo.database.Database: The MongoDB database object.
    """
    uri = st.secrets.mongodb.uri
    client = MongoClient(uri, server_api=ServerApi("1"))
    try:
        db = client.opencoredatabase
        return db
    except Exception as e:
        st.error(e)


@st.cache_resource(ttl=60 * 60 * 24)
def get_news_by_source():
    """
    Retrieves news data from various sources and displays the count of news articles by source.

    Returns:
        DataFrame: A DataFrame containing the count of news articles by source.
    """
    logos = {
        "adn": "https://opencore.onrender.com/static/img/source_logos/adn.png",
        "chvn": "https://opencore.onrender.com/static/img/source_logos/chvn.png",
        "cnn": "https://opencore.onrender.com/static/img/source_logos/cnn.png",
        "dinamo": "https://opencore.onrender.com/static/img/source_logos/dinamo.png",
        "elmostrador": "https://opencore.onrender.com/static/img/source_logos/elmostrador.png",
        "latercera": "https://opencore.onrender.com/static/img/source_logos/latercera.png",
        "meganoticias": "https://opencore.onrender.com/static/img/source_logos/meganoticias.png",
        "t13": "https://opencore.onrender.com/static/img/source_logos/t13.png",
    }

    news_df = get_all_news()
    news_df["logo"] = news_df["website"].map(logos)

    news_by_source = st.dataframe(
        news_df[["logo", "website"]].value_counts(),
        use_container_width=True,
        column_config={
            "logo": st.column_config.ImageColumn("Logo", width="small"),
            "website": "Nombre",
            "count": "Total",
        },
    )

    return news_by_source


def get_todays_news_summary():
    """
    Retrieves today's news articles and generates a summary for each article using a pre-trained model.

    Returns:
        str: A string containing the summaries of today's news articles.
    """
    model_name = "mrm8488/bert2bert_shared-spanish-finetuned-summarization"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = EncoderDecoderModel.from_pretrained(model_name).to(device)

    today = date.today()
    news_df = get_all_news()
    news_df["date_pulled"] = pd.to_datetime(news_df["date_pulled"])

    news_today = news_df[news_df["date_pulled"].dt.date == today]

    summaries = []

    for _, row in news_today.iterrows():
        content = row["content"]
        link = row["link"]
        inputs = tokenizer(
            [content],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        output = model.generate(input_ids, attention_mask=attention_mask)

        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(f"- {summary} [source]({link})")

    all_news_summary = "\n".join(summaries)

    return all_news_summary


@st.cache_resource(ttl=60 * 60 * 24)
def calc_wordcloud():
    """
    Calculates and returns a word cloud based on the content of news articles.

    Returns:
        wordcloud (WordCloud): The generated word cloud object.
    """
    nltk.data.path.append(st.secrets.nltk.download_path)
    nltk.download("stopwords", download_dir=st.secrets.nltk.download_path)
    db = db_connection()
    news_df = get_all_news()

    text = " ".join(news_df["content"])

    stop_words = set(stopwords.words("spanish"))
    with open("additional_stopwords.txt", "r", encoding="utf-8") as f:
        additional_stop_words = [word.strip() for word in f.readlines()]

    stop_words.update(additional_stop_words)

    wordcloud = WordCloud(
        stopwords=stop_words,
        background_color="white",
        contour_color="blue",
        colormap="viridis",
        width=800,
        height=400,
    ).generate(text)

    return wordcloud


@st.cache_resource(ttl=60 * 60 * 24)
def get_all_news():
    """
    Retrieves all news from the database and returns them as a DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing all the news.
    """
    db = db_connection()
    news_cursor = db.news_news.find()
    news_df = pd.DataFrame(list(news_cursor))
    news_df["_id"] = news_df["_id"].astype(str)
    return news_df


def main():
    st.set_page_config(page_title="OpenCore Stats", page_icon="🧊")

    db = db_connection()

    neutral_news = db.news_news.count_documents({"sentiment": "Neutro"})
    positive_news = db.news_news.count_documents({"sentiment": "Positivo"})
    negative_news = db.news_news.count_documents({"sentiment": "Negativo"})

    total_count = neutral_news + positive_news + negative_news

    neutral_percentage = (neutral_news / total_count) * 100
    positive_percentage = (positive_news / total_count) * 100
    negative_percentage = (negative_news / total_count) * 100

    st.markdown("## 📝 Noticias")

    _, col1, col2, col3, _ = st.columns([0.5, 1, 1, 1, 0.5])

    with col1:
        st.metric(
            label="🟩 Positivas",
            value=positive_news,
        )
    with col2:
        st.metric(
            label="➖ Neutrales",
            value=neutral_news,
        )
    with col3:
        st.metric(
            label="🟥 Negativas",
            value=negative_news,
        )

    st.markdown("## 📊 Gráfica")
    st.bar_chart(
        {
            "Positivas": positive_percentage,
            "Neutrales": neutral_percentage,
            "Negativas": negative_percentage,
        }
    )

    st.markdown("## ☁️ Word Cloud")
    wordcloud = calc_wordcloud()
    st.image(wordcloud.to_array())

    st.markdown("## 📰 Noticias por fuente")
    get_news_by_source()

    st.markdown("## 📰 Noticias del día")
    st.markdown("Te presentamos un resumen de las noticias del día.")
    todays_news_summary = get_todays_news_summary()
    st.markdown(todays_news_summary)


if __name__ == "__main__":
    main()
