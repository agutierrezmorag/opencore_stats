import datetime
import os
from datetime import date

import nltk
import pandas as pd
import plotly.express as px
import streamlit as st
from nltk.corpus import stopwords
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.feature_extraction.text import CountVectorizer
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


@st.cache_resource(ttl=60 * 60 * 8)
def get_news_by_source():
    """
    Retrieves news data from various sources and displays the count of news articles by source.

    Returns:
        DataFrame: A DataFrame containing the count of news articles by source.
    """
    logos = {
        "adn": "app/static/adn.png",
        "chvn": "app/static/chvn.png",
        "cnn": "app/static/cnn.png",
        "dinamo": "app/static/dinamo.png",
        "elmostrador": "app/static/elmostrador.png",
        "latercera": "app/static/latercera.png",
        "meganoticias": "app/static/meganoticias.png",
        "t13": "app/static/t13.png",
    }

    news_df = get_all_news()
    news_df["logo"] = news_df["website"].map(logos)

    news_by_source = news_df["logo"].value_counts().reset_index()
    news_by_source.columns = ["logo", "count"]

    news_by_source = st.dataframe(
        news_by_source,
        use_container_width=True,
        hide_index=True,
        column_config={
            "logo": st.column_config.ImageColumn("Fuente", width="small"),
            "count": "Total",
        },
    )

    return news_by_source


def get_stop_words():
    """
    Retrieves a set of stop words for the Spanish language.

    Returns:
        set: A set of stop words.
    """
    nltk.data.path.append(st.secrets.nltk.download_path)
    nltk.download("stopwords", download_dir=st.secrets.nltk.download_path)

    stop_words = set(stopwords.words("spanish"))
    with open("additional_stopwords.txt", "r", encoding="utf-8") as f:
        additional_stop_words = [word.strip() for word in f.readlines()]
    stop_words.update(additional_stop_words)

    return stop_words


@st.cache_resource(ttl=60 * 60 * 8)
def calc_wordcloud():
    """
    Calculates and returns a word cloud based on the content of news articles.

    Returns:
        wordcloud (WordCloud): The generated word cloud object.
    """
    db = db_connection()
    news_df = get_all_news()

    text = " ".join(news_df["content"])

    stop_words = get_stop_words()

    wordcloud = WordCloud(
        stopwords=stop_words,
        background_color="white",
        contour_color="blue",
        colormap="viridis",
        width=800,
        height=400,
    ).generate(text)

    return wordcloud


@st.cache_resource(ttl=60 * 60 * 8)
def display_news_metrics():
    """
    Displays news metrics including counts and percentages of positive, neutral, and negative news.
    """
    db = db_connection()
    two_weeks_ago = datetime.datetime.now() - datetime.timedelta(weeks=2)
    neutral_news = db.news_news.count_documents(
        {"sentiment": "Neutro", "date_published": {"$gte": two_weeks_ago}}
    )
    positive_news = db.news_news.count_documents(
        {"sentiment": "Positivo", "date_published": {"$gte": two_weeks_ago}}
    )
    negative_news = db.news_news.count_documents(
        {"sentiment": "Negativo", "date_published": {"$gte": two_weeks_ago}}
    )

    total_count = neutral_news + positive_news + negative_news

    neutral_percentage = (neutral_news / total_count) * 100
    positive_percentage = (positive_news / total_count) * 100
    negative_percentage = (negative_news / total_count) * 100

    st.markdown("## 📝 Noticias")
    st.caption("Estadisticas de todas las noticias.")

    col1, col2, col3 = st.columns(3)

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

    df = pd.DataFrame(
        {
            "Analisis de sentimiento": ["Positivas", "Neutrales", "Negativas"],
            "Porcentaje": [
                positive_percentage,
                neutral_percentage,
                negative_percentage,
            ],
        }
    )

    color_map = {
        "Positivas": "#008000ba",
        "Neutrales": "#98989b8f",
        "Negativas": "#d34040",
    }

    fig = px.pie(
        df,
        values="Porcentaje",
        names="Analisis de sentimiento",
        color="Analisis de sentimiento",
        color_discrete_map=color_map,
    )

    st.plotly_chart(fig)


@st.cache_resource(ttl=60 * 60 * 8)
def calc_trending_topics(n=10):
    """
    Calculate the trending topics based on the content of news articles.

    Parameters:
    n (int): The number of trending topics to return. Default is 10.

    Returns:
    pandas.DataFrame: A DataFrame containing the trending topics and their frequencies.
    """
    news_df = get_all_news()
    text = " ".join(news_df["content"])
    stop_words = get_stop_words()

    vectorizer = CountVectorizer(stop_words=list(stop_words))

    word_freq = vectorizer.fit_transform([text])

    words = vectorizer.get_feature_names_out()
    frequencies = word_freq.toarray().flatten()

    word_freq_df = pd.DataFrame({"Palabra": words, "Frecuencia": frequencies})
    word_freq_df = word_freq_df.sort_values("Frecuencia", ascending=False)

    return word_freq_df.head(n)


def get_word_frequency_over_time(word):
    """
    Retrieves the frequency of a specified word in the news articles over time.

    Parameters:
    word (str): The word to track.

    Returns:
    pandas.DataFrame: A DataFrame containing the date and count of articles containing the word for each date.
    """
    news_df = get_all_news()
    news_df["content"] = news_df["content"].str.lower()

    word = word.lower()
    news_df = news_df[news_df["content"].str.contains(word)]
    word_frequency = news_df.groupby("date_published").size().reset_index(name="count")

    return word_frequency


@st.cache_resource(ttl=60 * 60 * 8)
def get_all_news():
    """
    Retrieves all news from the database from the last two weeks and returns them as a DataFrame.

    Returns:
        pandas.DataFrame: A DataFrame containing all the news from the last two weeks.
    """
    db = db_connection()
    two_weeks_ago = datetime.datetime.now() - datetime.timedelta(weeks=2)
    news_cursor = db.news_news.find({"date_published": {"$gte": two_weeks_ago}})
    news_df = pd.DataFrame(list(news_cursor))
    news_df["_id"] = news_df["_id"].astype(str)
    news_df["date_published"] = pd.to_datetime(news_df["date_published"])

    return news_df


def main():
    st.set_page_config(
        page_title="OpenCore Stats",
        page_icon="🧊",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .embeddedAppMetaInfoBar_container__DxxL1 {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        display_news_metrics()
    with col2:
        st.markdown("## 📈 Frecuencia de palabras por dia")
        st.caption(
            "Realiza un seguimiento de la frecuencia de las palabras en las noticias a lo largo del tiempo."
        )
        words_string = st.text_input("Palabras a buscar, separadas por espacios")
        words = words_string.split()

        if words:
            frequency_df = pd.DataFrame()
            for word in words:
                word_frequency = get_word_frequency_over_time(word)
                word_frequency["word"] = word
                frequency_df = pd.concat([frequency_df, word_frequency])
            fig = px.line(
                frequency_df,
                x="date_published",
                y="count",
                color="word",
            )
            fig.update_layout(
                xaxis_title="Fecha de publicación",
                yaxis_title="Conteo",
            )
            st.plotly_chart(fig)

    st.markdown("## 📈 Trending topics")
    st.caption("Las palabras mas comunes presentes en las noticias.")
    col3, col4 = st.columns(2)
    with col3:
        wordcloud = calc_wordcloud()
        st.image(wordcloud.to_array())

    with col4:
        trending_topics = calc_trending_topics()
        st.dataframe(trending_topics, hide_index=True, use_container_width=True)

    st.markdown("## 📰 Noticias por fuente")
    with st.spinner("Calculando las noticias por fuente..."):
        get_news_by_source()

    st.divider()
    st.caption("*_Estos datos solo consideran las noticias de los ultimos 14 dias._")


if __name__ == "__main__":
    main()
