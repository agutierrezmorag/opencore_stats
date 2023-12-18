import streamlit as st
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


@st.cache_resource
def db_connection():
    uri = st.secrets.mongodb.uri
    client = MongoClient(uri, server_api=ServerApi("1"))
    try:
        db = client.opencoredatabase
        return db
    except Exception as e:
        st.error(e)


def main():
    st.set_page_config(page_title="OpenCore Stats", page_icon="🧊")

    db = db_connection()

    # Count the number of documents for each sentiment
    neutral_news = db.news_news.count_documents({"sentiment": "Neutro"})
    positive_news = db.news_news.count_documents({"sentiment": "Positivo"})
    negative_news = db.news_news.count_documents({"sentiment": "Negativo"})

    # Calculate the total number of documents
    total_count = neutral_news + positive_news + negative_news

    # Calculate the percentage for each sentiment
    neutral_percentage = (neutral_news / total_count) * 100
    positive_percentage = (positive_news / total_count) * 100
    negative_percentage = (negative_news / total_count) * 100

    st.markdown("## 📝 Noticias")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Positivas",
            value=positive_news,
        )
    with col2:
        st.metric(
            label="Neutrales",
            value=neutral_news,
        )
    with col3:
        st.metric(
            label="Negativas",
            value=negative_news,
        )


if __name__ == "__main__":
    main()
