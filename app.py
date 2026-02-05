import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Maxim",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("maxim_cleaned_labeled.csv")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_models():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model_xgb = joblib.load("model_xgb.pkl")
    model_rf = joblib.load("model_rf.pkl")
    return vectorizer, model_xgb, model_rf

df = load_data()
vectorizer, model_xgb, model_rf = load_models()

# =========================
# SIDEBAR
# =========================
menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Overview",
        "Performa Model",
        "Confusion Matrix",
        "Word Cloud",
        "Data Ulasan"
    ]
)

# =========================
# OVERVIEW
# =========================
if menu == "Overview":
    st.title("📊 Dashboard Analisis Sentimen Maxim")
    st.metric("Total Ulasan", len(df))

    fig, ax = plt.subplots()
    df["label"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

# =========================
# CONFUSION MATRIX
# =========================
elif menu == "Confusion Matrix":
    model_choice = st.selectbox("Pilih Model", ["XGBoost", "Random Forest"])
    model = model_xgb if model_choice == "XGBoost" else model_rf

    X = vectorizer.transform(df["clean_text"].astype(str))
    y_true = df["label"]
    y_pred = model.predict(X)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues")
    st.pyplot(fig)

# =========================
# WORD CLOUD
# =========================
elif menu == "Word Cloud":
    sent = st.selectbox("Pilih Sentimen", df["label"].unique())
    text = " ".join(df[df["label"] == sent]["clean_text"].astype(str))

    wc = WordCloud(background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

# =========================
# DATA
# =========================
elif menu == "Data Ulasan":
    st.dataframe(df, use_container_width=True)
