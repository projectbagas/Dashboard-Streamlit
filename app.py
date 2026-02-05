# =====================================================
# app.py — FINAL KHUSUS DATASET maxim_cleaned_labeled.csv
# SIAP DEPLOY STREAMLIT CLOUD
# =====================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Maxim",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("maxim_cleaned_labeled.csv")

# =====================================================
# LOAD MODEL & VECTORIZER
# =====================================================
@st.cache_resource
def load_models():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model_xgb = joblib.load("model_xgb.pkl")
    model_rf = joblib.load("model_rf.pkl")
    return vectorizer, model_xgb, model_rf

df = load_data()
vectorizer, model_xgb, model_rf = load_models()

# =====================================================
# AUTO DETEKSI KOLOM (KHUSUS DATASET KAMU)
# =====================================================
def detect_column(candidates, columns):
    for c in candidates:
        if c in columns:
            return c
    return None

# Kolom utama dataset
text_col = detect_column(["clean_text", "content"], df.columns)
score_col = detect_column(["score", "rating"], df.columns)
sentiment_col = detect_column(["label"], df.columns)

if text_col is None or score_col is None or sentiment_col is None:
    st.error("❌ Struktur dataset tidak sesuai")
    st.write("Kolom tersedia:", list(df.columns))
    st.stop()

# =====================================================
# KONVERSI LABEL TEKS → NUMERIK (WAJIB)
# =====================================================
label_map_text = {
    "tidak puas": 0,
    "netral": 1,
    "puas": 2
}

df["sentimen_encoded"] = (
    df[sentiment_col]
    .astype(str)
    .str.lower()
    .map(label_map_text)
)

if df["sentimen_encoded"].isna().any():
    st.error("❌ Ditemukan label sentimen di luar mapping")
    st.write("Label unik:", df[sentiment_col].unique())
    st.stop()

sentiment_enc_col = "sentimen_encoded"

# =====================================================
# SIDEBAR MENU
# =====================================================
menu = st.sidebar.radio(
    "📂 Pilih Halaman:",
    [
        "Overview",
        "Performa Model",
        "Confusion Matrix",
        "Word Cloud",
        "Data Ulasan",
        "Klasifikasi Ulasan Baru"
    ]
)

# =====================================================
# OVERVIEW
# =====================================================
if menu == "Overview":
    st.title("📊 Dashboard Analisis Sentimen Ulasan Aplikasi Maxim")

    st.markdown("""
    Dashboard ini menyajikan hasil analisis sentimen ulasan pengguna aplikasi **Maxim**
    berdasarkan data Google Play Store menggunakan algoritma
    **XGBoost** dan **Random Forest**.
    """)

    st.header("📌 Ringkasan Statistik")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Ulasan Dianalisis", f"{len(df)} Data")

    with col2:
        sentiment_counts = df[sentiment_col].value_counts()
        fig, ax = plt.subplots()
        ax.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)

# =====================================================
# PERFORMA MODEL
# =====================================================
elif menu == "Performa Model":
    st.header("📊 Perbandingan Performa Model")

    # Diisi manual sesuai hasil evaluasi skripsi
    model_metrics = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest"],
        "Akurasi": [0.87, 0.84],
        "Presisi": [0.86, 0.83],
        "Recall": [0.85, 0.82],
        "F1-Score": [0.85, 0.82]
    })

    metric_option = st.selectbox(
        "Pilih Metrik Evaluasi:",
        ["Akurasi", "Presisi", "Recall", "F1-Score"]
    )

    fig, ax = plt.subplots()
    ax.bar(model_metrics["Model"], model_metrics[metric_option])
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric_option)
    ax.set_title(f"Perbandingan {metric_option}")
    st.pyplot(fig)

# =====================================================
# CONFUSION MATRIX
# =====================================================
elif menu == "Confusion Matrix":
    st.header("📉 Confusion Matrix")

    model_choice = st.selectbox(
        "Pilih Model:",
        ["XGBoost", "Random Forest"]
    )

    X_tfidf = vectorizer.transform(df[text_col].astype(str))
    y_true = df[sentiment_enc_col]

    y_pred = model_xgb.predict(X_tfidf) if model_choice == "XGBoost" else model_rf.predict(X_tfidf)

    labels = [2, 1, 0]
    label_names = ["Puas", "Netral", "Tidak Puas"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_names
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix {model_choice}")
    st.pyplot(fig)

# =====================================================
# WORD CLOUD
# =====================================================
elif menu == "Word Cloud":
    st.header("☁️ Word Cloud Berdasarkan Sentimen")

    sentiment_option = st.selectbox(
        "Pilih Sentimen:",
        sorted(df[sentiment_col].unique())
    )

    text_data = " ".join(
        df[df[sentiment_col] == sentiment_option]["clean_text"].astype(str)
    )

    if text_data.strip():
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(text_data)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("Tidak ada data untuk sentimen ini.")

# =====================================================
# DATA ULASAN
# =====================================================
elif menu == "Data Ulasan":
    st.header("📋 Daftar Ulasan Terklasifikasi")

    filter_sentiment = st.selectbox(
        "Filter Sentimen:",
        ["Semua"] + sorted(df[sentiment_col].unique())
    )

    df_show = df if filter_sentiment == "Semua" else df[df[sentiment_col] == filter_sentiment]

    st.dataframe(
        df_show[[text_col, score_col, sentiment_col]],
        use_container_width=True
    )

# =====================================================
# KLASIFIKASI ULASAN BARU
# =====================================================
elif menu == "Klasifikasi Ulasan Baru":
    st.header("✍️ Klasifikasi Ulasan Baru")

    user_review = st.text_area(
        "Masukkan ulasan pengguna:",
        placeholder="Contoh: Driver lama dan aplikasi sering error"
    )

    model_choice = st.selectbox(
        "Pilih Model:",
        ["XGBoost", "Random Forest"]
    )

    if st.button("Prediksi Sentimen"):
        if user_review.strip():
            X_input = vectorizer.transform([user_review])
            prediction = model_xgb.predict(X_input) if model_choice == "XGBoost" else model_rf.predict(X_input)

            label_map_num = {0: "Tidak Puas", 1: "Netral", 2: "Puas"}
            st.success(f"🎯 Hasil Prediksi: **{label_map_num[prediction[0]]}**")
        else:
            st.warning("⚠️ Masukkan teks ulasan terlebih dahulu.")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    "<center>Dashboard Analisis Sentimen | Skripsi | 2026</center>",
    unsafe_allow_html=True
)
