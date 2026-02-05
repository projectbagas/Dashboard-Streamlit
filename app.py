# =====================================================
# app.py — FINAL SIAP DEPLOY
# Dashboard Analisis Sentimen Aplikasi Maxim
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
    return pd.read_csv("maxim_siap_pakai.csv")

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
# AUTO DETEKSI KOLOM (AMAN)
# =====================================================
def detect_column(candidates, columns):
    for c in candidates:
        if c in columns:
            return c
    return None

text_col = detect_column(
    ["content", "ulasan", "review", "text", "komentar", "review_text", "content_clean"],
    df.columns
)
score_col = detect_column(
    ["rating", "score", "stars", "nilai"],
    df.columns
)
sentiment_col = detect_column(
    ["sentimen", "label", "kategori", "sentiment"],
    df.columns
)
sentiment_enc_col = detect_column(
    ["sentimen_encoded", "label_encoded", "sentiment_encoded", "y"],
    df.columns
)

missing = []
if text_col is None: missing.append("kolom teks ulasan")
if score_col is None: missing.append("kolom rating")
if sentiment_col is None: missing.append("kolom sentimen")
if sentiment_enc_col is None: missing.append("kolom sentimen numerik")

if missing:
    st.error("❌ Dataset belum memenuhi struktur dashboard")
    st.write("Kolom tidak ditemukan:", missing)
    st.write("Kolom tersedia:", list(df.columns))
    st.stop()

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
    st.markdown(
        "Dashboard ini menyajikan analisis sentimen ulasan pengguna aplikasi **Maxim** "
        "menggunakan algoritma **XGBoost** dan **Random Forest**."
    )

    st.header("📌 Ringkasan Statistik")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Ulasan Dianalisis", f"{len(df)} Data")

    with col2:
        counts = df[sentiment_col].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

# =====================================================
# PERFORMA MODEL
# =====================================================
elif menu == "Performa Model":
    st.header("📊 Perbandingan Performa Model")

    model_metrics = pd.DataFrame({
        "Model": ["XGBoost", "Random Forest"],
        "Akurasi": [0.87, 0.84],
        "Presisi": [0.86, 0.83],
        "Recall": [0.85, 0.82],
        "F1-Score": [0.85, 0.82]
    })

    metric = st.selectbox("Pilih Metrik Evaluasi:", ["Akurasi", "Presisi", "Recall", "F1-Score"])
    fig, ax = plt.subplots()
    ax.bar(model_metrics["Model"], model_metrics[metric])
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    ax.set_title(f"Perbandingan {metric}")
    st.pyplot(fig)

# =====================================================
# CONFUSION MATRIX
# =====================================================
elif menu == "Confusion Matrix":
    st.header("📉 Confusion Matrix")

    model_choice = st.selectbox("Pilih Model:", ["XGBoost", "Random Forest"])
    X_tfidf = vectorizer.transform(df[text_col].astype(str))
    y_true = df[sentiment_enc_col]
    y_pred = model_xgb.predict(X_tfidf) if model_choice == "XGBoost" else model_rf.predict(X_tfidf)

    labels = sorted(y_true.unique(), reverse=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix {model_choice}")
    st.pyplot(fig)

# =====================================================
# WORD CLOUD
# =====================================================
elif menu == "Word Cloud":
    st.header("☁️ Word Cloud Berdasarkan Sentimen")

    s_opt = st.selectbox("Pilih Sentimen:", sorted(df[sentiment_col].unique()))
    text_data = " ".join(df[df[sentiment_col] == s_opt][text_col].astype(str))

    if text_data.strip():
        wc = WordCloud(width=800, height=400, background_color="white").generate(text_data)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("Tidak ada data untuk sentimen ini.")

# =====================================================
# DATA ULASAN
# =====================================================
elif menu == "Data Ulasan":
    st.header("📋 Daftar Ulasan Terklasifikasi")

    f = st.selectbox("Filter Sentimen:", ["Semua"] + sorted(df[sentiment_col].unique()))
    show = df if f == "Semua" else df[df[sentiment_col] == f]

    st.dataframe(show[[text_col, score_col, sentiment_col]], use_container_width=True)

# =====================================================
# KLASIFIKASI ULASAN BARU
# =====================================================
elif menu == "Klasifikasi Ulasan Baru":
    st.header("✍️ Klasifikasi Ulasan Baru")

    user_text = st.text_area("Masukkan ulasan pengguna:", placeholder="Contoh: Driver lama, aplikasi sering error...")
    m_choice = st.selectbox("Pilih Model:", ["XGBoost", "Random Forest"])

    if st.button("Prediksi Sentimen"):
        if user_text.strip():
            X_in = vectorizer.transform([user_text])
            pred = model_xgb.predict(X_in) if m_choice == "XGBoost" else model_rf.predict(X_in)
            label_map = {0: "Tidak Puas", 1: "Netral", 2: "Puas"}
            st.success(f"🎯 Hasil Prediksi: **{label_map.get(pred[0], pred[0])}**")
        else:
            st.warning("⚠️ Masukkan teks ulasan terlebih dahulu.")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("<center>Dashboard Analisis Sentimen | Skripsi | 2026</center>", unsafe_allow_html=True)
