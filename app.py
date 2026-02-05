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
# VALIDASI KOLOM (ANTI ERROR)
# =====================================================
required_cols = ["content", "score", "sentimen", "sentimen_encoded"]
for col in required_cols:
    if col not in df.columns:
        st.error(f"Kolom '{col}' tidak ditemukan di dataset")
        st.write("Kolom tersedia:", df.columns)
        st.stop()

# =====================================================
# JUDUL
# =====================================================
st.title("📊 Dashboard Analisis Sentimen Ulasan Aplikasi Maxim")
st.markdown("""
Dashboard ini menyajikan hasil analisis sentimen ulasan pengguna aplikasi **Maxim**
menggunakan algoritma **XGBoost** dan **Random Forest**.
""")

# =====================================================
# RINGKASAN STATISTIK
# =====================================================
st.header("📌 Ringkasan Statistik")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Ulasan Dianalisis", f"{len(df)} Data")

with col2:
    sentiment_counts = df["sentimen"].value_counts()
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
# PERBANDINGAN PERFORMA MODEL
# =====================================================
st.header("📊 Perbandingan Performa Model")

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
st.header("📉 Confusion Matrix")

model_choice = st.selectbox(
    "Pilih Model untuk Confusion Matrix:",
    ["XGBoost", "Random Forest"]
)

y_true = df["sentimen_encoded"]
X_tfidf = vectorizer.transform(df["content"].astype(str))

if model_choice == "XGBoost":
    y_pred = model_xgb.predict(X_tfidf)
else:
    y_pred = model_rf.predict(X_tfidf)

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
st.header("☁️ Word Cloud Berdasarkan Sentimen")

sentiment_option = st.selectbox(
    "Pilih Sentimen:",
    sorted(df["sentimen"].unique())
)

text_data = " ".join(
    df[df["sentimen"] == sentiment_option]["content"].astype(str)
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
# TABEL ULASAN
# =====================================================
st.header("📋 Daftar Ulasan Terklasifikasi")

filter_sentiment = st.selectbox(
    "Filter Sentimen:",
    ["Semua"] + sorted(df["sentimen"].unique())
)

df_show = df if filter_sentiment == "Semua" else df[df["sentimen"] == filter_sentiment]

st.dataframe(
    df_show[["content", "score", "sentimen"]],
    use_container_width=True
)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    "<center>Dashboard Analisis Sentimen | Skripsi | 2026</center>",
    unsafe_allow_html=True
)
