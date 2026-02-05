import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Maxim",
    page_icon="🚖",
    layout="wide"
)

st.title("🚖 Dashboard Analisis Sentimen Ulasan Aplikasi Maxim")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("maxim_cleaned_labeled.csv")
    return df

df = load_data()

text_col = "clean_text"
label_col = "label"

# =====================================================
# ENCODE LABEL
# =====================================================
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df[label_col])

# =====================================================
# LOAD MODEL & VECTORIZER
# =====================================================
@st.cache_resource
def load_models():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model_xgb = joblib.load("model_xgb.pkl")
    model_rf = joblib.load("model_rf.pkl")
    return vectorizer, model_xgb, model_rf

vectorizer, model_xgb, model_rf = load_models()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Pengaturan")

model_choice = st.sidebar.selectbox(
    "Pilih Model Klasifikasi",
    ["XGBoost", "Random Forest"]
)

model = model_xgb if model_choice == "XGBoost" else model_rf

menu = st.sidebar.radio(
    "Menu",
    ["Ringkasan Data", "Visualisasi Sentimen", "Evaluasi Model", "Klasifikasi Ulasan Baru"]
)

# =====================================================
# RINGKASAN DATA
# =====================================================
if menu == "Ringkasan Data":
    st.subheader("📊 Ringkasan Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Ulasan", len(df))
    col2.metric("Jumlah Label", df[label_col].nunique())
    col3.metric("Model Aktif", model_choice)

    st.markdown("#### Distribusi Sentimen")
    sentiment_count = df[label_col].value_counts()
    st.bar_chart(sentiment_count)

    st.markdown("#### Contoh Data")
    st.dataframe(df[[text_col, label_col]].head(10))

# =====================================================
# VISUALISASI SENTIMEN
# =====================================================
elif menu == "Visualisasi Sentimen":
    st.subheader("☁️ Word Cloud per Sentimen")

    selected_label = st.selectbox(
        "Pilih Sentimen",
        df[label_col].unique()
    )

    text_data = " ".join(df[df[label_col] == selected_label][text_col].astype(str))

    if text_data.strip() == "":
        st.warning("Tidak ada teks untuk sentimen ini.")
    else:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(text_data)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

# =====================================================
# EVALUASI MODEL
# =====================================================
elif menu == "Evaluasi Model":
    st.subheader("📈 Evaluasi Kinerja Model")

    X = vectorizer.transform(df[text_col].astype(str))
    y_true = df["label_encoded"]
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    st.metric("Akurasi Model", f"{acc:.2%}")

    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )
    disp.plot(ax=ax, cmap="Blues")
    st.pyplot(fig)

    st.markdown("#### Classification Report")
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=False
    )
    st.text(report)

# =====================================================
# KLASIFIKASI ULASAN BARU
# =====================================================
elif menu == "Klasifikasi Ulasan Baru":
    st.subheader("✍️ Klasifikasi Sentimen Ulasan Baru")

    user_input = st.text_area(
        "Masukkan ulasan pengguna:",
        height=150
    )

    if st.button("Prediksi Sentimen"):
        if user_input.strip() == "":
            st.warning("Ulasan tidak boleh kosong.")
        else:
            X_input = vectorizer.transform([user_input])
            pred = model.predict(X_input)
            label_pred = label_encoder.inverse_transform(pred)[0]

            st.success(f"🎯 Hasil Prediksi Sentimen: **{label_pred.upper()}**")
