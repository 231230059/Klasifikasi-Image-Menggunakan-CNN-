import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Dashboard Klasifikasi Oceanic Life CNN",
    layout="wide"
)

# ==========================================
# LOAD RESOURCE (MODEL & ARTEFAK EVALUASI)
# ==========================================
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model('oceanic_life_cnn_model.h5')

@st.cache_data
def load_metadata():
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    metrics_df = pd.read_csv('evaluation_metrics_augmented.csv', index_col=0)
    cm = np.load('confusion_matrix.npy')
    class_names = list(class_indices.keys())
    return class_indices, class_names, history, metrics_df, cm

try:
    model = load_trained_model()
    class_indices, class_names, history, metrics_df, cm = load_metadata()
except Exception as e:
    st.error("Gagal memuat model atau file evaluasi. Pastikan Anda telah menjalankan `train_model.py` terlebih dahulu.")
    st.stop()

# ==========================================
# DASHBOARD LAYOUT & STREAMLIT COMPONENTS
# ==========================================
st.title("🌊 Oceanic Life Dataset - CNN Classification Dashboard")
st.markdown("""
Aplikasi ini menampilkan ringkasan data, grafik performa pelatihan model Convolutional Neural Network (CNN), 
metrik evaluasi detail, serta pengujian klasifikasi gambar organisme laut secara real-time.
""")

tabs = st.tabs([
    "📊 Ringkasan Dataset & EDA", 
    "📈 Performa Pelatihan Model", 
    "🎯 Evaluasi & Confusion Matrix", 
    "🔍 Real-time Inference & Analisis"
])

# ------------------------------------------
# TAB 1: EDA & RINGKASAN DATASET
# ------------------------------------------
with tabs[0]:
    st.header("Ringkasan & Eksplorasi Data (EDA)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Kelas", len(class_names))
    col2.metric("Split Data Latih", "80%")
    col3.metric("Split Data Uji", "20%")

    st.subheader("Daftar Kategori Organisme Laut")
    st.write(", ".join([f"**{c}**" for c in class_names]))
    
    st.markdown("""
    **Analisis Kualitas & Ukuran Data:**
    - Format Input Visual: Standardized RGB (Red, Green, Blue).
    - Normalisasi: Nilai piksel direscale ke rentang $[0, 1]$.
    - Data Augmentation: Terintegrasi untuk mencegah *overfitting* pada variasi sudut dan pencahayaan air.
    """)

# ------------------------------------------
# TAB 2: PELATIHAN MODEL (ACCURACY & LOSS)
# ------------------------------------------
with tabs[1]:
    st.header("Grafik Konvergensi Pelatihan (Loss & Accuracy)")
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Grafik Akurasi
    ax[0].plot(epochs, history['accuracy'], label='Training Accuracy', color='blue')
    ax[0].plot(epochs, history['val_accuracy'], label='Validation Accuracy', color='orange')
    ax[0].set_title('Training vs Validation Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[0].grid(True)

    # Grafik Loss
    ax[1].plot(epochs, history['loss'], label='Training Loss', color='blue')
    ax[1].plot(epochs, history['val_loss'], label='Validation Loss', color='orange')
    ax[1].set_title('Training vs Validation Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid(True)

    st.pyplot(fig)

    st.subheader("Interpretasi Konvergensi Model")
    st.write("""
    - **Akurasi Tinggi & Loss Rendah:** Menunjukkan bahwa lapisan konvolusi berhasil mengekstrak fitur spasial (tekstur, warna, dan bentuk) spesifik organisme.
    - **Gap Antara Latih & Validasi:** Apabila garis validasi melandai mendatar sementara garis latih terus naik, hal itu mengindikasikan potensi *overfitting*.
    """)

# ------------------------------------------
# TAB 3: EVALUASI METRIK & CONFUSION MATRIX
# ------------------------------------------
with tabs[2]:
    st.header("Metrik Evaluasi Model")
    
    col_metric, col_cm = st.columns([1, 1])
    
    with col_metric:
        st.subheader("Ringkasan Laporan Klasifikasi")
        st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)
        
        st.markdown("""
        **Keterangan Metrik:**
        - **Accuracy:** Proporsi prediksi benar dari seluruh sampel uji.
        - **Precision:** Tingkat ketepatan model ketika memprediksi kelas tertentu.
        - **Recall:** Kemampuan model dalam mengidentifikasi seluruh sampel dari suatu kelas.
        - **F1-Score:** Harmonic mean antara Precision dan Recall (keseimbangan metrik).
        """)

    with col_cm:
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
        plt.ylabel('Label Asli (True Label)')
        plt.xlabel('Label Prediksi (Predicted Label)')
        plt.xticks(rotation=45)
        st.pyplot(fig_cm)

# ------------------------------------------
# TAB 4: REAL-TIME INFERENCE & ANALISIS GAMBAR
# ------------------------------------------
with tabs[3]:
    st.header("Pengujian Prediksi Real-Time")
    st.write("Unggah gambar baru organisme laut untuk dianalisis oleh model CNN secara langsung.")
    
    uploaded_file = st.file_uploader("Pilih file gambar (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        col_img, col_pred = st.columns([1, 1])
        
        image = Image.open(uploaded_file).convert('RGB')
        
        with col_img:
            st.image(image, caption="Gambar yang Diunggah", use_column_width=True)
        
        # Pra-pemrosesan gambar untuk model
        img_resized = image.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Prediksi
        predictions = model.predict(img_batch)[0]
        predicted_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_idx]
        confidence = predictions[predicted_idx] * 100
        
        with col_pred:
            st.subheader("Hasil Prediksi Real-Time")
            st.success(f"**Prediksi Kategori:** {predicted_class}")
            st.info(f"**Tingkat Kepercayaan (Confidence):** {confidence:.2f}%")
            
            # Bar chart distribusi probabilitas
            prob_df = pd.DataFrame({
                'Kategori': class_names,
                'Probabilitas (%)': predictions * 100
            }).sort_values(by='Probabilitas (%)', ascending=True)
            
            fig_prob, ax_prob = plt.subplots(figsize=(6, 4))
            ax_prob.barh(prob_df['Kategori'], prob_df['Probabilitas (%)'], color='teal')
            ax_prob.set_xlabel('Probabilitas (%)')
            ax_prob.set_xlim(0, 100)
            st.pyplot(fig_prob)

        # Analisis Gambar
        st.subheader("📋 Analisis Visual Gambar")
        st.write(f"""
        - **Prediksi Utama:** Model mengklasifikasikan gambar ini sebagai **{predicted_class}** dengan probabilitas **{confidence:.2f}%**.
        - **Deskripsi Ekstraksi Fitur:** Lapisan *Convolutional* pada model mendeteksi kombinasi pola tekstur, bentuk tepi (edges), dan distribusi warna yang dominan relevan dengan ciri fisik kategori **{predicted_class}**.
        - **Evaluasi Kepercayaan:** 
            - Jika nilai kepastian **> 80%**, fitur khas visual pada gambar terlihat sangat jelas dan minim gangguan latar belakang (*background noise*).
            - Jika nilai kepastian **< 60%**, terdapat kemungkinan kemiripan visual (misalnya antara terumbu karang/sponges atau bentuk pencahayaan air laut) yang membingungkan batas keputusan (*decision boundary*) model.
        """)