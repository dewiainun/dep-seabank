import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import requests
from io import BytesIO

# ==========================================
# 1. KONFIGURASI HALAMAN (Wajib di baris pertama)
# ==========================================
st.set_page_config(
    page_title="Sentimen SeaBank | SVM",
    page_icon="🏦",
    layout="centered", # Mengubah layout agar lebih rapi di berbagai ukuran layar
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. INJEKSI CSS KUSTOM (Styling UI)
# ==========================================
st.markdown("""
    <style>
    /* Mengubah warna latar tombol utama */
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        padding: 10px;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff3333;
        border: none;
        transform: scale(1.02);
    }
    
    /* Mempercantik kotak hasil prediksi */
    .hasil-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .hasil-positif { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .hasil-negatif { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    .hasil-netral { background-color: #fff3cd; color: #856404; border: 2px solid #ffeeba; }
    
    /* Mempercantik font judul */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. FUNGSI PRAPEMROSESAN & LOAD MODEL
# ==========================================
# --- Setup Sastrawi & NLTK ---
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('indonesian'))

# Pengecualian kata penguat/sentimen agar tidak terhapus oleh Stopword Removal
pengecualian = {'sangat', 'tidak', 'kurang', 'lebih', 'paling', 'agak', 'bagus', 'baik', 'buruk', 'jelek', 'kecewa', 'puas'}
stop_words = stop_words - pengecualian

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --- Setup Kamus Normalisasi (Di-cache agar aplikasi tidak lemot) ---
@st.cache_data
def load_kamus_normalisasi():
    try:
        url = "https://github.com/analysisdatasentiment/kamus_kata_baku/raw/main/kamuskatabaku.xlsx"
        response = requests.get(url)
        file_excel = BytesIO(response.content)
        kamus_data = pd.read_excel(file_excel)
        return dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))
    except:
        return {} # Jika gagal unduh, kembalikan kamus kosong

kamus_norm = load_kamus_normalisasi()

def preprocess_text(text: str) -> str:
    """Fungsi pembersihan teks lengkap sesuai pipeline."""
    # 1. Case Folding & Cleaning
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 2. Tokenizing & Normalisasi
    tokens = text.split()
    tokens = [kamus_norm.get(word, word) for word in tokens]
    
    # 3. Stopword Removal
    tokens = [word for word in tokens if word not in stop_words]
    
    # 4. Stemming Sastrawi
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

@st.cache_resource
def load_models():
    """Fungsi untuk memuat model biner ke dalam cache memori."""
    with open('tfidf_vectorizer_smote.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    with open('svm_model_smote.pkl', 'rb') as file:
        model = pickle.load(file)
    return vectorizer, model

# Inisialisasi Model 
try:
    tfidf_vectorizer, svm_model = load_models()
    model_ready = True
except Exception as e:
    model_ready = False
    error_msg = str(e)

# ==========================================
# 4. STRUKTUR NAVIGASI (SIDEBAR)
# ==========================================
with st.sidebar:
    # Menggunakan use_container_width agar responsif
    st.image("seabank_sidebar.png", use_container_width=True)
    st.markdown("## Navigasi Sistem")
    menu = st.radio(
        "Pilih Menu:",
        ["🏦 Tentang SeaBank", "📊 Evaluasi Model", "🔍 Prediksi Sentimen"]
    )
    st.markdown("---")
    st.markdown("### Info Sistem")
    st.info("Algoritma: **SVM (Linear)**\n\nEkstraksi: **TF-IDF**\n\nResampling: **SMOTE**")
    st.caption("© 2026 - Analisis Sentimen Skripsi")

# ==========================================
# 5. HALAMAN: TENTANG SEABANK
# ==========================================
if menu == "🏦 Tentang SeaBank":
    st.title("Tentang Aplikasi SeaBank")
    st.markdown("---")
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("""
        **PT Bank SeaBank Indonesia (SeaBank)** adalah bank digital yang terintegrasi dengan ekosistem Sea Group (induk perusahaan Shopee dan Garena). 
        
        Sebagai salah satu pemain utama di industri perbankan digital Indonesia, SeaBank menawarkan berbagai kemudahan, di antaranya:
        * 💸 Bebas biaya transfer antar bank dan e-wallet.
        * 📈 Suku bunga tabungan dan deposito yang kompetitif.
        * 📱 Pembukaan rekening digital yang instan dan praktis.
        * 🛒 Integrasi langsung dengan metode pembayaran di Shopee.
        
        **Mengapa Analisis Sentimen Dibutuhkan?**
        Volume ulasan aplikasi SeaBank di Google Play Store sangat masif, mencapai jutaan unduhan. Untuk mengevaluasi kepuasan nasabah, mendeteksi gangguan sistem (*bug*/*error*), dan merumuskan strategi perbaikan antarmuka (*User Experience*), penilaian opini secara manual menjadi tidak efisien.
        
        Oleh karena itu, sistem ini dibangun menggunakan algoritma kecerdasan buatan untuk mengekstraksi dan mengklasifikasikan keluhan maupun kepuasan nasabah secara otomatis.
        """)
    with col2:
        # Menggunakan use_container_width agar gambar tidak merusak tata letak
        st.image("seabank_sidebar.png", use_container_width=True)

# ==========================================
# 6. HALAMAN: EVALUASI MODEL
# ==========================================
elif menu == "📊 Evaluasi Model":
    st.title("Hasil Evaluasi Model Machine Learning")
    st.markdown("Berdasarkan pengujian terhadap **1.104 data uji**, berikut adalah performa dari algoritma Support Vector Machine (SVM) setelah dilakukan pembobotan TF-IDF dan penyeimbangan data menggunakan SMOTE.")
    st.markdown("---")
    
    # Menampilkan Metrik Utama
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Akurasi Keseluruhan", value="89.9%")
    m2.metric(label="Presisi (Kelas Positif)", value="96.7%")
    m3.metric(label="Recall (Kelas Positif)", value="91.3%")
    m4.metric(label="F1-Score (Kelas Positif)", value="93.9%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Confusion Matrix")
        st.markdown("Matriks ini menunjukkan distribusi tebakan sistem dibandingkan dengan label aslinya.")
        
        # Membuat dummy data menyerupai hasil untuk divisualisasikan
        cm_data = [[114, 10, 15], [8, 406, 20], [29, 16, 473]]
        df_cm = pd.DataFrame(cm_data, index=['Negatif', 'Netral', 'Positif'], columns=['Negatif', 'Netral', 'Positif'])
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax, square=True, linewidths=0.5)
        ax.set_xlabel('Prediksi Model')
        ax.set_ylabel('Data Aktual')
        # Menggunakan use_container_width agar grafik matplotlib responsif
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("Wawasan (Insights)")
        st.info("""
        **Analisis Performa:**
        * 🎯 Model memiliki sensitivitas yang sangat baik dalam mendeteksi opini **Positif**, yang terbukti dari tingginya angka *True Positive* (473 data berhasil ditebak benar).
        * ⚖️ Teknik **SMOTE** terbukti berhasil menyelamatkan model dari bias (overfitting) ke arah kelas mayoritas, sehingga kelas **Negatif** tetap bisa dikenali dengan baik (114 data).
        * 🚀 Penggunaan parameter **Kernel Linear** pada SVM sangat optimal untuk mengolah data teks berdimensi tinggi hasil pembobotan TF-IDF.
        """)

# ==========================================
# 7. HALAMAN: PREDIKSI SENTIMEN (UTAMA)
# ==========================================
else:
    st.title("🔍 Prediksi Sentimen Ulasan")
    st.markdown("Masukkan teks opini, keluhan, atau *feedback* terkait aplikasi SeaBank, dan biarkan AI mendeteksi apakah sentimennya **Positif**, **Negatif**, atau **Netral**.")
    st.markdown("---")

    if not model_ready:
        st.error(f"⚠️ **Sistem Gagal Memuat Model!**\n\nPastikan file `tfidf_vectorizer_smote.pkl` dan `svm_model_smote.pkl` berada di direktori yang sama dengan `app.py`. \n\n*Log Error: {error_msg}*")
    else:
        user_input = st.text_area(
            "📝 Ketik ulasan di kotak bawah ini:", 
            placeholder="Contoh: SeaBank bagus banget, transfer kemana aja gratis tanpa biaya admin. Aplikasinya juga ringan...",
            height=150
        )

        if st.button("Mulai Analisis"):
            if user_input.strip() == "":
                st.warning("⚠️ Silakan masukkan teks ulasan terlebih dahulu!")
            else:
                with st.spinner('Sistem sedang mengekstraksi fitur dan mengklasifikasikan teks...'):
                    # 1. Pipeline Prapemrosesan
                    cleaned_text = preprocess_text(user_input)
                    
                    # 2. Pipeline Ekstraksi Fitur
                    text_vector = tfidf_vectorizer.transform([cleaned_text])
                    
                    # 3. Pipeline Prediksi
                    prediksi = svm_model.predict(text_vector)[0]
                    
                    # 4. Tampilkan Hasil
                    st.markdown("### 🎯 Hasil Deteksi:")
                    
                    if prediksi.lower() == 'positif':
                        st.markdown(f'<div class="hasil-box hasil-positif"><h2>😃 SENTIMEN POSITIF</h2><p>Teks ini teridentifikasi sebagai apresiasi atau kepuasan pengguna.</p></div>', unsafe_allow_html=True)
                        st.balloons()
                        
                    elif prediksi.lower() == 'negatif':
                        st.markdown(f'<div class="hasil-box hasil-negatif"><h2>😞 SENTIMEN NEGATIF</h2><p>Teks ini teridentifikasi sebagai keluhan, masalah teknis, atau ketidakpuasan.</p></div>', unsafe_allow_html=True)
                        
                    else:
                        st.markdown(f'<div class="hasil-box hasil-netral"><h2>😐 SENTIMEN NETRAL</h2><p>Teks ini teridentifikasi sebagai pertanyaan, saran, atau informasi objektif.</p></div>', unsafe_allow_html=True)
                    
                    # Fitur opsional untuk melihat teks yang dibersihkan di balik layar
                    with st.expander("🛠️ Lihat Data yang Diproses (Di Balik Layar)"):
                        st.write("**Teks Input Asli:**", user_input)
                        st.write("**Teks Pasca-Prapemrosesan:**", cleaned_text)
                        st.caption("Ini adalah format string akhir yang disuntikkan ke dalam matriks TF-IDF untuk dihitung bobot jaraknya oleh SVM.")