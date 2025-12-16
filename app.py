import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import nltk
import gensim.downloader as api
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import os
import sys
import altair as alt 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib 

# --- SETUP NLTK & FUNGSI ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
    
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Clean text untuk AWE
def clean_text_for_ml(text):
    text = str(text).lower()
    text = re.sub(r'[\u200b\u200c\u200d\uFEFF]', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^\w\s]", " ", text) 
    text = re.sub(r'\s+', ' ', text).strip()
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words] 
    return " ".join(words)

# AWE Vektor
def document_vector(text, wv):
    words = text.split()
    word_vecs = [wv[word] for word in words if word in wv]
    if not word_vecs:
        return np.zeros(100) 
    return np.mean(word_vecs, axis=0)

# --- MUAT MODEL & DATA CACHE (SOLUSI FINAL DUPLIKASI KOLOM) ---
@st.cache_resource
def load_resources():
    model_svm_loaded = None
    if os.path.exists('model_svm_winner.joblib'):
        try:
            with open('model_svm_winner.joblib', 'rb') as file:
                model_svm_loaded = joblib.load(file)
        except Exception as e:
            st.error(f"Gagal memuat Model SVM: {e}. PASTIKAN FILE MODEL BERADA DI 'model_svm_winner.joblib'.")
            
    word_vectors_loaded = None
    try:
        word_vectors_loaded = api.load("glove-wiki-gigaword-100") 
    except Exception as e:
        st.error(f"Gagal memuat Word Vectors dari Gensim: {e}")

    df_summary = pd.DataFrame()
    if os.path.exists('final_summary_results.csv'):
        df_summary = pd.read_csv('final_summary_results.csv')
        
        # 1. CLEANING: Bersihkan whitespace dan standardisasi ke nama Indonesia
        df_summary.columns = df_summary.columns.str.strip()
        
        # 2. Hapus kolom 'Accuracy' (English) jika ada, dan pastikan 'Akurasi' yang dipakai
        if 'Accuracy' in df_summary.columns:
            df_summary.drop(columns=['Accuracy'], errors='ignore', inplace=True)
            
        # 3. Rename 'Accuracy' (Jika ada) menjadi 'Akurasi'
        if 'Accuracy' in df_summary.columns and 'Akurasi' not in df_summary.columns:
            df_summary.rename(columns={'Accuracy': 'Akurasi'}, inplace=True)
            
        # 4. Rename metrik weighted avg jika nama di CSV masih pendek
        if 'F1-Score' in df_summary.columns and 'F1-Score (Weighted Avg)' not in df_summary.columns:
            df_summary.rename(columns={'F1-Score': 'F1-Score (Weighted Avg)'}, inplace=True)
        if 'Precision' in df_summary.columns and 'Precision (Weighted Avg)' not in df_summary.columns:
            df_summary.rename(columns={'Precision': 'Precision (Weighted Avg)'}, inplace=True)
        if 'Recall' in df_summary.columns and 'Recall (Weighted Avg)' not in df_summary.columns:
            df_summary.rename(columns={'Recall': 'Recall (Weighted Avg)'}, inplace=True)
            
        # 5. HAPUS DUPLIKASI YANG MASIH ADA (Final Check)
        df_summary = df_summary.loc[:, ~df_summary.columns.duplicated(keep='first')]

        # FIX: KEYERROR 'Metode Imbalance'
        if 'Skenario' in df_summary.columns:
            df_summary['Metode Imbalance'] = df_summary['Skenario'].apply(
                lambda x: 'SMOTE' if 'SMOTE' in x else ('Class Weight' if 'CW' in x else 'Original Imbalance')
            )
        
    df_eda = pd.DataFrame()
    if os.path.exists('eda_data.csv'):
         df_eda = pd.read_csv('eda_data.csv')
         df_eda['review_clean'] = df_eda['review'].apply(clean_text_for_ml) 
         df_eda['word_count'] = df_eda['review'].apply(lambda x: len(str(x).split()))
         analyzer = SentimentIntensityAnalyzer()
         df_eda['label_vader'] = df_eda['review'].apply(lambda x: 'positive' if analyzer.polarity_scores(x)['compound'] >= 0 else 'negative')

    df_raw = pd.DataFrame()
    if os.path.exists('raw_data_stats.csv'):
        df_raw = pd.read_csv('raw_data_stats.csv')
        df_raw = df_raw.dropna(subset=['rating']) 
        
        # FIX: KEYERROR 'label_vader' di df_raw
        if 'label_vader' not in df_raw.columns:
            analyzer = SentimentIntensityAnalyzer()
            df_raw['label_vader'] = df_raw['review'].apply(lambda x: 'positive' if analyzer.polarity_scores(x)['compound'] >= 0 else 'negative')


    return model_svm_loaded, word_vectors_loaded, df_eda, df_summary, df_raw

model_svm, word_vectors, df_eda, df_summary, df_raw = load_resources()
le_classes = {0: 'negative', 1: 'positive'} 

st.set_page_config(layout="wide")
st.title("🌟 Proyek Analisis Sentimen Film La La Land")
st.markdown("Metodologi Klasifikasi dengan Word Embeddings (AWE) dan Penanganan Imbalance Data")
st.markdown("---")

# =======================================================
# SIDEBAR NAVIGATION (MENGGUNAKAN RADIO BUTTONS)
# =======================================================
st.sidebar.title("Data Mining Kelompok 4")
st.sidebar.markdown("---")

# Menggunakan st.sidebar.radio
selected_page = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "🚀 Prediksi Real-Time",
        "⚙️ Pipeline Data & Proses Awal",
        "📊 Dashboard Analisis Data",
        "🔍 Komparasi Pelabelan",
        "🏆 Kinerja Model (9 Skenario)", 
        "📈 Perbandingan Metrik Global"  
    ],
    index=0 
)
st.sidebar.markdown("---")


# =======================================================
# KONTEN BERDASARKAN PILIHAN SIDEBAR
# =======================================================

if selected_page == "🚀 Prediksi Real-Time":
    # KODE TAB 1 (Prediksi Real-Time)
    st.header("Deteksi Sentimen Ulasan Baru 💬")
    st.subheader("Model Pemenang: **SVM LinearSVC (AWE + SMOTE)**")
    
    col_input, col_pred = st.columns([2, 1])
    
    with col_input:
        user_input = st.text_area(
            "Tulis ulasan film *La La Land* di sini:", 
            height=200, 
            placeholder="Contoh: The jazz music was excellent, but the plot dragged on way too long and I found the main characters annoying."
        )

    with col_pred:
        st.markdown("---")
        if st.button("PREDIKSI SENTIMEN", use_container_width=True, type="primary"):
            if not user_input:
                st.warning("⚠️ Mohon masukkan ulasan terlebih dahulu.")
            elif model_svm is None or word_vectors is None:
                st.error("🚨 Model belum berhasil dimuat. Coba refresh halaman.")
            else:
                with st.spinner('Memproses...'):
                    cleaned_review = clean_text_for_ml(user_input)
                    AWE_vector = document_vector(cleaned_review, word_vectors).reshape(1, -1)

                    prediction_id = model_svm.predict(AWE_vector)[0]
                    prediction_label = le_classes.get(prediction_id, 'Unknown')
                    
                    st.subheader("Hasil:")
                    if prediction_label == 'positive':
                        st.success(f"**POSITIF** 🎉", icon="👍")
                        st.markdown("Ulasan ini cenderung berisi **apresiasi**.")
                    else:
                        st.error(f"**NEGATIF** 🙁", icon="👎")
                        st.markdown("Ulasan ini mendeteksi adanya **kritik**.")
                
                    st.caption(f"Input Setelah Preprocessing: {cleaned_review}")

elif selected_page == "⚙️ Pipeline Data & Proses Awal":
    # KODE TAB 5 (Pipeline Data & Preprocessing INTERAKTIF)
    with st.container():
        st.header("Pipeline Data & Proses Awal ⚙️")

        st.subheader("1. Data Mentah dan Proses Preprocessing")
        
        # Penjelasan Proses Preprocessing
        st.info(
            """
            **Tujuan Preprocessing:** Mengubah ulasan mentah menjadi format yang dapat dipahami model (fitur numerik AWE). 
            Proses ini meliputi:
            1. **Case Folding** (Semua teks menjadi huruf kecil).
            2. **Cleaning** (Menghapus tautan, simbol, dan *punctuation*).
            3. **Stopword Removal** (Menghapus kata-kata umum seperti 'the', 'a', 'is').
            4. **Lemmatization** (Mengubah kata ke bentuk dasar, e.g., 'running' menjadi 'run').
            """
        )

        if not df_raw.empty:
            
            # Tampilkan 5 data mentah awal
            st.caption("Contoh 5 Data Ulasan Mentah (*raw data*):")
            st.dataframe(df_raw[['review', 'rating', 'label_vader']].head(5), use_container_width=True)

            st.markdown("---")

            # --- Tombol Pemicu Preprocessing ---
            if 'show_processed' not in st.session_state:
                st.session_state['show_processed'] = False
            
            # Ubah Tombol menjadi tombol untuk melihat hasil preprocessing
            if st.button("Tampilkan Hasil Preprocessing", type="primary"):
                st.session_state['show_processed'] = not st.session_state['show_processed'] # Toggle state

            if st.session_state['show_processed']:
                st.subheader("2. Hasil Preprocessing (Teks Bersih) & Vektorisasi")
                
                # Menggunakan DataFrame EDA (hasil preprocessing)
                if not df_eda.empty:
                    processed_df = df_raw[['review', 'rating']].head(5).copy()
                    
                    # Kolom review yang sudah diproses (Cleaned text)
                    processed_df['Cleaned Text (Input AWE)'] = df_eda['review_clean'].head(5) 
                    
                    st.caption("Perbandingan Data Mentah vs. Data Setelah Preprocessing:")
                    st.dataframe(processed_df, use_container_width=True)

                    st.markdown("---")
                    
                    # Penjelasan Vektorisasi
                    col_vect = st.columns(1)[0]
                    with col_vect:
                        st.subheader("3. Proses Vektorisasi Fitur")
                        st.markdown(
                            """
                            Teks bersih kemudian diubah menjadi numerik menggunakan **Average Word Embeddings (AWE)** dari GloVe 100-dimensi.
                            * **Metode:** Setiap kata diubah menjadi vektor (GloVe), lalu vektor rata-rata dihitung untuk seluruh ulasan.
                            * **Output:** Setiap ulasan menjadi 1 vektor dengan **100 dimensi**.
                            """
                        )
                        if word_vectors is not None:
                            st.success("✅ Model Word Embeddings (GloVe 100D) Berhasil Dimuat.")
                        else:
                            st.error("🚨 Model GloVe gagal dimuat.")
            
            else:
                st.markdown("Klik tombol di atas untuk melihat bagaimana ulasan mentah diubah menjadi data siap-model!")


elif selected_page == "📊 Dashboard Analisis Data":
    # KODE TAB 2 (EDA Bersih)
    with st.container():
        st.header("Analisis Data Eksplorasi (EDA) Ulasan 🔬")
        
        if not df_eda.empty:
            
            # 1. VISUALISASI UTAMA: Word Cloud vs Histogram
            st.subheader("1. Analisis Struktur Teks")
            col2_1, col2_2 = st.columns(2)

            with col2_1:
                st.caption("Histogram: Distribusi Panjang Ulasan")
                fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
                sns.histplot(df_eda['word_count'], ax=ax_hist, kde=True, bins=30, color='#1f77b4')
                ax_hist.set_title("Distribusi Jumlah Kata per Ulasan")
                ax_hist.set_xlabel("Jumlah Kata")
                st.pyplot(fig_hist)
                st.info("Mayoritas ulasan berada pada rentang singkat (<50 kata), membenarkan fitur AWE.")


            with col2_2:
                st.caption("Word Cloud Kata Kunci Ulasan")
                text = " ".join(df_eda['review'].astype(str))
                wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(text)
                fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
                st.info("Kata kunci utama menunjukkan fitur semantik yang dominan ('music', 'love', 'city').")
            
            st.markdown("---")

            # 2. BOX PLOT SENTIMEN
            st.subheader("2. Analisis Perilaku Penulisan (Panjang vs Sentimen)")
            
            col_box, col_box_info = st.columns([2, 1])
            with col_box:
                st.caption("Box Plot: Panjang Ulasan vs. Sentimen (Berdasarkan Label VADER)")
                if 'label_vader' in df_eda.columns:
                    fig_box, ax_box = plt.subplots(figsize=(8, 4))
                    sns.boxplot(x='label_vader', y='word_count', data=df_eda, palette={'positive': '#33FF57', 'negative': '#FF5733'}, ax=ax_box)
                    ax_box.set_title('Distribusi Panjang Kata Berdasarkan Sentimen')
                    ax_box.set_xlabel('Sentimen (VADER)')
                    ax_box.set_ylabel('Jumlah Kata')
                    st.pyplot(fig_box)
                else:
                    st.error("Kolom label_vader tidak tersedia untuk Box Plot.")
                
            with col_box_info:
                st.markdown("---")
                st.warning(
                    """
                    **Insight Box Plot:** Ulasan Negatif cenderung memiliki variasi panjang yang lebih besar, menunjukkan perlunya detail lebih saat mengkritik.
                    """
                )
            
            st.markdown("---")

elif selected_page == "🔍 Komparasi Pelabelan":
    # KODE TAB 3 (Komparasi Pelabelan)
    with st.container():
        st.header("Komparasi Hasil Pelabelan Otomatis 🔎")
        
        if all(col in df_eda.columns for col in ['label_vader', 'label_flair', 'label_zeroshot']):
            
            # 1. VISUALISASI PERBANDINGAN (Tiga Grafik Terpisah dalam Kolom)
            st.subheader("1. Distribusi Label dari Setiap Metode")
            
            col_vader, col_flair, col_zeroshot = st.columns(3)
            
            # --- Fungsi Pembantu untuk Membuat Grafik Batang ---
            def create_bar_chart(df, column, title):
                counts = df[column].value_counts().reset_index()
                counts.columns = ['Sentimen', 'Count']
                
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.barplot(x='Sentimen', y='Count', data=counts, palette={'positive': '#33FF57', 'negative': '#FF5733'}, ax=ax)
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('')
                ax.set_ylabel('Count', fontsize=9)
                ax.tick_params(axis='both', which='major', labelsize=8)
                plt.close(fig) 
                return fig
            # --------------------------------------------------

            with col_vader:
                st.caption("Grafik 1: Pelabelan VADER")
                st.pyplot(create_bar_chart(df_eda, 'label_vader', 'VADER'))
                
            with col_flair:
                st.caption("Grafik 2: Pelabelan FLAIR")
                st.pyplot(create_bar_chart(df_eda, 'label_flair', 'FLAIR'))
                
            with col_zeroshot:
                st.caption("Grafik 3: Pelabelan ZERO-SHOT")
                st.pyplot(create_bar_chart(df_eda, 'label_zeroshot', 'ZERO-SHOT'))

            st.markdown("---")
            
            # 2. RINGKASAN DATA (Presisi Tabel sebagai Komparasi)
            st.subheader("2. Ringkasan Komparasi Numerik dan Justifikasi")
            
            col_table, col_justifikasi = st.columns([1, 1])

            with col_table:
                summary_data = {
                    'Metode': ['VADER', 'FLAIR', 'ZERO-SHOT'],
                    'Positif': [
                        df_eda['label_vader'].value_counts().get('positive', 0),
                        df_eda['label_flair'].value_counts().get('positive', 0),
                        df_eda['label_zeroshot'].value_counts().get('positive', 0)
                    ],
                    'Negatif': [
                        df_eda['label_vader'].value_counts().get('negative', 0),
                        df_eda['label_flair'].value_counts().get('negative', 0),
                        df_eda['label_zeroshot'].value_counts().get('negative', 0)
                    ]
                }
                summary_table = pd.DataFrame(summary_data).set_index('Metode')
                
                st.caption("Tabel Komparasi Jumlah Ulasan Positif/Negatif:")
                st.dataframe(summary_table, use_container_width=True)
            
            with col_justifikasi:
                st.info("Justifikasi Pilihan *Ground Truth* (VADER)")
                st.markdown(
                    """
                    * **Fokus Visual:** Grafik terpisah menunjukkan distribusi kelas setiap metode secara detail.
                    * **VADER** dipilih karena menghasilkan rasio Positif/Negatif yang **paling realistis** (mayoritas positif) untuk ulasan film populer, dan konsisten (*lexicon-based*).
                    * Tabel ringkasan berfungsi sebagai komparasi utama antar metode.
                    """
                )
            
        else:
            st.error("Data EDA tidak mengandung semua kolom labeling yang diperlukan (VADER, FLAIR, Zero-Shot).")

elif selected_page == "🏆 Kinerja Model (9 Skenario)":
    # KODE TAB INTERAKTIF MODEL DETAIL (Tampilan Kotak)
    with st.container():
        st.header("🏆 Kinerja Model (Pilih Skenario) 📊")
        st.info("Pilih salah satu dari 9 skenario untuk melihat detail hasil evaluasi.")

        if not df_summary.empty:
            # Dapatkan daftar skenario yang unik untuk radio button
            scenario_options = df_summary['Skenario'].unique().tolist()
            
            st.subheader("1. Pilih Skenario Model")
            selected_scenario = st.radio(
                "Skenario:",
                options=scenario_options,
                index=0 # Default ke skenario pertama
            )

            st.markdown("---")

            # 2. Tampilkan Hasil Skenario Terpilih
            if selected_scenario:
                st.subheader(f"2. Detail Evaluasi Skenario: {selected_scenario}")
                
                # Filter data berdasarkan pilihan
                selected_data = df_summary[df_summary['Skenario'] == selected_scenario].iloc[0]

                # Tampilan Kotak/Card Metrik
                # Baris 1: Akurasi, F1-Score
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.container(border=True, height=100).metric(
                        label="Akurasi", 
                        value=f"{selected_data['Akurasi']:.4f}"
                    )
                with col2:
                    st.container(border=True, height=100).metric(
                        label="F1-Score (Weighted Avg)", 
                        value=f"{selected_data['F1-Score (Weighted Avg)']:.4f}"
                    )
                with col3:
                    st.container(border=True, height=100).metric(
                        label="Precision (Weighted Avg)", 
                        value=f"{selected_data['Precision (Weighted Avg)']:.4f}"
                    )
                with col4:
                    st.container(border=True, height=100).metric(
                        label="Recall (Weighted Avg)", 
                        value=f"{selected_data['Recall (Weighted Avg)']:.4f}"
                    )
                    
                st.markdown("---")
                
                # Baris 2: Metode Imbalance
                col_imbalance = st.columns(1)[0]
                with col_imbalance:
                     st.metric(label="Metode Imbalance", value=selected_data['Metode Imbalance']) 
                
                # CATATAN: Classification Report Detail (col_cr) Dihapus sesuai permintaan.


elif selected_page == "📈 Perbandingan Metrik Global":
    # TAB BARU UNTUK PERBANDINGAN GRAFIK GLOBAL
    with st.container():
        st.header("📈 Perbandingan Metrik Global Antar Skenario")
        
        if not df_summary.empty:
            df_summary_display = df_summary.copy()
            
            # --- Pilihan Metrik untuk Visualisasi ---
            st.subheader("1. Pilih Metrik untuk Perbandingan (Grafik)")
            metric_to_plot = st.radio(
                "Pilih Metrik untuk Perbandingan:",
                # MENGHAPUS 'Akurasi' DARI PILIHAN
                options=['F1-Score (Weighted Avg)', 'Recall (Weighted Avg)', 'Precision (Weighted Avg)'],
                index=0
            )

            st.markdown("---")

            # 1. VISUALISASI PERBANDINGAN 9 SKENARIO
            st.subheader(f"2. Grafik Perbandingan {metric_to_plot} (Diurutkan Menurun)")
            
            # Kolom 'Akurasi' sekarang seharusnya unik
            chart_data = df_summary_display[[
                'Skenario', 
                metric_to_plot,
                'Akurasi' 
            ]].sort_values(by=metric_to_plot, ascending=False).reset_index(drop=True)
            
            chart_data['Algoritma'] = chart_data['Skenario'].apply(lambda x: x.split(' ')[0])
            
            chart_f1 = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X(f'{metric_to_plot}:Q', title=metric_to_plot),
                y=alt.Y('Skenario:N', title='Skenario Model', sort=alt.EncodingSortField(field=metric_to_plot, order='descending')), 
                color=alt.Color('Algoritma:N'),
                tooltip=['Skenario', metric_to_plot, 'Akurasi'] 
            ).properties(
                title=f'Perbandingan {metric_to_plot} 9 Skenario'
            ).interactive()
            
            st.altair_chart(chart_f1, use_container_width=True)

            st.markdown("---")
            
            # 2. TABEL RINGKASAN AKHIR
            st.subheader("3. Tabel Ringkasan Kinerja Semua Skenario")
            
            # Kolom yang ingin ditampilkan di tabel
            display_cols = ['Skenario', 'Akurasi', 'F1-Score (Weighted Avg)', 'Recall (Weighted Avg)', 'Precision (Weighted Avg)', 'Metode Imbalance']
            
            def highlight_winner(s):
                is_winner = s['Skenario'] == 'SVM 3.2 (AWE+SMOTE)'
                return ['background-color: #f7ffdd; color: #4CAF50; font-weight: bold' if is_winner else '' for v in s]

            st.dataframe(
                df_summary_display[display_cols].sort_values(by='F1-Score (Weighted Avg)', ascending=False).style.apply(highlight_winner, axis=1).format('{:.4f}', subset=['Akurasi', 'F1-Score (Weighted Avg)', 'Recall (Weighted Avg)', 'Precision (Weighted Avg)']), 
                use_container_width=True
            )
            
            st.markdown("---")
            st.subheader("4. Kesimpulan Model Pemenang:")
            
            winner_f1 = df_summary_display[df_summary_display['Skenario'] == 'SVM 3.2 (AWE+SMOTE)']['F1-Score (Weighted Avg)'].iloc[0]
            st.success(
                f"**Model Pemenang:** **SVM LinearSVC 3.2 (AWE + SMOTE)**\n\n"
                f"- **F1-Score Tertinggi:** **{winner_f1:.4f}**\n"
                f"- **Kunci Sukses:** Penanganan *imbalance* data menggunakan **SMOTE** mendominasi kinerja, membuktikan bahwa penanganan data tidak seimbang lebih penting daripada pemilihan algoritma (LR, RF, atau SVM) itu sendiri dalam kasus ini. "
            )
