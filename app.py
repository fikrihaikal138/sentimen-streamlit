import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import joblib

st.set_page_config("Analisis Sentimen Tiket.com - SVM & Asosiasi", layout="wide")

st.markdown("""
<style>
    body { background-color: #e0e0e0; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .stButton > button {
        background-color: white;
        color: black;
        width: 100%;
        height: 2.5em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background-color: white; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 20px;'>
    <h2 style='margin: 0; color: black;'>Analisis Sentimen Aplikasi Tiket.com</h2>
</div>
""", unsafe_allow_html=True)

# Navigasi Menu dengan Session State
if "menu" not in st.session_state:
    st.session_state.menu = "📈 Evaluasi Metrik"

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    if st.button("Dashboard"):
        st.session_state.menu = "🏠 Dashboard"
with col2:
    if st.button("Confusion Mtrx"):
        st.session_state.menu = "📊 Confusion Matrix"
with col3:
    if st.button("WordCloud"):
        st.session_state.menu = "💬 Wordcloud"
with col4:
    if st.button("Evaluasi Metrik"):
        st.session_state.menu = "📈 Evaluasi Metrik"
with col5:
    if st.button("Asosiasi kata"):
        st.session_state.menu = "📎 Asosiasi Kata (Apriori)"
with col6:
    if st.button("Prediksi Sentimen"):
        st.session_state.menu = "🔍 Prediksi Ulasan Baru"

# Gunakan menu dari session_state
menu = st.session_state.menu

# Fungsi bantu
def preprocess_text(text):
    return text.lower().strip() if isinstance(text, str) else ""

@st.cache_data
def load_data():
    df = pd.read_excel("hasil_preprocessing_tiketcom3.xlsx")
    df = df.dropna(subset=['stemming'])
    df['Sentimen'] = df['score'].apply(lambda x: 'Positif' if x >= 4 else 'Negatif')
    return df

@st.cache_resource
def load_model_vectorizer():
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

@st.cache_resource
def prepare_eval_data(df):
    X = tfidf.transform(df['stemming'])
    y = df['Sentimen'].values
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, df.index, test_size=0.2, stratify=y, random_state=42)
    train_texts = df.loc[idx_train].copy()
    test_texts = df.loc[idx_test].copy()
    return X_train, X_test, y_train, y_test, train_texts, test_texts

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    if y_pred.dtype in ['int64', 'int32', 'float64']:
        y_pred = pd.Series(y_pred).map({1: 'Positif', 0: 'Negatif'}).values
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label='Positif')
    rec = recall_score(y_true, y_pred, pos_label='Positif')
    f1 = f1_score(y_true, y_pred, pos_label='Positif')
    cm = confusion_matrix(y_true, y_pred, labels=['Positif', 'Negatif'])
    return acc, prec, rec, f1, cm, y_pred

# Load data dan model
data = load_data()
svm_model, tfidf = load_model_vectorizer()
X_train, X_test, y_train, y_test, train_texts, test_texts = prepare_eval_data(data)

# Menu: Evaluasi Metrik
if menu == "📈 Evaluasi Metrik":
    st.subheader("📈 Evaluasi Kinerja Model SVM")
    acc, prec, rec, f1, _, _ = evaluate_model(svm_model, X_test, y_test)
    metrics = ["Akurasi", "Presisi", "Recall", "F1-Score"]
    scores = [acc, prec, rec, f1]
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(metrics, scores, color="#2ecc71")
    ax.set_xlim(0, 1.05)
    for bar in bars:
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2f}", va='center')
    st.pyplot(fig)

# Menu: Confusion Matrix
elif menu == "📊 Confusion Matrix":
    st.subheader("📊 Confusion Matrix")
    _, _, _, _, cm, _ = evaluate_model(svm_model, X_test, y_test)
    
    # Adjust the figure size
    fig, ax = plt.subplots(figsize=(1, 1))  
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Positif', 'Negatif'], 
                yticklabels=['Positif', 'Negatif'], ax=ax)

    # Set font size for annotationsS
    ax.tick_params(labelsize=5)  # Change size of tick labels
    ax.set_xlabel("Prediksi", fontsize=5)  # Font size for x label
    ax.set_ylabel("Aktual", fontsize=5)  # Font size for y label
    for text in ax.texts:  # Adjust annotation text size
        text.set_size(5)  # Smaller size for annotation text

    st.pyplot(fig)



# Menu: WordCloud
elif menu == "💬 Wordcloud":
    st.subheader("💬 WordCloud Ulasan Positif dan Negatif")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**WordCloud - Positif**")
        pos_text = " ".join(data[data['Sentimen']=='Positif']['stemming'])
        wc_pos = WordCloud(width=600, height=300, background_color="white").generate(pos_text)
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        ax1.imshow(wc_pos, interpolation='bilinear')
        ax1.axis("off")
        st.pyplot(fig1)
    with col2:
        st.markdown("**WordCloud - Negatif**")
        neg_text = " ".join(data[data['Sentimen']=='Negatif']['stemming'])
        wc_neg = WordCloud(width=600, height=300, background_color="white").generate(neg_text)
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.imshow(wc_neg, interpolation='bilinear')
        ax2.axis("off")
        st.pyplot(fig2)

# Menu: Apriori
 
elif menu == "📎 Asosiasi Kata (Apriori)":
    st.subheader("📎 Analisis Asosiasi Kata (Apriori) Berdasarkan Sentimen")

    # Pastikan data sudah dimuat
    if 'data' not in globals() or data.empty:
        st.error("❌ Data belum dimuat. Silakan jalankan menu preprocessing terlebih dahulu.")
    else:
        total_ulasan = len(data)
        st.write(f"📊 Total ulasan: **{total_ulasan}**")

        # Slider Minimum Support (0.005 - 0.05)
        min_sup = st.slider(
        "🔧 Minimum Support:",
        min_value=0.005,
        max_value=0.05,
        value=0.01,   # nilai default
        step=0.001,   # langkah kecil biar presisi
        format="%.3f"
        )


        # Slider Minimum Confidence (0.50 - 0.90)
        min_conf = st.slider(
            "🔧 Minimum Confidence:",
            min_value=0.50,
            max_value=0.90,
            value=0.60,  # default 50%
            step=0.01,
            format="%.2f"
        )

        # Stopwords tambahan
        extra_stopwords = {
            'bisa', 'sudah', 'yang', 'akan', 'jadi', 'karena', 'untuk', 'e', 'yg',
            'dengan', 'pada', 'agar', 'dari', 'ke', 'di', 'itu', 'ini'
        }

        # Loop untuk setiap sentimen
        for sentimen in ['Positif', 'Negatif']:
            st.markdown(f"### 🔹 Asosiasi Kata - {sentimen}")

            # Ambil data sesuai sentimen
            subset = data[data['Sentimen'] == sentimen]['stemming'].dropna().apply(lambda x: x.split())
            subset = subset.apply(lambda tokens: [w for w in tokens if w not in extra_stopwords])

            if subset.empty:
                st.warning(f"⚠ Tidak ada data untuk sentimen {sentimen}")
                continue

            # Konversi ke format transaksi
            te = TransactionEncoder()
            te_ary = te.fit(subset).transform(subset)
            df_trans = pd.DataFrame(te_ary, columns=te.columns_)

            # Jalankan Apriori
            freq_item = apriori(df_trans, min_support=min_sup, use_colnames=True)

            if freq_item.empty:
                st.info(f"❗ Tidak ditemukan itemset dengan support ≥ {min_sup} untuk sentimen {sentimen}")
                continue

            # Buat aturan asosiasi
            rules = association_rules(freq_item, metric="confidence", min_threshold=min_conf)

            # Filter aturan yang valid dan lift ≥ 2
            rules = rules[
                (rules['antecedents'].apply(len) > 0) &
                (rules['consequents'].apply(len) > 0) &
                (rules['lift'] >= 1)

            ]

            if not rules.empty:
                rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

                # Urutkan berdasarkan lift tertinggi
                rules_sorted = rules.sort_values(by='lift', ascending=False).head(10).reset_index(drop=True)

                st.write(f"(Support ≥ {min_sup}, Confidence ≥ {min_conf}, Lift ≥ 2)")
                st.dataframe(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
            else:
                st.info(f"❗ Tidak ditemukan aturan asosiasi untuk sentimen {sentimen} "
                        f"dengan Support {min_sup}, Confidence {min_conf}, dan Lift ≥ 2")




# Menu: Prediksi Sentimen Baru
elif menu == "🔍 Prediksi Ulasan Baru":
    st.title("🔍 Prediksi Ulasan Baru dari File (SVM)")

    uploaded_file = st.file_uploader("Unggah file CSV atau Excel yang berisi kolom 'ulasan' dan 'score'", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_new = pd.read_csv(uploaded_file)
            else:
                df_new = pd.read_excel(uploaded_file)

            # Display total number of reviews
            total_reviews = df_new.shape[0]
            st.write(f"📊 Total Ulasan: {total_reviews}")
            st.write("📋 Kolom ditemukan:", df_new.columns.tolist())
            st.dataframe(df_new.head())


            # Pastikan kolom 'score' ada dan konversi ke numerik
            if 'score' in df_new.columns:
                df_new['score'] = pd.to_numeric(df_new['score'], errors='coerce')
                df_new['label'] = df_new['score'].apply(
                    lambda x: 'Positif' if x in [4, 5] else 'Negatif' if x in [1, 2, 3] else None
                )

            if 'ulasan' not in df_new.columns:
                st.error("❌ Kolom 'ulasan' tidak ditemukan.")
                st.stop()

            df_new = df_new.dropna(subset=['ulasan'])

            # Preprocessing AAAAAAAAAAAAAAAAAA  "JHh
            df_new['clean'] = df_new['ulasan'].apply(preprocess_text)
            tfidf_new = tfidf.transform(df_new['clean'])

            # Prediksi dan konversi ke label
            df_new['Prediksi_SVM'] = pd.Series(svm_model.predict(tfidf_new)).map({1: 'Positif', 0: 'Negatif'})

            # Tampilkan hasil
            st.subheader("📄 Contoh Hasil Prediksi")
            st.dataframe(df_new[['ulasan', 'score', 'label', 'Prediksi_SVM']].head(30))

            # Visualisasi distribusi
            st.subheader("📊 Distribusi Prediksi Sentimen")
            svm_counts = df_new['Prediksi_SVM'].value_counts()
            fig_svm = px.bar(
                x=svm_counts.index, y=svm_counts.values,
                labels={'x': 'Sentimen', 'y': 'Jumlah'},
                color=svm_counts.index,
                color_discrete_map={'Positif': '#2ecc71', 'Negatif': '#e74c3c'},
                title='Distribusi Sentimen - SVM'
            )
            st.plotly_chart(fig_svm, use_container_width=True)

            # Confusion Matrix dan Evaluasi
            if 'label' in df_new.columns:
                df_eval = df_new.dropna(subset=['label'])
                df_eval = df_eval[df_eval['label'].isin(['Positif', 'Negatif'])]

                if not df_eval.empty:
                    st.subheader("📊 Confusion Matrix (SVM)")
                    cm = confusion_matrix(df_eval['label'], df_eval['Prediksi_SVM'], labels=['Positif', 'Negatif'])
                    fig_cm, ax_cm = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=['Positif', 'Negatif'],
                                yticklabels=['Positif', 'Negatif'], ax=ax_cm)
                    ax_cm.set_xlabel("Prediksi")
                    ax_cm.set_ylabel("Aktual")
                    st.pyplot(fig_cm)

                    # Evaluasi metrik
                    acc = accuracy_score(df_eval['label'], df_eval['Prediksi_SVM'])
                    prec = precision_score(df_eval['label'], df_eval['Prediksi_SVM'], pos_label='Positif')
                    rec = recall_score(df_eval['label'], df_eval['Prediksi_SVM'], pos_label='Positif')
                    f1 = f1_score(df_eval['label'], df_eval['Prediksi_SVM'], pos_label='Positif')

                    df_metrik = pd.DataFrame({
                        'Metrik': ['Akurasi', 'Presisi', 'Recall', 'F1-Score'],
                        'Skor': [round(acc, 2), round(prec, 2), round(rec, 2), round(f1, 2)]
                    })

                    st.subheader("📈 Metrik Evaluasi SVM")
                    fig_bar = px.bar(
                        df_metrik, x='Metrik', y='Skor',
                        color='Metrik', color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning("⚠️ Tidak ada data valid untuk evaluasi.")

            # Aturan asosiasi kata (Apriori) dari hasil prediksi
            st.subheader("📎 Asosiasi Kata dari Hasil Prediksi (Apriori)")
            # Slider Minimum Support (0.005 - 0.05)
            min_sup = st.slider(
                "🔧 Minimum Support:",
                min_value=0.005,
                max_value=0.05,
                value=0.01,   # nilai default
                step=0.001,   # langkah kecil biar presisi
                format="%.3f"
            )
            # Slider Minimum Confidence (0.50 - 0.90)
            min_conf = st.slider(
                "🔧 Minimum Confidence:",
                min_value=0.50,
                max_value=0.90,
                value=0.60,  
                step=0.01,
                format="%.2f"
            )

            for sentimen in ['Positif', 'Negatif']:
                st.markdown(f"#### 💬 {sentimen}")
                subset = df_new[df_new['Prediksi_SVM'] == sentimen]

               
                extra_stopwords = {
                    'bisa', 'sudah', 'yang', 'akan', 'jadi', 'karena', 'untuk', 'e', 'yg',
                    'dengan', 'pada', 'agar', 'dari', 'ke', 'di', 'itu', 'ini', 'dan', 'atau','tidak','ada'
                }

                transaksi = [
                    [w for w in row.split() if w not in extra_stopwords]
                    for row in subset['clean']
                    if isinstance(row, str)
                ]


                st.caption(f"Jumlah transaksi (prediksi {sentimen}): {len(transaksi)}")

                if transaksi:
                    te = TransactionEncoder()
                    te_ary = te.fit(transaksi).transform(transaksi)
                    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

                    freq_item = apriori(df_trans, min_support=min_sup, use_colnames=True)
                    rules = association_rules(freq_item, metric="lift", min_threshold=1.0)

                    
                    if not rules.empty:
                        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by="lift", ascending=False).head(10))
                    else:
                        st.info("❗ Tidak ditemukan aturan asosiasi yang memenuhi syarat.")
                else:
                    st.warning("❗ Tidak ada data bersih untuk label ini.")

        except Exception as e:
            st.error(f"❌ Gagal memproses file: {e}")

    else:
        st.info("ℹ️ Silakan unggah file terlebih dahulu.")


# Menu: Dashboard
elif menu == "🏠 Dashboard":
    st.subheader("🏠 Statistik Dataset Tiket.com")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Ulasan", len(data))
    with col2:
        sentiment_counts = data['Sentimen'].value_counts()
        fig = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, title='Distribusi Sentimen')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔢 Distribusi Rating")
    fig2 = px.histogram(data, x='score', nbins=5, title='Distribusi Skor Rating')
    st.plotly_chart(fig2, use_container_width=True)
    
  
