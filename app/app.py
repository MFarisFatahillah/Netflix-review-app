import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

from typing import Tuple

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Sentiment Analytics",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL STYLES (NETFLIX THEME)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root & background ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #141414; /* Netflix dark background */
    color: #e5e5e5; /* Netflix light gray text */
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #000000 !important; /* Pure black for sidebar */
    border-right: 1px solid #222222;
}

[data-testid="stSidebar"] .stRadio label {
    color: #aaaaaa !important;
    font-size: 0.88rem;
    padding: 6px 0;
    transition: color 0.2s;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: #e50914 !important; /* Netflix Red */
}

/* ── Metric cards ── */
.metric-card {
    background: #181818;
    border: 1px solid #222222;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: #e5091455;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #e50914);
    border-radius: 14px 14px 0 0;
}

.metric-icon {
    font-size: 1.8rem;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent, #e50914);
    line-height: 1;
    margin-bottom: 4px;
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #777777;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Section headers ── */
.section-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: #e5e5e5;
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #222222;
    margin-left: 12px;
}

/* ── Page title ── */
.page-header {
    padding: 28px 0 8px 0;
    border-bottom: 1px solid #222222;
    margin-bottom: 24px;
}

.page-header h1 {
    font-size: 1.9rem;
    font-weight: 700;
    color: #e5e5e5;
    margin: 0;
}

.page-header p {
    color: #777777;
    font-size: 0.9rem;
    margin: 6px 0 0 0;
}

/* ── Insight cards ── */
.insight-box {
    background: #181818;
    border: 1px solid #222222;
    border-left: 4px solid #e50914;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.88rem;
    color: #cccccc;
    line-height: 1.6;
}

.insight-box strong {
    color: #e5e5e5;
}

/* ── Prediction box ── */
.pred-positive {
    background: linear-gradient(135deg, #1c3d27 0%, #181818 100%);
    border: 1px solid #2ecc7144;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}

.pred-negative {
    background: linear-gradient(135deg, #3b1111 0%, #181818 100%);
    border: 1px solid #e5091444;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}

.pred-emoji {
    font-size: 3.5rem;
    margin-bottom: 8px;
}

.pred-label-pos {
    font-size: 1.4rem;
    font-weight: 700;
    color: #2ecc71;
}

.pred-label-neg {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e50914;
}

/* ── Tag badge ── */
.badge {
    display: inline-block;
    background: #2ecc7122;
    color: #2ecc71;
    border: 1px solid #2ecc7144;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}

.badge-red {
    background: #e5091422;
    color: #e50914;
    border-color: #e5091444;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Text area ── */
.stTextArea textarea {
    background: #181818 !important;
    border: 1px solid #222222 !important;
    color: #e5e5e5 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}

.stTextArea textarea:focus {
    border-color: #e50914 !important;
    box-shadow: 0 0 0 2px #e5091422 !important;
}

/* ── Button ── */
.stButton > button {
    background: #e50914 !important; /* Netflix Red Button */
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 12px 32px !important;
    font-size: 0.9rem !important;
    transition: background 0.2s, transform 0.15s !important;
}

.stButton > button:hover {
    background: #b80710 !important;
    transform: translateY(-2px) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #181818 !important;
    border-color: #222222 !important;
    color: #e5e5e5 !important;
    border-radius: 10px !important;
}

/* ── About cards ── */
.about-card {
    background: #181818;
    border: 1px solid #222222;
    border-radius: 14px;
    padding: 24px;
    height: 100%;
}

.about-card h3 {
    color: #e50914;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 12px 0;
}

.about-card p, .about-card li {
    color: #aaaaaa;
    font-size: 0.88rem;
    line-height: 1.7;
}

.about-card ul {
    padding-left: 16px;
    margin: 0;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: #e50914 !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB THEME (NETFLIX STYLE)
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#181818",
    "axes.facecolor":    "#181818",
    "axes.edgecolor":    "#222222",
    "axes.labelcolor":   "#aaaaaa",
    "xtick.color":       "#777777",
    "ytick.color":       "#777777",
    "text.color":        "#e5e5e5",
    "grid.color":        "#222222",
    "grid.alpha":        0.6,
})

# ─────────────────────────────────────────────
#  DATA & MODEL
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "label_Netflix.csv"
MODEL_PATH = BASE_DIR / "naive_bayes_model.pkl"
TFIDF_PATH = BASE_DIR / "tfidf.pkl"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model() -> Tuple[]:
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    return model, tfidf

df = load_data()
model, tfidf = load_model()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 28px 0;">
        <div style="font-size:2.5rem">🎬</div>
        <div style="font-weight:700; font-size:1.05rem; color:#e50914; margin-top:6px; letter-spacing: 0.05em;">NETFLIX ANALYTICS</div>
        <div style="font-size:0.72rem; color:#777777; margin-top:2px;">Sentiment Intelligence Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigasi",
        ["🏠  Dashboard", "☁️  WordCloud", "🎯  Evaluasi Model", "🤖  Prediksi", "📄  Dataset", "📚  Tentang"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="margin-top:auto; padding: 20px 0 8px 0; border-top:1px solid #222222;">
        <div style="font-size:0.72rem; color:#777777; text-align:center;">
            Naive Bayes · TF-IDF · 30K ulasan
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SHARED DATA
# ─────────────────────────────────────────────
total      = len(df)
positif    = int((df['sentiment'] == 1).sum())
negatif    = int((df['sentiment'] == 0).sum())
pct_pos    = positif / total * 100
pct_neg    = negatif / total * 100

CM = np.array([[1377, 208], [324, 4091]])

# ══════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════
if "Dashboard" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>🏠 Dashboard Analisis Sentimen</h1>
        <p>Gambaran umum distribusi sentimen ulasan pengguna aplikasi Netflix</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards ──
    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (c1, "📊", f"{total:,}", "Total Ulasan", "#e50914"),
        (c2, "😊", f"{positif:,}", "Sentimen Positif", "#2ecc71"),
        (c3, "😞", f"{negatif:,}", "Sentimen Negatif", "#e50914"),
        (c4, "🎯", "91.13%", "Akurasi Model", "#6c8eff"),
    ]

    for col, icon, value, label, color in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{color}">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value" style="color:{color}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Distribusi Sentimen</div>', unsafe_allow_html=True)

    col_chart, col_info = st.columns([3, 2], gap="large")

    with col_chart:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Bar chart
        ax = axes[0]
        bars = ax.bar(
            ["Positif", "Negatif"],
            [positif, negatif],
            color=["#2ecc71", "#e50914"],
            width=0.45,
            edgecolor="none",
            zorder=3
        )
        ax.set_ylabel("Jumlah Ulasan", fontsize=9)
        ax.set_title("Distribusi Kelas", fontsize=10, fontweight='600', pad=12)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['top','right','left']].set_visible(False)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 40,
                f"{int(bar.get_height()):,}",
                ha='center', va='bottom', fontsize=9, color='#e5e5e5', fontweight='600'
            )

        # Pie chart
        ax2 = axes[1]
        wedges, texts, autotexts = ax2.pie(
            [positif, negatif],
            labels=["Positif", "Negatif"],
            colors=["#2ecc71", "#e50914"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="#141414", linewidth=2),
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_color("#e5e5e5")
            at.set_fontsize(9)
            at.set_fontweight('600')
        for t in texts:
            t.set_color("#aaaaaa")
            t.set_fontsize(9)
        ax2.set_title("Proporsi Kelas", fontsize=10, fontweight='600', pad=12)

        fig.tight_layout(pad=2)
        st.pyplot(fig)

    with col_info:
        st.markdown("""
        <div class="insight-box">
            <strong>📌 Komposisi Dataset</strong><br>
            Dataset berisi ulasan pengguna Netflix dari Google Play Store yang sudah dilabeli secara manual ke dalam dua kelas sentimen.
        </div>
        """, unsafe_allow_html=True)

        pos_pct = f"{pct_pos:.1f}%"
        neg_pct = f"{pct_neg:.1f}%"

        st.markdown(f"""
        <div class="insight-box" style="border-left-color: #2ecc71">
            <strong>😊 Positif — {pos_pct}</strong><br>
            Sebagian besar pengguna menikmati kualitas streaming, variasi film, dan performa aplikasi.
        </div>
        <div class="insight-box">
            <strong>😞 Negatif — {neg_pct}</strong><br>
            Ulasan negatif umumnya menyoroti kendala langganan, error pemutaran video, atau masalah harga paket.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE: WORDCLOUD
# ══════════════════════════════════════════════
elif "WordCloud" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>☁️ Word Cloud Visualisasi</h1>
        <p>Kata-kata yang paling sering muncul dalam ulasan positif dan negatif</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["😊  Ulasan Positif", "😞  Ulasan Negatif"])

    def make_wordcloud(text, colormap, bg="#181818"):
        wc = WordCloud(
            width=1000,
            height=420,
            background_color=bg,
            colormap=colormap,
            max_words=120,
            prefer_horizontal=0.85,
            collocations=False,
            margin=8
        ).generate(text)
        return wc

    with tab1:
        positive_text = " ".join(df[df['sentiment'] == 1]['content'].dropna())
        wc = make_wordcloud(positive_text, "YlGn") # Menggunakan skema warna hijau-kuning cerah
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_facecolor("#181818")
        fig.tight_layout(pad=0)
        st.pyplot(fig)

        # Top words
        words = positive_text.lower().split()
        top_words = Counter(words).most_common(10)
        st.markdown('<div class="section-title">Top 10 Kata — Positif</div>', unsafe_allow_html=True)
        tw_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.barh(tw_df["Kata"][::-1], tw_df["Frekuensi"][::-1], color="#2ecc71", edgecolor="none")
        ax2.set_xlabel("Frekuensi", fontsize=9)
        ax2.xaxis.grid(True)
        ax2.set_axisbelow(True)
        ax2.spines[['top','right','bottom']].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)

    with tab2:
        negative_text = " ".join(df[df['sentiment'] == 0]['content'].dropna())
        wc = make_wordcloud(negative_text, "Reds") # Skema warna merah khas Netflix
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_facecolor("#181818")
        fig.tight_layout(pad=0)
        st.pyplot(fig)

        words = negative_text.lower().split()
        top_words = Counter(words).most_common(10)
        st.markdown('<div class="section-title">Top 10 Kata — Negatif</div>', unsafe_allow_html=True)
        tw_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.barh(tw_df["Kata"][::-1], tw_df["Frekuensi"][::-1], color="#e50914", edgecolor="none")
        ax2.set_xlabel("Frekuensi", fontsize=9)
        ax2.xaxis.grid(True)
        ax2.set_axisbelow(True)
        ax2.spines[['top','right','bottom']].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)

# ══════════════════════════════════════════════
#  PAGE: EVALUASI MODEL
# ══════════════════════════════════════════════
elif "Evaluasi" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>🎯 Evaluasi Model</h1>
        <p>Performa model Naive Bayes pada data uji</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric row ──
    TP, FN = CM[1][1], CM[1][0]
    FP, TN = CM[0][1], CM[0][0]

    accuracy  = (TP + TN) / CM.sum()
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1        = 2 * precision * recall / (precision + recall)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, color in [
        (m1, "Accuracy",  f"{accuracy*100:.2f}%",  "#6c8eff"),
        (m2, "Precision", f"{precision*100:.2f}%", "#2ecc71"),
        (m3, "Recall",    f"{recall*100:.2f}%",    "#f5a623"),
        (m4, "F1-Score",  f"{f1*100:.2f}%",        "#c47aff"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{color}">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)

    col_cm, col_report = st.columns([1, 1], gap="large")

    with col_cm:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            CM,
            annot=True,
            fmt='d',
            cmap='Reds', # Mengubah heatmap menjadi palet merah Netflix
            linewidths=1.5,
            linecolor="#141414",
            cbar=False,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold", "color": "#e5e5e5" if CM.max() > 4000 else "#141414"}
        )
        # Menyesuaikan warna teks anotasi manual agar kontras dengan tema gelap
        for text in ax.texts:
            if int(text.get_text()) > 2000:
                text.set_color('#ffffff')
            else:
                text.set_color('#e5e5e5')

        ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
        ax.set_ylabel("Actual Label", fontsize=10, labelpad=8)
        ax.set_xticklabels(["Negatif", "Positif"], fontsize=9)
        ax.set_yticklabels(["Negatif", "Positif"], fontsize=9, rotation=0)
        fig.tight_layout()
        st.pyplot(fig)

    with col_report:
        st.markdown("""
        <div class="insight-box" style="border-left-color:#2ecc71">
            <strong>True Positive (TP)</strong> — 4,091<br>
            Ulasan positif yang diprediksi benar sebagai positif.
        </div>
        <div class="insight-box" style="border-left-color:#2ecc71">
            <strong>True Negative (TN)</strong> — 1,377<br>
            Ulasan negatif yang diprediksi benar sebagai negatif.
        </div>
        <div class="insight-box">
            <strong>False Positive (FP)</strong> — 208<br>
            Ulasan negatif yang keliru diprediksi sebagai positif.
        </div>
        <div class="insight-box">
            <strong>False Negative (FN)</strong> — 324<br>
            Ulasan positif yang keliru diprediksi sebagai negatif.
        </div>
        """, unsafe_allow_html=True)

    # Classification report per class
    st.markdown('<div class="section-title">Laporan Per Kelas</div>', unsafe_allow_html=True)

    report_data = {
        "Kelas":     ["Negatif (0)", "Positif (1)"],
        "Precision": [f"{TN/(TN+FN)*100:.1f}%", f"{precision*100:.1f}%"],
        "Recall":    [f"{TN/(TN+FP)*100:.1f}%",  f"{recall*100:.1f}%"],
        "F1-Score":  ["—", f"{f1*100:.1f}%"],
        "Support":   [TN+FP, TP+FN],
    }
    st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
#  PAGE: PREDIKSI
# ══════════════════════════════════════════════
elif "Prediksi" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>🤖 Prediksi Sentimen for Netflix</h1>
        <p>Masukkan ulasan untuk diklasifikasikan oleh model</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        review = st.text_area(
            "Tulis ulasan Netflix di sini…",
            placeholder="Contoh: Film-filmnya lengkap dan tidak pernah buffering!",
            height=180
        )

        examples = [
            "Banyak pilihan serial seru dan kualitas videonya jernih",
            "Aplikasi sering hang di smart TV dan biaya langganan makin mahal",
            "Suka sekali dengan dokumenternya, subtitle juga akurat",
        ]

        st.markdown("<div style='font-size:0.8rem; color:#777777; margin:12px 0 6px 0;'>✨ Coba contoh ulasan:</div>", unsafe_allow_html=True)
        for ex in examples:
            if st.button(ex[:45] + "…" if len(ex) > 45 else ex, key=ex):
                review = ex

        predict_btn = st.button("🔍  Analisis Sentimen", use_container_width=True)

    with col_result:
        if predict_btn and review.strip():
            vector     = tfidf.transform([review])
            pred       = model.predict(vector)[0]
            prob       = model.predict_proba(vector)[0]
            confidence = max(prob) * 100

            if pred == 1:
                st.markdown(f"""
                <div class="pred-positive">
                    <div class="pred-emoji">😊</div>
                    <div class="pred-label-pos">Sentimen Positif</div>
                    <div style="color:#aaaaaa; font-size:0.82rem; margin-top:6px;">
                        Model yakin ulasan ini mengandung sentimen positif
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-negative">
                    <div class="pred-emoji">😞</div>
                    <div class="pred-label-neg">Sentimen Negatif</div>
                    <div style="color:#aaaaaa; font-size:0.82rem; margin-top:6px;">
                        Model yakin ulasan ini mengandung sentimen negatif
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:20px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span style="font-size:0.82rem; color:#aaaaaa; font-weight:500;">Confidence Score</span>
                    <span style="font-size:0.9rem; color:#e5e5e5; font-weight:700;">{confidence:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(confidence / 100)

            # Prob bars
            st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 1.5))
            classes = ["Negatif", "Positif"]
            colors  = ["#e50914", "#2ecc71"]
            bars = ax.barh(classes, [prob[0]*100, prob[1]*100], color=colors, edgecolor="none", height=0.45)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probabilitas (%)", fontsize=8)
            ax.spines[['top','right','bottom']].set_visible(False)
            for bar, p in zip(bars, [prob[0]*100, prob[1]*100]):
                ax.text(p + 1, bar.get_y() + bar.get_height()/2, f"{p:.1f}%", va='center', fontsize=8, color='#e5e5e5')
            fig.tight_layout()
            st.pyplot(fig)

        elif predict_btn and not review.strip():
            st.warning("⚠️ Masukkan ulasan terlebih dahulu.")
        else:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#777777;">
                <div style="font-size:3rem; margin-bottom:12px;">🤖</div>
                <div style="font-size:0.88rem;">Hasil prediksi akan muncul di sini</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE: DATASET
# ─────────────────────────────────────────────
elif "Dataset" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>📄 Dataset Explorer</h1>
        <p>Jelajahi dan filter dataset ulasan Netflix</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        search_term = st.text_input("🔍 Cari kata dalam ulasan", placeholder="Ketik kata kunci…")
    with c2:
        pilihan = st.selectbox("Filter Sentimen", ["Semua", "Positif 😊", "Negatif 😞"])
    with c3:
        n_rows = st.selectbox("Tampilkan", [50, 100, 200, 500, "Semua"], index=1)

    data = df.copy()
    if "Positif" in pilihan:
        data = data[data['sentiment'] == 1]
    elif "Negatif" in pilihan:
        data = data[data['sentiment'] == 0]

    if search_term:
        data = data[data['content'].str.contains(search_term, case=False, na=False)]

    if n_rows != "Semua":
        data = data.head(int(n_rows))

    st.markdown(f"""
    <div style="font-size:0.82rem; color:#777777; margin-bottom:12px;">
        Menampilkan <strong style="color:#e5e5e5">{len(data):,}</strong> baris
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        data.reset_index(drop=True),
        use_container_width=True,
        height=480
    )

    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️  Download CSV",
        data=csv,
        file_name="netflix_sentiment_filtered.csv",
        mime="text/csv"
    )

# ─────────────────────────────────────────────
#  PAGE: TENTANG
# ─────────────────────────────────────────────
elif "Tentang" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>📚 Tentang Penelitian</h1>
        <p>Informasi metodologi dan detail teknis proyek ini</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="about-card">
            <h3>🎯 Tujuan Penelitian</h3>
            <p>
                Mengklasifikasikan sentimen ulasan pengguna aplikasi Netflix dari Google Play Store
                ke dalam dua kelas: <strong style="color:#2ecc71">positif</strong> dan
                <strong style="color:#e50914">negatif</strong>, untuk membantu memahami
                persepsi pengguna secara otomatis menggunakan pendekatan machine learning.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="about-card">
            <h3>📊 Dataset</h3>
            <ul>
                <li>Sumber: Google Play Store</li>
                <li>Total: <strong style="color:#e5e5e5">30.000 ulasan</strong></li>
                <li>Label: Positif & Negatif</li>
                <li>Metode labeling: Semi-otomatis + verifikasi manual</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown("""
        <div class="about-card">
            <h3>⚙️ Pipeline NLP</h3>
            <ul>
                <li>Preprocessing: case folding, stopword removal, stemming</li>
                <li>Ekstraksi fitur: <strong style="color:#e5e5e5">TF-IDF Vectorizer</strong></li>
                <li>Classifier: <strong style="color:#e5e5e5">Multinomial Naive Bayes</strong></li>
                <li>Validasi: train-test split 80:20</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="about-card">
            <h3>📈 Hasil Evaluasi</h3>
            <ul>
                <li>Accuracy: <strong style="color:#6c8eff">91.13%</strong></li>
                <li>Precision: <strong style="color:#2ecc71">95.2%</strong></li>
                <li>Recall: <strong style="color:#f5a623">92.7%</strong></li>
                <li>F1-Score: <strong style="color:#c47aff">93.9%</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#181818; border:1px solid #222222; border-radius:14px; padding:20px 24px; text-align:center;">
        <span style="font-size:0.8rem; color:#777777;">
            Dibuat dengan ❤️ menggunakan Streamlit · Naive Bayes · TF-IDF
        </span>
    </div>
    """, unsafe_allow_html=True)
