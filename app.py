import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re
import string
import time
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
import nltk

from tensorflow import keras
from nltk.corpus import stopwords


# Page Config

st.set_page_config(
    page_title="TruthLens",
    page_icon="📰",
    layout="wide"
)

# Download Stopwords Once


try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


# Cache Stopwords

@st.cache_resource
def load_stopwords():

    extra = {
        "reuters","washington","said","monday","tuesday",
        "wednesday","thursday","friday","saturday","sunday",
        "jan","feb","mar","apr","jun","jul","aug",
        "sep","oct","nov","dec","mr","mrs","rep","sen",
        "image","images","video","wire","via"
    }

    return set(stopwords.words("english")).union(extra)

STOPWORDS = load_stopwords()

# --------------------------------------------------
# Regex
# --------------------------------------------------

URL_PATTERN = re.compile(r"http\S+|www\S+")

NUMBER_PATTERN = re.compile(r"\d+")

SPACE_PATTERN = re.compile(r"\s+")

PUNCT_TABLE = str.maketrans(
    "",
    "",
    string.punctuation
)

# --------------------------------------------------
# Load ANN Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = keras.models.load_model(
        "ann_model.keras",
        compile=False
    )

    tfidf = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    dummy = np.zeros(
        (1,10000),
        dtype=np.float32
    )

    model(dummy,training=False)

    return model,tfidf

model,tfidf = load_model()

# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------

def clean_text(text):

    text=text.lower()

    text=URL_PATTERN.sub("",text)

    text=text.translate(PUNCT_TABLE)

    text=NUMBER_PATTERN.sub("",text)

    text=SPACE_PATTERN.sub(" ",text).strip()

    words=[
        word
        for word in text.split()
        if word not in STOPWORDS
    ]

    return " ".join(words)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_news(text):

    start=time.perf_counter()

    cleaned=clean_text(text)

    features=tfidf.transform(
        [cleaned]
    ).toarray().astype(np.float32)

    probability=float(
        model(
            features,
            training=False
        ).numpy()[0][0]
    )

    label="REAL"

    if probability<0.5:
        label="FAKE"

    confidence=probability

    if label=="FAKE":
        confidence=1-probability

    end=time.perf_counter()

    return{

        "label":label,

        "confidence":confidence,

        "probability":probability,

        "cleaned":cleaned,

        "time":round(
            (end-start)*1000,
            2
        )
    }

# --------------------------------------------------
# Session
# --------------------------------------------------

if "result" not in st.session_state:

    st.session_state.result=None



# ==========================================================
# Custom Styling
# ==========================================================

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1200px;
}

h1,h2,h3{
    color:white;
}

.stApp{
    background:#0E1117;
}

textarea{
    font-size:17px !important;
}

div[data-testid="stMetric"]{
    background:#1A1C23;
    padding:15px;
    border-radius:12px;
    border:1px solid #30363D;
}

.stButton>button{
    width:100%;
    height:50px;
    font-size:17px;
    font-weight:bold;
    border-radius:10px;
}

.result-box{
    padding:25px;
    border-radius:15px;
    text-align:center;
}

.real{
    background:#103B2C;
    border:2px solid #2ECC71;
}

.fake{
    background:#4B1E1E;
    border:2px solid #E74C3C;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    st.title("📌 TruthLens")

    st.write("---")

    st.subheader("Model")

    st.write("• ANN (TensorFlow/Keras)")
    st.write("• TF-IDF (10,000 Features)")
    st.write("• Binary Classification")

    st.write("---")

    st.subheader("Dataset")

    st.write("44,898 News Articles")

    st.write("2017–2018")

    st.write("---")

    st.info(
        "This project is for educational purposes."
    )

# ==========================================================
# Hero Section
# ==========================================================

st.title("📰 TruthLens")

st.caption(
    "AI Powered Fake News Detection using Artificial Neural Networks"
)

st.write("")

# ==========================================================
# Metrics
# ==========================================================

m1,m2,m3,m4=st.columns(4)

with m1:
    st.metric(
        "Articles",
        "44,898"
    )

with m2:
    st.metric(
        "TF-IDF",
        "10K"
    )

with m3:
    st.metric(
        "Model",
        "ANN"
    )

with m4:
    st.metric(
        "Accuracy",
        "99%"
    )

st.write("")

# ==========================================================
# Input Section
# ==========================================================

st.subheader("Paste News Article")

news=st.text_area(

    label="",

    placeholder="""
Paste a news article or headline here...

Example:

Scientists discovered a new treatment...

""",

    height=250

)

# ==========================================================
# Statistics
# ==========================================================

if news.strip():

    total_characters=len(news)

    total_words=len(news.split())

    estimated_read=max(
        1,
        total_words//200
    )

    c1,c2,c3=st.columns(3)

    with c1:
        st.metric(
            "Characters",
            total_characters
        )

    with c2:
        st.metric(
            "Words",
            total_words
        )

    with c3:
        st.metric(
            "Reading Time",
            f"{estimated_read} min"
        )

st.write("")

# ==========================================================
# Buttons
# ==========================================================

b1,b2=st.columns([4,1])

with b1:

    analyze=st.button(
        "🔍 Analyze Article",
        type="primary"
    )

with b2:

    clear=st.button("Clear")

if clear:

    st.rerun()



# ==========================================================
# Prediction
# ==========================================================

if analyze:

    if not news.strip():

        st.warning("⚠️ Please paste a news article.")

    else:

        with st.spinner("Analyzing article..."):

            result = predict_news(news)

        st.session_state.result = result


# ==========================================================
# Show Result
# ==========================================================

if st.session_state.result is not None:

    result = st.session_state.result

    st.divider()

    left, right = st.columns([2, 1])

    # -----------------------------
    # Left Column
    # -----------------------------

    with left:

        if result["label"] == "REAL":

            st.markdown(
                """
                <div class="result-box real">
                    <h1>✅ REAL NEWS</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="result-box fake">
                    <h1>❌ FAKE NEWS</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        st.subheader("Confidence")

        st.progress(result["confidence"])

        st.write(
            f"**Confidence : {result['confidence']*100:.2f}%**"
        )

        st.write("")

        st.subheader("Probability")

        st.write(
            f"Raw Model Output : **{result['probability']:.4f}**"
        )

        st.write("")

        st.subheader("Inference Time")

        st.success(
            f"{result['time']} ms"
        )

    # -----------------------------
    # Right Column
    # -----------------------------

    with right:

        st.metric(
            "Prediction",
            result["label"]
        )

        st.metric(
            "Confidence",
            f"{result['confidence']*100:.2f}%"
        )

        st.metric(
            "Time",
            f"{result['time']} ms"
        )

        st.metric(
            "Probability",
            f"{result['probability']:.4f}"
        )

# ==========================================================
# Cleaned Text
# ==========================================================

if st.session_state.result is not None:

    with st.expander("🔍 View Preprocessed Text"):

        st.write(
            st.session_state.result["cleaned"]
        )

# ==========================================================
# About Model
# ==========================================================

st.divider()

st.subheader("About This Model")

st.info(
"""
### TruthLens

**Model**
- Artificial Neural Network (TensorFlow/Keras)

**Input Features**
- TF-IDF Vectorizer
- 10,000 Features

**Dataset**
- 44,898 News Articles

**Output**
- REAL
- FAKE

This application predicts whether a news article is likely to be genuine or fake based on the text content.
"""
)

# ==========================================================
# Footer
# ==========================================================

st.markdown(
"""
<div class="footer">

Made with ❤️ using Streamlit, TensorFlow & Scikit-Learn

<br><br>

TruthLens © 2026

</div>
""",
unsafe_allow_html=True
)
