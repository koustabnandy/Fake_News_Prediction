import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Ensure project root on sys.path so 'utils' imports work regardless of run dir
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.data_utils import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

MODEL_PATH = os.path.join("models", "fake_news_pipeline.pkl")

st.title("📰 Fake News Detection")
st.write("Paste a headline or short article and I'll predict whether it's **fake** or **real**.")

try:
    cache_resource = st.cache_resource
except Exception:
    # Fallback for older Streamlit versions
    cache_resource = st.cache

@cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    # Train a quick model if not present (uses available data)
    df = load_dataset("data")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1,2),
            max_df=0.9,
            min_df=1
        )),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipeline.fit(X_train, y_train)
    # Save for reuse
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline

model = load_or_train_model()

text_input = st.text_area("Enter text:", height=160, placeholder="e.g. Breaking: Eating chocolate cures all diseases overnight!")
col1, col2 = st.columns([1,1])

with col1:
    if st.button("Predict"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            pred = model.predict([text_input])[0]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([text_input])[0]
                # Determine class order from pipeline's final estimator if needed
                if hasattr(model, "classes_"):
                    classes = model.classes_
                elif hasattr(model, "named_steps") and "clf" in model.named_steps and hasattr(model.named_steps["clf"], "classes_"):
                    classes = model.named_steps["clf"].classes_
                else:
                    classes = None
                if classes is not None and pred in classes:
                    idx = list(classes).index(pred)
                    confidence = float(probs[idx])
                else:
                    confidence = float(max(probs))
                st.write(f"**Prediction:** `{pred.upper()}`\n**Confidence:** {confidence:.2%}")
            else:
                st.write(f"**Prediction:** `{pred.upper()}`")

with col2:
    if st.button("Train/Re-train model"):
        with st.spinner("Training..."):
            # Force re-train from current data
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.linear_model import LogisticRegression
            from sklearn.feature_extraction.text import TfidfVectorizer

            df = load_dataset("data")
            X_train, X_test, y_train, y_test = train_test_split(
                df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
            )
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1,2),
                    max_df=0.9,
                    min_df=1
                )),
                ("clf", LogisticRegression(max_iter=200))
            ])
            pipeline.fit(X_train, y_train)
            os.makedirs("models", exist_ok=True)
            joblib.dump(pipeline, MODEL_PATH)
            st.success("Model trained and saved!")
            try:
                cache_resource.clear()
            except Exception:
                pass
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

st.markdown("---")
st.caption("Tip: Replace `data/sample_news.csv` with your own data at `data/news.csv` (columns: text,label). Then click **Train/Re-train model**.")
