import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from utils.data_utils import load_dataset

DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_pipeline.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")
df = load_dataset(DATA_DIR)
print(f"Loaded {len(df)} rows.")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2),
        max_df=1.0,   # allow all terms
        min_df=1      # keep terms that appear at least once
    )),
    ("clf", LogisticRegression(max_iter=200))
])


print("Training model...")
pipeline.fit(X_train, y_train)

print("Evaluating...")
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

print(f"Saving model to {MODEL_PATH}")
joblib.dump(pipeline, MODEL_PATH)
print("Done.")
