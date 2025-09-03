import os
import pandas as pd

def load_dataset(data_dir: str) -> pd.DataFrame:
    """Load dataset from one of the supported formats.
    Priority:
      1) data/news.csv  with columns: text,label (labels: 'fake'/'real')
      2) data/Fake.csv & data/True.csv (Kaggle-style) with a 'text' column
      3) data/sample_news.csv (tiny fallback)
    Returns a DataFrame with columns: text (str), label (str).
    """
    # 1) combined file
    news_path = os.path.join(data_dir, "news.csv")
    if os.path.exists(news_path):
        df = pd.read_csv(news_path)
        # Normalize columns
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("data/news.csv must have columns: text,label")
        # normalize labels
        df["label"] = df["label"].str.strip().str.lower().map({"fake": "fake", "real": "real"})
        if df["label"].isna().any():
            raise ValueError("Labels must be 'fake' or 'real'.")
        return df[["text", "label"]].dropna()

    # 2) separate files
    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")
    if os.path.exists(fake_path) and os.path.exists(true_path):
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)
        # Attempt to find a 'text' column
        def pick_text(df):
            for c in df.columns:
                if c.strip().lower() in {"text", "content", "article", "body", "title"}:
                    return df[c]
            raise ValueError("Could not find a text-like column in dataframe.")
        df_fake = pd.DataFrame({"text": pick_text(df_fake), "label": "fake"})
        df_true = pd.DataFrame({"text": pick_text(df_true), "label": "real"})
        df = pd.concat([df_fake, df_true], ignore_index=True)
        df = df.dropna().reset_index(drop=True)
        return df

    # 3) fallback
    sample_path = os.path.join(data_dir, "sample_news.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        df = df.rename(columns={c: c.strip().lower() for c in df.columns})
        return df[["text", "label"]].dropna()

    raise FileNotFoundError("No dataset found. Provide data/news.csv or Fake.csv & True.csv or sample_news.csv.")
