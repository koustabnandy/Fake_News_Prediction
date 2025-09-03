# Fake News Detection – Complete Web App Project

This project gives you everything you need to train a **Fake News Detection** model and run it as a **web app** (Streamlit or Flask).

## Quick Start

### 1) Create environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Prepare data (choose one)
- Put a single combined CSV at `data/news.csv` with columns:
  - `text` (string) – the article/headline
  - `label` (string) – `"fake"` or `"real"`
- **OR** place Kaggle-style files: `data/Fake.csv` and `data/True.csv` (with a `text` column).
- If you don't have data yet, the small `data/sample_news.csv` will be used for a minimal demo.

### 3) Train the model
You can use either the **Jupyter Notebook** or the **script**:

- Notebook:
  ```bash
  jupyter notebook notebooks/01_train_fake_news_model.ipynb
  ```

- Script:
  ```bash
  python train.py
  ```

This will create: `models/fake_news_pipeline.pkl`

### 4) Run the web app (Streamlit – recommended)
```bash
streamlit run app_streamlit/app.py
```

### 5) Optional: Run the Flask app
```bash
python app_flask/app.py
# open http://127.0.0.1:5000
```

---

## Project Structure

```
fake-news-webapp/
├─ data/
│  ├─ news.csv                 # (optional) combined dataset: text,label
│  ├─ Fake.csv                 # (optional) Kaggle-style fake news
│  ├─ True.csv                 # (optional) Kaggle-style true news
│  └─ sample_news.csv          # tiny fallback dataset
├─ models/
│  └─ fake_news_pipeline.pkl   # created after training
├─ notebooks/
│  └─ 01_train_fake_news_model.ipynb
├─ app_streamlit/
│  └─ app.py
├─ app_flask/
│  ├─ app.py
│  └─ templates/
│     └─ index.html
├─ utils/
│  └─ data_utils.py
├─ train.py
├─ requirements.txt
└─ README.md
```

---

## Notes

- The model is a **scikit-learn Pipeline**: `TfidfVectorizer` + `LogisticRegression`.
- The Streamlit app will **auto-train** on first run if it can't find `models/fake_news_pipeline.pkl` (using available data).
- For production, consider better data cleaning, class balance handling, and model evaluation.
