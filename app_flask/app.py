from flask import Flask, request, render_template
import os, joblib

app = Flask(__name__)
MODEL_PATH = os.path.join("..", "models", "fake_news_pipeline.pkl")

def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run `python train.py` first.")
    return joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    text = ""
    if request.method == "POST":
        text = request.form.get("news", "")
        model = get_model()
        pred = model.predict([text])[0]
        try:
            probs = model.predict_proba([text])[0]
            idx = list(model.classes_).index(pred)
            confidence = probs[idx]
        except Exception:
            confidence = None
        prediction = pred.upper()
    return render_template("index.html", prediction=prediction, confidence=confidence, text=text)

if __name__ == "__main__":
    app.run(debug=True)

#Set-Location "d:\programs\Python\fake-news-webapp\app_flask"; python app.pyS