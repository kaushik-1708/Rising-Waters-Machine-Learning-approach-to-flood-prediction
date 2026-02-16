from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "floods.save")
SCALER_PATH = os.path.join(BASE_DIR, "transform.save")

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = joblib.load(SCALER_PATH)

FEATURES = [
    "Temp", "Humidity", "Cloud Cover", "ANNUAL",
    "Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec",
    "avgjune", "sub"
]

@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        vals = []
        for f in FEATURES:
            v = request.form.get(f)
            vals.append(float(v))

        x = np.array(vals).reshape(1, -1)
        x_scaled = scaler.transform(x)

        pred = int(model.predict(x_scaled)[0])  # 1 = flood, 0 = no flood

        if pred == 1:
            return render_template("chance.html")
        else:
            return render_template("nochance.html")

    except Exception as e:
        return render_template("index.html", features=FEATURES, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
