import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

import pickle
from joblib import dump


# -----------------------------
# 0) Robust paths (FIX)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../RisingWaters/Training
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "flood dataset.xlsx")
FLASK_DIR = os.path.join(BASE_DIR, "..", "Flask")

SCALER_PATH = os.path.join(FLASK_DIR, "transform.save")
MODEL_PATH = os.path.join(FLASK_DIR, "floods.save")

print("Working directory:", os.getcwd())
print("Looking for dataset at:", os.path.abspath(DATA_PATH))

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at: {os.path.abspath(DATA_PATH)}")


# -----------------------------
# 1) Load dataset
# -----------------------------
dataset = pd.read_excel(DATA_PATH)

print("\nDataset shape:", dataset.shape)
print("Columns:", list(dataset.columns))
print(dataset.head())


# -----------------------------
# 2) Split X and y
# -----------------------------
if "flood" not in dataset.columns:
    raise ValueError(f"'flood' column not found. Available columns: {list(dataset.columns)}")

X = dataset.drop("flood", axis=1).values
y = dataset["flood"].values


# -----------------------------
# 3) Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10
)


# -----------------------------
# 4) Standard scaling + save scaler
# -----------------------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

dump(sc, SCALER_PATH)
print("\nSaved scaler:", SCALER_PATH)


# -----------------------------
# 5) Train models
# -----------------------------
dtree = DecisionTreeClassifier(random_state=10)
rf = RandomForestClassifier(random_state=10, n_estimators=300)
knn = KNeighborsClassifier(n_neighbors=7)

xgb_model = xgb.XGBClassifier(
    eval_metric="logloss",
    random_state=10,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4
)

models = {
    "DecisionTree": dtree,
    "RandomForest": rf,
    "KNN": knn,
    "XGBoost": xgb_model
}

scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    scores[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

best_name = max(scores, key=scores.get)
best_model = models[best_name]

print("\nBest model:", best_name, "Accuracy:", scores[best_name])


# -----------------------------
# 6) Final evaluation
# -----------------------------
final_pred = best_model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, final_pred))
print("\nClassification Report:\n", classification_report(y_test, final_pred))


# -----------------------------
# 7) Save best model to Flask folder
# -----------------------------
pickle.dump(best_model, open(MODEL_PATH, "wb"))
print("\nSaved model:", MODEL_PATH)
