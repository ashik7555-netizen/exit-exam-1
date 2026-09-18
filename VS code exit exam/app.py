import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

project = {
    "title": "Enterprise Revenue Prediction Dashboard",
    "subtitle": "A machine learning project for forecasting company revenue using structured business data.",
    "overview": "This project analyzes enterprise and company information to predict revenue using a Ridge Regression model. It processes missing values, encodes categorical fields, trains on historical data, and saves the trained model for later use.",
    "goals": [
        "Clean and prepare company data for model training",
        "Handle missing numerical and categorical values effectively",
        "Train a reliable predictive model for revenue estimation",
        "Save the model and feature list for future predictions",
    ],
    "features": [
        "Data preprocessing and feature engineering",
        "Categorical one-hot encoding",
        "Train/test split for model validation",
        "Ridge regression for predictive modeling",
        "Model serialization with joblib",
    ],
    "dataset": "The project uses a structured dataset from G3.csv containing enterprise-level company metadata and financial indicators.",
    "status": "Ready for model training and deployment",
}


def load_model():
    if os.path.exists(MODEL_PATH):
        payload = joblib.load(MODEL_PATH)
        if isinstance(payload, dict) and "model" in payload:
            return payload["model"], payload.get("feature_names", [])
    return None, []


def estimate_prediction(form_data):
    model, feature_names = load_model()

    if model is None:
        employees = float(form_data.get("employees", 0) or 0)
        market_cap = float(form_data.get("market_cap", 0) or 0)
        estimate = max(0.0, (employees * 1800) + (market_cap * 0.65))
        return estimate, "Demo estimate (the trained model is not available yet)."

    feature_values = {}
    for feature in feature_names:
        value = form_data.get(feature, 0)
        if value in (None, ""):
            value = 0
        try:
            feature_values[feature] = float(value)
        except ValueError:
            feature_values[feature] = 0

    input_df = pd.DataFrame([feature_values], columns=feature_names)
    prediction = model.predict(input_df)[0]
    return float(prediction), "Prediction based on the trained model."


@app.route("/")
def home():
    return render_template("index.html", project=project)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        result = estimate_prediction(request.form.to_dict())
    return render_template("predict.html", project=project, result=result)


@app.route("/result", methods=["POST"])
def result():
    result_data = estimate_prediction(request.form.to_dict())
    return render_template("result.html", project=project, result=result_data)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
