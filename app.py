from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import numpy as np

app = Flask(__name__)

# Load model, encoder, and symptoms
model = joblib.load("model/model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
symptom_columns = joblib.load("model/symptom_columns.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    selected_symptoms = []

    if request.method == "POST":
        # Collect selected symptoms
        selected_symptoms = request.form.getlist("symptoms")

        # Create input vector (0/1 for all symptoms)
        values = [1 if symptom in selected_symptoms else 0 for symptom in symptom_columns]
        input_df = pd.DataFrame([values], columns=symptom_columns)

        # Get probability predictions
        probs = model.predict_proba(input_df)[0]

        # Top 3 classes
        top_indices = np.argsort(probs)[::-1][:3]
        predictions = [(label_encoder.inverse_transform([i])[0], round(probs[i] * 100, 2)) for i in top_indices]

    return render_template("index.html", symptoms=symptom_columns,
                           selected_symptoms=selected_symptoms, predictions=predictions)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
 