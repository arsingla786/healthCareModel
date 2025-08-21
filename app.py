from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model, encoder, and symptoms
model = joblib.load("model/model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
symptom_columns = joblib.load("model/symptom_columns.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_disease = None

    if request.method == "POST":
        # Collect selected symptoms
        selected_symptoms = request.form.getlist("symptoms")

        # Create input vector (0/1 for all symptoms)
        values = [1 if symptom in selected_symptoms else 0 for symptom in symptom_columns]
        input_df = pd.DataFrame([values], columns=symptom_columns)

        # Predict disease
        pred_class = int(model.predict(input_df).flatten()[0])
        predicted_disease = label_encoder.inverse_transform([pred_class])[0]

    return render_template("index.html", symptoms=symptom_columns, prediction=predicted_disease)


if __name__ == "__main__":
    app.run(debug=True)
