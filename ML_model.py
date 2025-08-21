import pandas as pd
print(1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("C:\\Users\\Arnav Singla\\Downloads\\disease_dataset.csv")

# Features and target
X = df.drop('PredictedDisease', axis=1)
y = df['PredictedDisease']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train CatBoost
cat_model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.05,
    loss_function='MultiClass',
    random_seed=42,
    verbose=50
)
cat_model.fit(X_train, y_train)

# Evaluate
y_pred = cat_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model & encoder
os.makedirs('model', exist_ok=True)
joblib.dump(cat_model, 'model/model.pkl')
joblib.dump(le, 'model/label_encoder.pkl')
joblib.dump(list(X.columns), 'model/symptom_columns.pkl')  # save symptom names
