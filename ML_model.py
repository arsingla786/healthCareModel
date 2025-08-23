import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("C:\\Users\\Arnav Singla\\Downloads\\new_balanced_data (1).csv")
print(df.head())

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

# Random Forest Model (tuned for better generalization)
rf_model = RandomForestClassifier(
    n_estimators=300,       # number of trees
    max_depth=12,           # deeper trees capture more patterns
    min_samples_split=4,    # control overfitting
    min_samples_leaf=2,     # prevent tiny leaves
    class_weight="balanced",# handle class imbalance
    random_state=42,
    n_jobs=-1
)

# Train
rf_model.fit(X_train, y_train)
 
# Evaluate
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Cross-validation for more realistic evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
print("\nCross-validation scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Save model & encoder (same filenames as CatBoost version)
os.makedirs('model', exist_ok=True)
joblib.dump(rf_model, 'model/model.pkl')
joblib.dump(le, 'model/label_encoder.pkl')
joblib.dump(list(X.columns), 'model/symptom_columns.pkl')
