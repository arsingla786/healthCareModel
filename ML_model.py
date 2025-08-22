import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from catboost import CatBoostClassifier
import joblib
import os

# Load dataset
df = pd.read_csv("C:\\Users\\Arnav Singla\\Downloads\\cleaned_disease_dataset .csv")

# OPTIONAL: Add small noise to make data less "perfect"
# (simulate real-world patient variation)
for col in df.columns[:-1]:  # skip target column
    noise = np.random.binomial(1, 0.05, size=len(df))  # 5% random flips
    df[col] = np.abs(df[col] - noise)

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

# CatBoost with stronger regularization
cat_model = CatBoostClassifier(
    iterations=500,
    depth=3,                    # shallower trees
    learning_rate=0.03,
    l2_leaf_reg=15,             # stronger regularization
    loss_function='MultiClass',
    eval_metric='MultiClass',
    random_seed=42,
    od_type="Iter",             # early stopping
    od_wait=30,
    verbose=100
)

# Train
cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

# Evaluate
y_pred = cat_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Cross-validation for more realistic evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(cat_model, X, y_encoded, cv=cv, scoring='accuracy')
print("\nCross-validation scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Save model & encoder
os.makedirs('model', exist_ok=True)
joblib.dump(cat_model, 'model/model.pkl')
joblib.dump(le, 'model/label_encoder.pkl')
joblib.dump(list(X.columns), 'model/symptom_columns.pkl')




