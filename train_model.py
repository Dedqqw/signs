import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load CSV dataset created by collect_data.py
csv_file = "hand_landmarks_dataset.csv"
df = pd.read_csv(csv_file)

# Separate features (63 landmark coordinates) and target labels
X = df.drop(columns=['label']).values
y = df['label'].values

# Split into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Save trained model file
model_filename = "asl_model.p"
joblib.dump(model, model_filename)
print(f"Model saved successfully as '{model_filename}'!")