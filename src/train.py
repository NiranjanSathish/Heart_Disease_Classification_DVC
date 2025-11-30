import pandas as pd
import yaml
import json
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Load processed data
print("Loading processed data...")
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# Train model
print("Training model...")
model = RandomForestClassifier(
    n_estimators=params['train']['n_estimators'],
    max_depth=params['train']['max_depth'],
    random_state=params['train']['random_state']
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'f1_score': float(f1_score(y_test, y_pred)),
    'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
}

print("\nModel Performance:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Save model
os.makedirs('models', exist_ok=True)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved to models/model.pkl")

# Save metrics
os.makedirs('metrics', exist_ok=True)
with open('metrics/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("Metrics saved to metrics/metrics.json")