import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Load raw data
print("Loading raw data...")
df = pd.read_csv('data/raw/heart.csv')
print(f"Original dataset shape: {df.shape}")

# Remove duplicates
df = df.drop_duplicates()

# NEW: Remove outliers using IQR method
print("\n=== Removing Outliers ===")
numerical_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    before = len(df)
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    after = len(df)
    print(f"{col}: Removed {before - after} outliers")

print(f"Dataset shape after outlier removal: {df.shape}")

# Separate features and target
X = df.drop('output', axis=1)
y = df['output']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=params['preprocess']['test_size'],
    random_state=params['preprocess']['random_state'],
    stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Save processed data
os.makedirs('data/processed', exist_ok=True)

X_train_scaled.to_csv('data/processed/X_train.csv', index=False)
X_test_scaled.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("\nPreprocessing complete!")
print(f"Training set size: {X_train_scaled.shape[0]}")
print(f"Test set size: {X_test_scaled.shape[0]}")