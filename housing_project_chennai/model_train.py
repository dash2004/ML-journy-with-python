import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# 1. Load Data
print("Loading Real Dataset...")
# Try reading with common encodings if default fails
try:
    df = pd.read_csv('data/clean_data.csv')
except:
    df = pd.read_csv('data/clean_data.csv', encoding='ISO-8859-1')

# 2. Data Cleaning (The "Real World" part)
print("Cleaning Data...")

# Select only the columns we need for the App
# Columns: price, area, status, bhk, bathroom, age, location, builder
useful_cols = ['location', 'area', 'bhk', 'bathroom', 'age', 'price']
df = df[useful_cols]

# Handle Missing Values
df['bhk'] = df['bhk'].fillna(df['bhk'].median())
df['bathroom'] = df['bathroom'].fillna(df['bathroom'].median())
df['age'] = df['age'].fillna(df['age'].median())

# Clean Text Data (Fix potential spelling variations)
df['location'] = df['location'].str.strip().str.title()

# 3. Preprocessing
X = df.drop('price', axis=1)
y = df['price']

# One-Hot Encoding for Locations
X = pd.get_dummies(X, columns=['location'])

# 4. Train Model
print("Training Random Forest Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
print(f"✅ Model Accuracy (R2 Score): {r2_score(y_test, predictions):.2f}")

# 6. Save Model & Column Info
# We MUST save the column names so the App knows the exact order
model_data = {
    'model': model, 
    'columns': X_train.columns.tolist(),
    'locations': df['location'].unique().tolist()  # Save list of locations for the dropdown
}
joblib.dump(model_data, 'chennai_real_model.pkl')
print("Model saved as 'chennai_real_model.pkl'")