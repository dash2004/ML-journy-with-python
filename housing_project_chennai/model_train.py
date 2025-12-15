"""
Chennai Housing Price Prediction - Model Training
==================================================
This script trains a Random Forest model using advanced feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering
from feature_engineering import (
    create_derived_features, 
    get_feature_columns,
    get_all_feature_categories
)

print("=" * 70)
print("CHENNAI HOUSING PRICE PREDICTION - MODEL TRAINING")
print("=" * 70)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n[1/6] Loading Dataset...")
try:
    df = pd.read_csv('data/clean_data.csv')
except:
    df = pd.read_csv('data/clean_data.csv', encoding='ISO-8859-1')

print(f"    Loaded {len(df)} records with {len(df.columns)} columns")

# ============================================
# 2. DATA CLEANING
# ============================================
print("\n[2/6] Cleaning Data...")

# Handle Missing Values
df['bhk'] = df['bhk'].fillna(df['bhk'].median())
df['bathroom'] = df['bathroom'].fillna(df['bathroom'].median())
df['age'] = df['age'].fillna(df['age'].median())

# Clean Text Data
df['location'] = df['location'].str.strip().str.title()
df['status'] = df['status'].str.strip()
df['builder'] = df['builder'].str.strip()

print(f"    Missing values handled")
print(f"    Text data cleaned")

# ============================================
# 3. FEATURE ENGINEERING
# ============================================
print("\n[3/6] Creating Derived Features...")

# Apply feature engineering
df_featured = create_derived_features(df)

print(f"    Original features: {len(df.columns)}")
print(f"    After engineering: {len(df_featured.columns)}")
print(f"    New features added: {len(df_featured.columns) - len(df.columns)}")

# Show feature categories
categories = get_all_feature_categories()
print(f"\n    Feature Categories:")
for cat_name, features in categories.items():
    print(f"      - {cat_name}: {len(features)} features")

# ============================================
# 4. PREPARE FEATURES FOR TRAINING
# ============================================
print("\n[4/6] Preparing Features...")

# Target variable
y = df_featured['price']

# Select features for training
# Use base features + derived features (excluding price-related leak features)
base_features = ['area', 'bhk', 'bathroom', 'age']

# Derived features (excluding features that would cause data leakage)
derived_features = [
    # Location Intelligence
    'distance_to_beach', 'distance_to_airport', 'distance_to_railway',
    'distance_to_it_corridor', 'distance_to_city_center',
    'connectivity_score', 'locality_tier',
    'near_it_corridor', 'near_beach', 'near_airport', 'is_central',
    'is_premium_location', 'is_suburban', 'is_it_hub',
    
    # Room-based
    'bathroom_bhk_ratio', 'total_rooms', 'area_per_bhk', 'area_per_room',
    'is_well_equipped',
    
    # Age-based
    'is_new_construction', 'is_established', 'is_old',
    
    # Status-based
    'is_ready', 'is_under_construction',
    
    # Builder-based
    'is_premium_builder',
    
    # Property Characteristics
    'is_spacious', 'log_area',
    'bedroom_density', 'is_spacious_per_bhk',
    'is_compact', 'is_family_home', 'is_studio_single',
    
    # Geospatial (location clusters)
    'location_cluster'
]

# Filter to only available features
available_derived = [f for f in derived_features if f in df_featured.columns]

print(f"    Base features: {len(base_features)}")
print(f"    Derived features: {len(available_derived)}")

# Create feature matrix
X = df_featured[base_features + available_derived].copy()

# One-Hot Encode location
location_dummies = pd.get_dummies(df_featured['location'], prefix='location')
X = pd.concat([X, location_dummies], axis=1)

print(f"    Location categories: {len(location_dummies.columns)}")
print(f"    Total features for training: {len(X.columns)}")

# ============================================
# 5. TRAIN MODEL
# ============================================
print("\n[5/6] Training Model...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"    Training samples: {len(X_train)}")
print(f"    Test samples: {len(X_test)}")

# Train Random Forest with optimized parameters
print("\n    Training Random Forest...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# ============================================
# 6. EVALUATE MODEL
# ============================================
print("\n[6/6] Evaluating Model...")

# Metrics
train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print(f"\n    [RESULTS]")
print(f"    ----------------------------------------")
print(f"    Training R2 Score:  {train_r2:.4f}")
print(f"    Test R2 Score:      {test_r2:.4f}")
print(f"    Mean Absolute Error: Rs.{test_mae:.2f} Lakhs")
print(f"    RMSE:               Rs.{test_rmse:.2f} Lakhs")

# Cross-validation
print("\n    Performing Cross-Validation...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"    CV R2 Scores: {cv_scores.round(4)}")
print(f"    CV Mean R2:   {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Feature Importance
print("\n    [TOP 15 IMPORTANT FEATURES]")
print(f"    ----------------------------------------")
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(15).iterrows():
    bar = "#" * int(row['importance'] * 100)
    print(f"    {row['feature']:<30} {row['importance']:.4f} {bar}")

# ============================================
# 7. SAVE MODEL
# ============================================
print("\n" + "=" * 70)
print("SAVING MODEL...")
print("=" * 70)

# Save model and metadata
model_data = {
    'model': model,
    'columns': X_train.columns.tolist(),
    'locations': df_featured['location'].unique().tolist(),
    'base_features': base_features,
    'derived_features': available_derived,
    'feature_importance': feature_importance.to_dict(),
    'metrics': {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'mae': test_mae,
        'rmse': test_rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
}

joblib.dump(model_data, 'chennai_model_v2.pkl')
print(f"\nModel saved as 'chennai_model_v2.pkl'")

# Also save the old format for backward compatibility
old_model_data = {
    'model': model,
    'columns': X_train.columns.tolist(),
    'locations': df_featured['location'].unique().tolist()
}
joblib.dump(old_model_data, 'chennai_real_model.pkl')
print(f"Backward compatible model saved as 'chennai_real_model.pkl'")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\nFinal Model Performance:")
print(f"  - R2 Score: {test_r2:.4f} ({test_r2*100:.1f}% variance explained)")
print(f"  - Average Error: Rs.{test_mae:.2f} Lakhs")
print(f"  - Features Used: {len(X_train.columns)}")