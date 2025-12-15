"""
Test Features for Chennai Housing Price Prediction Model (V2)
==============================================================
Run this file to test model predictions with different inputs.
Update the test cases below to experiment with different values.
"""

import joblib
import pandas as pd
import numpy as np
import sys

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Import feature engineering functions
from feature_engineering import (
    get_location_distances,
    get_locality_tier,
    get_all_feature_categories
)

# ============================================
# LOAD MODEL
# ============================================
print("=" * 70)
print("CHENNAI HOUSING PRICE PREDICTION - MODEL TEST")
print("=" * 70)

print("\nLoading model...")
data = joblib.load('chennai_real_model.pkl')
model = data['model']
model_columns = data['columns']
locations = data['locations']

print(f"[OK] Model loaded successfully!")
print(f"[INFO] Total features: {len(model_columns)}")
print(f"[INFO] Total locations: {len(locations)}")

# Check if new model with extra info
if 'metrics' in data:
    metrics = data['metrics']
    print(f"\n[MODEL METRICS]")
    print(f"  R2 Score (Test): {metrics['test_r2']:.4f}")
    print(f"  MAE: Rs.{metrics['mae']:.2f} Lakhs")
    print(f"  CV Mean: {metrics['cv_mean']:.4f}")

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_price(location, area, bhk, bathroom, age, status="Ready to move", is_premium_builder=False):
    """
    Make a prediction for given inputs using all derived features.
    """
    # Get location distances
    loc_distances = get_location_distances(location)
    loc_tier = get_locality_tier(location)
    
    # Premium locations list
    premium_locations = [
        'Anna Nagar', 'Adyar', 'T Nagar', 'Velachery', 'Nungambakkam',
        'Besant Nagar', 'Alwarpet', 'Mylapore', 'Thiruvanmiyur', 'Egmore',
        'Royapettah', 'Kotturpuram', 'Guindy', 'Gopalapuram'
    ]
    
    suburban_locations = [
        'Sholinganallur', 'Siruseri', 'Kelambakkam', 'Navallur',
        'Padur', 'Thalambur', 'Perumbakkam', 'Medavakkam'
    ]
    
    it_hub_locations = [
        'Sholinganallur', 'Siruseri', 'Navallur', 'Perungudi',
        'Thoraipakkam OMR', 'Karapakkam', 'Guindy', 'Taramani'
    ]
    
    # Calculate derived features
    total_rooms = bhk + bathroom + 1
    avg_distance = (loc_distances[0] + loc_distances[1] + loc_distances[2] + loc_distances[4]) / 4
    
    # Prepare input data with all features
    input_data = {
        # Base features
        'area': area,
        'bhk': bhk,
        'bathroom': bathroom,
        'age': age,
        
        # Location distances
        'distance_to_beach': loc_distances[0],
        'distance_to_airport': loc_distances[1],
        'distance_to_railway': loc_distances[2],
        'distance_to_it_corridor': loc_distances[3],
        'distance_to_city_center': loc_distances[4],
        
        # Location scores
        'connectivity_score': 1 / (avg_distance + 1),
        'locality_tier': loc_tier,
        'near_it_corridor': 1 if loc_distances[3] <= 10 else 0,
        'near_beach': 1 if loc_distances[0] <= 5 else 0,
        'near_airport': 1 if loc_distances[1] <= 10 else 0,
        'is_central': 1 if loc_distances[4] <= 8 else 0,
        'is_premium_location': 1 if location in premium_locations else 0,
        'is_suburban': 1 if location in suburban_locations else 0,
        'is_it_hub': 1 if location in it_hub_locations else 0,
        
        # Room-based
        'bathroom_bhk_ratio': bathroom / max(bhk, 1),
        'total_rooms': total_rooms,
        'area_per_bhk': area / max(bhk, 1),
        'area_per_room': area / total_rooms,
        'is_well_equipped': 1 if bathroom >= bhk else 0,
        
        # Age-based
        'is_new_construction': 1 if age == 0 else 0,
        'is_established': 1 if age >= 5 else 0,
        'is_old': 1 if age >= 10 else 0,
        
        # Status-based
        'is_ready': 1 if status == "Ready to move" else 0,
        'is_under_construction': 1 if status == "Under Construction" else 0,
        
        # Builder-based
        'is_premium_builder': 1 if is_premium_builder else 0,
        
        # Property characteristics
        'is_spacious': 1 if area > 1100 else 0,
        'log_area': np.log1p(area),
        'bedroom_density': bhk / (area / 1000),
        'is_spacious_per_bhk': 1 if (area / max(bhk, 1)) > 500 else 0,
        'is_compact': 1 if (area < 800 and bhk <= 2) else 0,
        'is_family_home': 1 if (bhk >= 3 and area >= 1200 and bathroom >= 2) else 0,
        'is_studio_single': 1 if (bhk == 1 and area < 700) else 0,
        
        # Geospatial
        'location_cluster': 5 - loc_tier
    }
    
    # Handle one-hot encoded locations
    for col in model_columns:
        if col.startswith('location_'):
            if col == f'location_{location}':
                input_data[col] = 1
            else:
                input_data[col] = 0
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure all columns exist
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[model_columns]
    
    # Predict
    prediction = model.predict(input_df)
    return prediction[0], loc_tier

# ============================================
# TEST CASES
# ============================================
print("\n" + "=" * 70)
print("TEST CASES")
print("=" * 70)

test_cases = [
    # Format: (location, area_sqft, bhk, bathroom, age)
    ("Anna Nagar", 1500, 3, 2, 0),
    ("Adyar", 1200, 2, 2, 5),
    ("Velachery", 1000, 2, 1, 3),
    ("T Nagar", 2000, 3, 3, 0),
    ("Sholinganallur", 1800, 4, 3, 2),
    ("Perumbakkam", 1200, 3, 2, 0),
    ("Chromepet", 900, 2, 1, 5),
    ("Siruseri", 1500, 3, 2, 0),
]

print("\n[Prediction Results]")
print("-" * 80)
print(f"{'Location':<18} {'Area':<7} {'BHK':<4} {'Bath':<5} {'Age':<4} {'Tier':<5} {'Price (Lakhs)':<15}")
print("-" * 80)

for location, area, bhk, bathroom, age in test_cases:
    try:
        price, tier = predict_price(location, area, bhk, bathroom, age)
        tier_stars = '*' * (4 - tier)
        print(f"{location:<18} {area:<7} {bhk:<4} {bathroom:<5} {age:<4} {tier:<5} Rs.{price:,.2f}")
    except Exception as e:
        print(f"{location:<18} ERROR: {e}")

# ============================================
# COMPARISON: PREMIUM vs BUDGET
# ============================================
print("\n" + "=" * 70)
print("COMPARISON: PREMIUM vs BUDGET LOCATIONS")
print("=" * 70)

# Same property in different locations
same_property = (1200, 3, 2, 0)  # area, bhk, bathroom, age

print(f"\nSame Property: {same_property[0]} sqft, {same_property[1]} BHK, {same_property[2]} Bath, {same_property[3]} yrs old")
print("-" * 60)

comparison_locations = ['T Nagar', 'Anna Nagar', 'Velachery', 'Sholinganallur', 'Perumbakkam', 'Thirumazhisai']

for loc in comparison_locations:
    try:
        price, tier = predict_price(loc, *same_property)
        diff_from_base = ""
        print(f"  {loc:<18} Tier {tier}: Rs.{price:,.2f} Lakhs")
    except Exception as e:
        print(f"  {loc:<18} ERROR: {e}")

# ============================================
# FEATURE IMPACT TEST
# ============================================
print("\n" + "=" * 70)
print("FEATURE IMPACT TEST")
print("=" * 70)

base_location = "Velachery"
base_area = 1200
base_bhk = 2
base_bathroom = 2
base_age = 0

base_price, _ = predict_price(base_location, base_area, base_bhk, base_bathroom, base_age)
print(f"\nBase Property: {base_location}, {base_area} sqft, {base_bhk} BHK")
print(f"Base Price: Rs.{base_price:,.2f} Lakhs\n")

print("Impact of changing each feature:")
print("-" * 60)

# Area impact
price_bigger, _ = predict_price(base_location, base_area + 300, base_bhk, base_bathroom, base_age)
print(f"  +300 sqft area:     Rs.{price_bigger:,.2f} Lakhs (Diff: Rs.{price_bigger - base_price:+,.2f})")

# BHK impact
price_more_bhk, _ = predict_price(base_location, base_area, base_bhk + 1, base_bathroom, base_age)
print(f"  +1 BHK:             Rs.{price_more_bhk:,.2f} Lakhs (Diff: Rs.{price_more_bhk - base_price:+,.2f})")

# Bathroom impact
price_more_bath, _ = predict_price(base_location, base_area, base_bhk, base_bathroom + 1, base_age)
print(f"  +1 Bathroom:        Rs.{price_more_bath:,.2f} Lakhs (Diff: Rs.{price_more_bath - base_price:+,.2f})")

# Age impact
price_older, _ = predict_price(base_location, base_area, base_bhk, base_bathroom, 5)
print(f"  5 years old:        Rs.{price_older:,.2f} Lakhs (Diff: Rs.{price_older - base_price:+,.2f})")

# Premium location impact
price_premium, _ = predict_price("T Nagar", base_area, base_bhk, base_bathroom, base_age)
print(f"  Premium location:   Rs.{price_premium:,.2f} Lakhs (Diff: Rs.{price_premium - base_price:+,.2f})")

# ============================================
# CUSTOM TEST
# ============================================
print("\n" + "=" * 70)
print("CUSTOM TEST")
print("=" * 70)

# >>> UPDATE THESE VALUES TO TEST YOUR OWN INPUT <<<
custom_location = "Anna Nagar"
custom_area = 1500
custom_bhk = 3
custom_bathroom = 2
custom_age = 0
custom_status = "Ready to move"
custom_premium_builder = True

print(f"\nCustom Property:")
print(f"  Location: {custom_location}")
print(f"  Area: {custom_area} sq.ft")
print(f"  BHK: {custom_bhk}")
print(f"  Bathrooms: {custom_bathroom}")
print(f"  Age: {custom_age} years")
print(f"  Status: {custom_status}")
print(f"  Premium Builder: {custom_premium_builder}")

try:
    custom_price, tier = predict_price(
        custom_location, custom_area, custom_bhk, 
        custom_bathroom, custom_age, custom_status, custom_premium_builder
    )
    print(f"\n  >>> Estimated Price: Rs.{custom_price:,.2f} Lakhs <<<")
    print(f"  >>> Price/Sq.Ft: Rs.{(custom_price * 100000 / custom_area):,.0f} <<<")
    print(f"  >>> Locality Tier: {tier} <<<")
except Exception as e:
    print(f"\n  [ERROR] {e}")

print("\n" + "=" * 70)
print("[OK] Test completed!")
print("=" * 70)
