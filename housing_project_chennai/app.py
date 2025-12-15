"""
Chennai Housing Price Prediction - Streamlit App
=================================================
AI-Powered Real Estate Valuation with Advanced Features
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Import feature engineering functions
from feature_engineering import (
    get_location_distances,
    get_locality_tier,
    LOCATION_DISTANCES
)

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Chennai Housing AI",
    page_icon="🏠",
    layout="wide"
)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    data = joblib.load('chennai_real_model.pkl')
    return data

data = load_model()
model = data['model']
model_columns = data['columns']
location_list = sorted(data['locations'])

# ============================================
# HEADER
# ============================================
st.title("🏠 Chennai Real Estate Predictor")
st.markdown("### AI-Powered Valuation with 52+ Advanced Features")
st.markdown("---")

# ============================================
# SIDEBAR - Additional Info
# ============================================
with st.sidebar:
    st.header("📊 Model Info")
    st.metric("R² Score", "96.2%")
    st.metric("Avg Error", "±₹11.43 Lakhs")
    st.metric("Features Used", "215")
    
    st.markdown("---")
    st.header("📍 Locality Tiers")
    st.markdown("""
    - **Tier 1**: Premium (Adyar, T Nagar, Anna Nagar)
    - **Tier 2**: Good (Velachery, Sholinganallur)
    - **Tier 3**: Developing (Outskirts)
    """)

# ============================================
# INPUT FORM
# ============================================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 Location")
    location = st.selectbox("Select Location", location_list, index=location_list.index("Velachery") if "Velachery" in location_list else 0)
    
    # Show location insights
    distances = get_location_distances(location)
    tier = get_locality_tier(location)
    
    st.markdown(f"""
    <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;margin-top:10px;">
        <small>
        <b>Location Tier:</b> {'⭐' * (4-tier)} Tier {tier}<br>
        <b>Beach:</b> {distances[0]} km | <b>Airport:</b> {distances[1]} km<br>
        <b>IT Corridor:</b> {distances[3]} km | <b>City Center:</b> {distances[4]} km
        </small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("📏 Property Details")
    area = st.number_input("Area (Sq. Ft.)", min_value=300, max_value=10000, value=1200, step=50)
    bhk = st.slider("BHK (Bedrooms)", 1, 6, 2)
    bathroom = st.slider("Bathrooms", 1, 6, 2)

with col3:
    st.subheader("🏗️ Property Status")
    age = st.number_input("Property Age (Years)", min_value=0, max_value=50, value=0)
    status = st.radio("Construction Status", ["Ready to move", "Under Construction"])
    is_premium_builder = st.checkbox("Premium Builder", value=False)

st.markdown("---")

# ============================================
# PREDICTION
# ============================================
if st.button("💰 Estimate Property Value", type="primary", use_container_width=True):
    
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
        'is_spacious': 1 if area > 1100 else 0,  # Approximate median
        'log_area': np.log1p(area),
        'bedroom_density': bhk / (area / 1000),
        'is_spacious_per_bhk': 1 if (area / max(bhk, 1)) > 500 else 0,
        'is_compact': 1 if (area < 800 and bhk <= 2) else 0,
        'is_family_home': 1 if (bhk >= 3 and area >= 1200 and bathroom >= 2) else 0,
        'is_studio_single': 1 if (bhk == 1 and area < 700) else 0,
        
        # Geospatial (use default cluster based on tier)
        'location_cluster': 5 - loc_tier  # Approximate
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
    
    # Ensure all columns exist and are in correct order
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[model_columns]
    
    # Predict
    prediction = model.predict(input_df)
    price = prediction[0]
    
    # Display results
    st.markdown("---")
    
    col_result1, col_result2 = st.columns([2, 1])
    
    with col_result1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 30px; border-radius: 15px; text-align: center; color: white;">
            <h1 style="margin: 0; font-size: 3em;">₹ {price:,.2f} Lakhs</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em;">Estimated Property Value</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Price range
        lower_bound = price * 0.9
        upper_bound = price * 1.1
        st.info(f"📊 Estimated Range: ₹{lower_bound:,.2f} - ₹{upper_bound:,.2f} Lakhs")
    
    with col_result2:
        st.markdown("### 📈 Property Insights")
        
        # Property classification
        if price > 150:
            category = "🏆 Luxury"
        elif price > 80:
            category = "⭐ Premium"
        elif price > 50:
            category = "✅ Mid-Range"
        else:
            category = "💰 Budget-Friendly"
        
        st.metric("Category", category)
        st.metric("Price/Sq.Ft", f"₹{(price * 100000 / area):,.0f}")
        st.metric("Locality Tier", f"Tier {loc_tier}")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>Built with Python, Scikit-Learn & Streamlit | 52+ Engineered Features | ~96% Accuracy</p>
    <p>Data Source: Chennai Real Estate Market Data</p>
</div>
""", unsafe_allow_html=True)