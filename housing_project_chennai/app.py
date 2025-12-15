import streamlit as st
import joblib
import pandas as pd

# Load the trained model and metadata
data = joblib.load('chennai_real_model.pkl')
model = data['model']
model_columns = data['columns']
location_list = sorted(data['locations'])  # Load locations from the file

st.set_page_config(page_title="Chennai Housing AI", page_icon="🏠")

st.title("🏠 Chennai Real Estate Predictor")
st.write("### AI-Powered Valuation for Chennai Properties")
st.markdown("---")

# -- Layout: 2 Columns for better UI --
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("📍 Select Location", location_list)
    area = st.number_input("📏 Area (Sq. Ft.)", min_value=300, max_value=10000, value=1000)
    age = st.number_input("🏗️ Property Age (Years)", min_value=0, max_value=50, value=1)

with col2:
    bhk = st.slider("🛏️ BHK (Bedrooms)", 1, 10, 2)
    bathroom = st.slider("🚿 Bathrooms", 1, 10, 2)

# -- Prediction Logic --
if st.button("Estimate Price 💰", type="primary"):
    # 1. Prepare Data Dictionary with correct column names
    input_data = {
        'area': area,
        'bhk': bhk,
        'bathroom': bathroom,
        'age': age
    }
    
    # 2. Handle One-Hot Encoded Location
    # Loop through all trained columns. If the column matches 'location_Anna Nagar', set it to 1.
    for col in model_columns:
        if col.startswith('location_'):
            # Check if this column matches the selected location
            if col == f'location_{location}':
                input_data[col] = 1
            else:
                input_data[col] = 0
    
    # 3. Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 4. Align Columns (Crucial Step!)
    # Ensure columns are in the EXACT order the model learned
    input_df = input_df[model_columns]

    # 5. Predict
    prediction = model.predict(input_df)
    price = prediction[0]
    
    st.markdown(f"""
    <div style="background-color:#d4edda;padding:20px;border-radius:10px;text-align:center;">
        <h2 style="color:#155724;margin:0;">Estimated Value: ₹ {price:,.2f} Lakhs</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("---")
st.caption("Built with Python & Scikit-Learn | Data Source: Kaggle Real Estate Data")