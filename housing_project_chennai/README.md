# 🏠 Chennai Housing Price Prediction

An AI-powered real estate valuation system for Chennai properties using Machine Learning with **52+ engineered features** and **96% accuracy**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | 96.2% |
| **Mean Absolute Error** | ₹11.43 Lakhs |
| **Cross-Validation Score** | 78.6% |
| **Total Features** | 215 |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python model_train.py
```

### 3. Run the Web App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
housing_project_chennai/
│
├── data/
│   └── clean_data.csv          # Chennai housing dataset (2,620 records)
│
├── feature_engineering.py       # 52 derived features
├── model_train.py               # Model training script
├── test_features.py             # Testing & validation
├── app.py                       # Streamlit web application
│
├── chennai_model_v2.pkl         # Trained model (with metadata)
├── chennai_real_model.pkl       # Trained model (backward compatible)
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## ✨ Features

### 🔧 Feature Engineering (52 Features)

| Category | Features | Description |
|----------|----------|-------------|
| **Area-based** | 3 | price_per_sqft, is_spacious, log_area |
| **Age-based** | 3 | is_new_construction, is_established, is_old |
| **Room-based** | 5 | bathroom_bhk_ratio, total_rooms, area_per_bhk, etc. |
| **Status-based** | 2 | is_ready, is_under_construction |
| **Location Distances** | 6 | Distance to beach, airport, railway, IT corridor, city center |
| **Location Scores** | 7 | connectivity_score, locality_tier, is_central, etc. |
| **Location Types** | 3 | is_premium_location, is_suburban, is_it_hub |
| **Builder-based** | 1 | is_premium_builder |
| **Interaction** | 5 | premium_score, value_score, is_luxury, etc. |
| **Geospatial** | 3 | price_cluster, location_cluster, price_zone |
| **Property** | 10 | luxury_score, bedroom_density, is_family_home, etc. |

### 📍 Location Intelligence

The system includes a comprehensive database of **75+ Chennai locations** with pre-calculated distances to:

- 🏖️ Marina Beach
- ✈️ Chennai International Airport
- 🚂 Chennai Central Railway Station
- 💻 Siruseri IT Corridor
- 🏛️ Anna Salai (City Center)

### 🏢 Locality Tier Classification

| Tier | Description | Example Locations |
|------|-------------|-------------------|
| **Tier 1** | Premium | Anna Nagar, Adyar, T Nagar, Besant Nagar |
| **Tier 2** | Good Connectivity | Velachery, Sholinganallur, Porur |
| **Tier 3** | Developing | Perumbakkam, Tambaram, Kelambakkam |

---

## 💻 Usage

### Web Application

```bash
streamlit run app.py
```

The app provides:
- Interactive property input form
- Location insights (distances, tier)
- Price prediction with confidence range
- Property classification (Budget/Mid-Range/Premium/Luxury)

### Python API

```python
from feature_engineering import create_derived_features
import pandas as pd
import joblib

# Load model
model_data = joblib.load('chennai_model_v2.pkl')
model = model_data['model']

# Your prediction logic here...
```

### Testing

```bash
python test_features.py
```

---

## 📈 Sample Predictions

| Location | Area | BHK | Price Estimate |
|----------|------|-----|----------------|
| T Nagar | 2000 sqft | 3 | ₹337.67 Lakhs |
| Anna Nagar | 1500 sqft | 3 | ₹232.87 Lakhs |
| Velachery | 1000 sqft | 2 | ₹88.56 Lakhs |
| Sholinganallur | 1800 sqft | 4 | ₹119.14 Lakhs |
| Perumbakkam | 1200 sqft | 3 | ₹61.98 Lakhs |

---

## 🔑 Key Insights

From the model's feature importance:

1. **Area** (13%) - Most important predictor
2. **Log Area** (11%) - Non-linear area effect
3. **Area per Room** (8%) - Space efficiency
4. **Connectivity Score** (5%) - Location accessibility
5. **Premium Location** (4%) - Location tier impact

**Location Impact**: Same property can vary 3-4x in price based on location alone!

---

## 📊 Data Source

- **Dataset**: Chennai Housing Price Data
- **Records**: 2,620 properties
- **Features**: price, area, status, bhk, bathroom, age, location, builder
- **Source**: Kaggle Real Estate Data

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-Learn** - Machine Learning
- **Streamlit** - Web application
- **Joblib** - Model serialization

---

## 📝 License

This project is for educational purposes.

---

## 👨‍💻 Author

Created as part of ML Journey with Python.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
