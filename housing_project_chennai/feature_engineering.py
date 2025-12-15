"""
Feature Engineering for Chennai Housing Price Prediction
=========================================================
This file contains all derived features.
Import this in model_train.py and test_features.py

Available columns in data: price, area, status, bhk, bathroom, age, location, builder
"""

import pandas as pd
import numpy as np

# ============================================
# LOCATION INTELLIGENCE DATABASE
# ============================================
# Approximate distances (in km) from key landmarks for Chennai locations
# Reference points:
#   - Beach: Marina Beach (13.0500, 80.2824)
#   - Airport: Chennai International Airport (12.9941, 80.1709)
#   - Central Railway: Chennai Central (13.0827, 80.2707)
#   - IT Corridor: Siruseri IT Park (12.8253, 80.2194)
#   - City Center: Anna Salai / Mount Road (13.0604, 80.2496)

LOCATION_DISTANCES = {
    # Format: 'Location': (beach_km, airport_km, railway_km, it_corridor_km, city_center_km)
    
    # Premium Central Locations
    'Adyar': (3, 12, 8, 18, 6),
    'Anna Nagar': (8, 18, 6, 28, 7),
    'T Nagar': (5, 10, 5, 20, 2),
    'Nungambakkam': (5, 12, 4, 22, 3),
    'Alwarpet': (4, 11, 6, 19, 3),
    'Mylapore': (2, 13, 5, 20, 4),
    'Royapettah': (3, 12, 4, 21, 2),
    'Egmore': (4, 14, 2, 24, 3),
    'Gopalapuram': (6, 14, 5, 24, 5),
    'Kotturpuram': (4, 11, 7, 17, 4),
    'Besant Nagar': (1, 14, 9, 18, 7),
    'Thiruvanmiyur': (2, 14, 10, 15, 8),
    
    # IT Corridor / OMR Locations
    'Sholinganallur': (5, 12, 15, 8, 14),
    'Siruseri': (8, 10, 18, 2, 18),
    'Navallur': (6, 11, 16, 5, 15),
    'Perungudi': (4, 11, 12, 10, 10),
    'Thoraipakkam OMR': (4, 12, 14, 8, 12),
    'Karapakkam': (5, 11, 14, 7, 13),
    'Padur': (7, 10, 17, 4, 16),
    'Kelambakkam': (10, 8, 20, 3, 19),
    'Thalambur': (8, 9, 18, 4, 17),
    
    # South Chennai
    'Velachery': (6, 8, 10, 12, 9),
    'Medavakkam': (8, 7, 12, 10, 11),
    'Perumbakkam': (9, 6, 14, 8, 13),
    'Madipakkam': (7, 8, 10, 13, 10),
    'Chromepet': (10, 5, 12, 15, 12),
    'Pallavaram': (12, 3, 14, 17, 13),
    'Tambaram': (14, 4, 16, 18, 15),
    'East Tambaram': (13, 5, 15, 17, 14),
    'Selaiyur': (12, 5, 14, 14, 13),
    'Gowrivakkam': (10, 6, 13, 12, 12),
    'Sembakkam': (11, 5, 13, 13, 12),
    'Pammal': (13, 4, 15, 18, 14),
    'Anakaputhur': (14, 4, 16, 19, 15),
    'Perungalathur': (16, 5, 18, 20, 17),
    
    # West Chennai
    'Porur': (12, 10, 12, 22, 10),
    'Mugalivakkam': (10, 9, 11, 20, 9),
    'Manapakkam': (11, 9, 11, 21, 9),
    'Virugambakkam': (9, 12, 8, 24, 7),
    'Vadapalani': (8, 11, 7, 23, 6),
    'Valasaravakkam': (10, 10, 9, 22, 8),
    'Iyappanthangal': (12, 12, 11, 25, 10),
    'Gerugambakkam': (14, 11, 14, 25, 13),
    'Kundrathur': (18, 10, 18, 28, 17),
    'Poonamallee': (20, 15, 18, 32, 18),
    'Ayanambakkam': (15, 16, 12, 30, 13),
    'Vanagaram': (14, 15, 11, 29, 12),
    'Kolapakkam': (13, 9, 13, 24, 12),
    
    # North Chennai
    'Ambattur': (15, 18, 10, 32, 12),
    'Mogappair': (12, 17, 9, 30, 10),
    'Maduravoyal': (14, 15, 10, 28, 11),
    'Thirumullaivoyal': (18, 20, 13, 35, 15),
    'Moolakadai': (10, 18, 7, 30, 9),
    'Madhavaram': (12, 22, 8, 35, 11),
    'Tiruvottiyur': (8, 25, 10, 38, 12),
    'Kolathur': (9, 19, 6, 32, 8),
    'Villivakkam': (8, 17, 5, 30, 7),
    
    # ECR / Beach Side
    'Kanathur Reddikuppam': (2, 16, 16, 12, 15),
    'Neelankarai': (1, 15, 12, 14, 11),
    'Palavakkam': (1, 14, 11, 16, 10),
    
    # Outer Areas
    'Ottiyambakkam': (10, 6, 15, 10, 14),
    'Thaiyur': (12, 6, 18, 6, 17),
    'Sithalapakkam': (9, 6, 14, 9, 13),
    'Madambakkam': (11, 5, 14, 12, 13),
    'Thirumazhisai': (22, 16, 20, 35, 20),
    'Thandalam': (24, 18, 22, 38, 22),
    'Guindy': (7, 8, 8, 18, 6),
    
    # Default for unknown locations
    'DEFAULT': (15, 15, 15, 20, 15)
}

# Locality Tier Classification
LOCALITY_TIERS = {
    # Tier 1: Premium/Posh areas
    'Tier 1': [
        'Adyar', 'Anna Nagar', 'T Nagar', 'Nungambakkam', 'Alwarpet', 
        'Mylapore', 'Besant Nagar', 'Gopalapuram', 'Kotturpuram',
        'Egmore', 'Royapettah', 'Thiruvanmiyur', 'Guindy'
    ],
    
    # Tier 2: Developing/Good connectivity
    'Tier 2': [
        'Velachery', 'Sholinganallur', 'Porur', 'Vadapalani', 'Mogappair',
        'Ambattur', 'Perungudi', 'Thoraipakkam OMR', 'Medavakkam',
        'Madipakkam', 'Chromepet', 'Virugambakkam', 'Siruseri', 'Navallur',
        'Karapakkam', 'Pallavaram', 'Kolathur', 'Villivakkam'
    ],
    
    # Tier 3: Outskirts/Developing
    'Tier 3': [
        'Perumbakkam', 'Tambaram', 'East Tambaram', 'Selaiyur', 'Gowrivakkam',
        'Sembakkam', 'Pammal', 'Anakaputhur', 'Perungalathur', 'Kundrathur',
        'Poonamallee', 'Gerugambakkam', 'Kelambakkam', 'Thalambur', 'Padur',
        'Madhavaram', 'Thirumullaivoyal', 'Thaiyur', 'Sithalapakkam',
        'Madambakkam', 'Ottiyambakkam', 'Kolapakkam', 'Mugalivakkam',
        'Manapakkam', 'Vanagaram', 'Ayanambakkam', 'Iyappanthangal',
        'Thirumazhisai', 'Thandalam', 'Tiruvottiyur', 'Moolakadai'
    ]
}


def get_location_distances(location):
    """
    Get distances from key landmarks for a location.
    Returns: (beach, airport, railway, it_corridor, city_center) in km
    """
    # Clean location name
    location = str(location).strip().title()
    
    # Check for exact match
    if location in LOCATION_DISTANCES:
        return LOCATION_DISTANCES[location]
    
    # Check for partial match
    for loc_name, distances in LOCATION_DISTANCES.items():
        if loc_name in location or location in loc_name:
            return distances
    
    # Return default
    return LOCATION_DISTANCES['DEFAULT']


def get_locality_tier(location):
    """
    Get the locality tier (1, 2, or 3) for a location.
    Tier 1 = Premium, Tier 2 = Good, Tier 3 = Developing
    """
    location = str(location).strip().title()
    
    for tier, locations in LOCALITY_TIERS.items():
        if location in locations:
            return int(tier.split()[-1])
        # Check partial match
        for loc in locations:
            if loc in location or location in loc:
                return int(tier.split()[-1])
    
    return 3  # Default to Tier 3


def create_derived_features(df):
    """
    Create all derived features from raw data.
    
    Parameters:
        df: DataFrame with columns [price, area, status, bhk, bathroom, age, location, builder]
    
    Returns:
        DataFrame with original + derived features
    """
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # ============================================
    # 1. AREA-BASED FEATURES
    # ============================================
    
    # Price per square foot (in Lakhs)
    df['price_per_sqft'] = df['price'] / df['area']
    
    # Area categories
    df['area_category'] = pd.cut(
        df['area'],
        bins=[0, 600, 1000, 1500, 2000, float('inf')],
        labels=['Compact', 'Small', 'Medium', 'Large', 'Luxury']
    )
    
    # Is the property spacious? (above median area)
    median_area = df['area'].median()
    df['is_spacious'] = (df['area'] > median_area).astype(int)
    
    # Log of area (reduces skewness)
    df['log_area'] = np.log1p(df['area'])
    
    # ============================================
    # 2. AGE-BASED FEATURES
    # ============================================
    
    # Is new construction? (0 years or under construction)
    df['is_new_construction'] = (df['age'] == 0).astype(int)
    
    # Is established property? (5+ years)
    df['is_established'] = (df['age'] >= 5).astype(int)
    
    # Is old property? (10+ years)
    df['is_old'] = (df['age'] >= 10).astype(int)
    
    # Age categories
    df['age_category'] = pd.cut(
        df['age'],
        bins=[-1, 0, 2, 5, 10, float('inf')],
        labels=['New', 'Recent', 'Moderate', 'Established', 'Old']
    )
    
    # ============================================
    # 3. ROOM-BASED FEATURES
    # ============================================
    
    # Bathroom to bedroom ratio
    df['bathroom_bhk_ratio'] = df['bathroom'] / df['bhk'].replace(0, 1)
    
    # Total rooms (approx: bhk + bathrooms + 1 for living)
    df['total_rooms'] = df['bhk'] + df['bathroom'] + 1
    
    # Area per bedroom
    df['area_per_bhk'] = df['area'] / df['bhk'].replace(0, 1)
    
    # Area per room
    df['area_per_room'] = df['area'] / df['total_rooms']
    
    # Is it well-equipped? (bathroom >= bhk)
    df['is_well_equipped'] = (df['bathroom'] >= df['bhk']).astype(int)
    
    # BHK category
    df['bhk_category'] = pd.cut(
        df['bhk'],
        bins=[0, 1, 2, 3, float('inf')],
        labels=['Studio', '1-2BHK', '3BHK', 'Premium']
    )
    
    # ============================================
    # 4. STATUS-BASED FEATURES
    # ============================================
    
    # Is ready to move?
    df['is_ready'] = (df['status'] == 'Ready to move').astype(int)
    
    # Is under construction?
    df['is_under_construction'] = (df['status'] == 'Under Construction').astype(int)
    
    # ============================================
    # 5. LOCATION INTELLIGENCE FEATURES
    # ============================================
    
    # Get distances for each location
    distances = df['location'].apply(get_location_distances)
    
    df['distance_to_beach'] = distances.apply(lambda x: x[0])
    df['distance_to_airport'] = distances.apply(lambda x: x[1])
    df['distance_to_railway'] = distances.apply(lambda x: x[2])
    df['distance_to_it_corridor'] = distances.apply(lambda x: x[3])
    df['distance_to_city_center'] = distances.apply(lambda x: x[4])
    
    # Average distance (connectivity score - lower is better)
    df['avg_distance'] = (
        df['distance_to_beach'] + 
        df['distance_to_airport'] + 
        df['distance_to_railway'] + 
        df['distance_to_city_center']
    ) / 4
    
    # Connectivity score (inverse of average distance)
    df['connectivity_score'] = 1 / (df['avg_distance'] + 1)
    
    # Near IT corridor (for tech employees)
    df['near_it_corridor'] = (df['distance_to_it_corridor'] <= 10).astype(int)
    
    # Near beach (premium feature)
    df['near_beach'] = (df['distance_to_beach'] <= 5).astype(int)
    
    # Near airport (convenience)
    df['near_airport'] = (df['distance_to_airport'] <= 10).astype(int)
    
    # Central location (close to city center)
    df['is_central'] = (df['distance_to_city_center'] <= 8).astype(int)
    
    # Locality tier
    df['locality_tier'] = df['location'].apply(get_locality_tier)
    
    # Is prime location (Tier 1)
    df['is_prime_location'] = (df['locality_tier'] == 1).astype(int)
    
    # Premium locations list
    premium_locations = [
        'Anna Nagar', 'Adyar', 'T Nagar', 'Velachery', 'Nungambakkam',
        'Besant Nagar', 'Alwarpet', 'Mylapore', 'Thiruvanmiyur', 'Egmore',
        'Royapettah', 'Kotturpuram', 'Guindy', 'Gopalapuram'
    ]
    df['is_premium_location'] = df['location'].isin(premium_locations).astype(int)
    
    # Suburban locations
    suburban_locations = [
        'Sholinganallur', 'Siruseri', 'Kelambakkam', 'Navallur',
        'Padur', 'Thalambur', 'Perumbakkam', 'Medavakkam'
    ]
    df['is_suburban'] = df['location'].isin(suburban_locations).astype(int)
    
    # IT Hub location (good for rentals)
    it_hub_locations = [
        'Sholinganallur', 'Siruseri', 'Navallur', 'Perungudi',
        'Thoraipakkam OMR', 'Karapakkam', 'Guindy', 'Taramani'
    ]
    df['is_it_hub'] = df['location'].isin(it_hub_locations).astype(int)
    
    # ============================================
    # 6. BUILDER-BASED FEATURES
    # ============================================
    
    # Premium builders (can be updated based on domain knowledge)
    premium_builders = [
        'Casagrand Builder Private Limited', 'Prestige Estates Projects Ltd',
        'Puravankara Limited', 'Appaswamy Real Estate', 'India Builders Limited',
        'Olympia Group', 'Radiance Realty Developers India Ltd'
    ]
    df['is_premium_builder'] = df['builder'].isin(premium_builders).astype(int)
    
    # ============================================
    # 7. INTERACTION FEATURES
    # ============================================
    
    # Premium property score (combination)
    df['premium_score'] = (
        df['is_premium_location'] + 
        df['is_premium_builder'] + 
        df['is_spacious'] + 
        df['is_well_equipped'] +
        df['is_prime_location']
    )
    
    # Value score (area * rooms / price) - higher means better value
    df['value_score'] = (df['area'] * df['total_rooms']) / (df['price'] + 0.01)
    
    # Luxury indicator
    df['is_luxury'] = (
        (df['bhk'] >= 3) & 
        (df['area'] >= 1500) & 
        (df['bathroom'] >= 2)
    ).astype(int)
    
    # Location advantage score
    df['location_advantage'] = (
        df['near_beach'] * 2 +
        df['near_it_corridor'] * 1.5 +
        df['is_central'] * 1.5 +
        df['near_airport'] * 0.5 +
        (4 - df['locality_tier'])  # Higher score for better tier
    )
    
    # Investment score (good for investment)
    df['investment_score'] = (
        df['is_it_hub'] * 2 +
        df['is_suburban'] * 1.5 +
        df['is_new_construction'] * 1 +
        df['connectivity_score'] * 5
    )
    
    # ============================================
    # 8. ADVANCED GEOSPATIAL FEATURES
    # ============================================
    
    # Price cluster based on location characteristics
    # Using location distances as proxy for lat/long
    try:
        from sklearn.cluster import KMeans
        
        # Create feature matrix for clustering
        cluster_features = df[[
            'distance_to_beach', 'distance_to_airport', 
            'distance_to_it_corridor', 'distance_to_city_center', 'price'
        ]].copy()
        
        # Handle any missing values
        cluster_features = cluster_features.fillna(cluster_features.median())
        
        # Fit KMeans clustering (5 price zones)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df['price_cluster'] = kmeans.fit_predict(cluster_features)
        
        # Location cluster (without price, just geography)
        location_features = df[[
            'distance_to_beach', 'distance_to_airport', 
            'distance_to_it_corridor', 'distance_to_city_center'
        ]].copy()
        location_features = location_features.fillna(location_features.median())
        
        kmeans_loc = KMeans(n_clusters=6, random_state=42, n_init=10)
        df['location_cluster'] = kmeans_loc.fit_predict(location_features)
        
        # Price zone categories based on cluster
        cluster_price_means = df.groupby('price_cluster')['price'].mean().sort_values()
        price_zone_map = {cluster: zone for zone, cluster in enumerate(cluster_price_means.index)}
        df['price_zone'] = df['price_cluster'].map(price_zone_map)
        
    except ImportError:
        # If sklearn not available, use simple binning
        df['price_cluster'] = pd.qcut(df['price'], q=5, labels=[0, 1, 2, 3, 4]).astype(int)
        df['location_cluster'] = pd.qcut(df['distance_to_city_center'], q=6, labels=[0, 1, 2, 3, 4, 5]).astype(int)
        df['price_zone'] = df['price_cluster']
    
    # ============================================
    # 9. PROPERTY CHARACTERISTICS
    # ============================================
    
    # Luxury score (comprehensive)
    df['luxury_score'] = (
        (df['bathroom'] >= 3).astype(int) +          # Multiple bathrooms
        (df['area'] >= 2000).astype(int) +           # Large area
        (df['bhk'] >= 4).astype(int) +               # 4+ bedrooms
        (df['is_premium_location']).astype(int) +    # Premium location
        (df['is_premium_builder']).astype(int)       # Premium builder
    )
    
    # Layout efficiency - bedroom to sqft ratio
    df['bedroom_to_sqft'] = df['bhk'] / df['area']
    
    # Is spacious per bedroom (more than 500 sqft per bedroom)
    df['is_spacious_per_bhk'] = (df['area'] / df['bhk'].replace(0, 1) > 500).astype(int)
    
    # Bedroom density (how compact are the bedrooms)
    df['bedroom_density'] = df['bhk'] / (df['area'] / 1000)  # bedrooms per 1000 sqft
    
    # Bathroom luxury (more bathrooms than bedrooms is luxury)
    df['bathroom_luxury'] = (df['bathroom'] > df['bhk']).astype(int)
    
    # Property grade based on multiple factors
    df['property_grade'] = (
        df['luxury_score'] + 
        df['is_spacious_per_bhk'] + 
        df['is_well_equipped'] +
        (5 - df['locality_tier']) +  # Higher tier = higher grade
        df['is_ready']  # Ready to move is preferred
    )
    
    # Compact apartment indicator
    df['is_compact'] = (
        (df['area'] < 800) & 
        (df['bhk'] <= 2)
    ).astype(int)
    
    # Family home indicator (3+ BHK, spacious)
    df['is_family_home'] = (
        (df['bhk'] >= 3) & 
        (df['area'] >= 1200) & 
        (df['bathroom'] >= 2)
    ).astype(int)
    
    # Studio/Single indicator
    df['is_studio_single'] = (
        (df['bhk'] == 1) & 
        (df['area'] < 700)
    ).astype(int)
    
    # Value for money score
    df['value_for_money'] = (
        df['area'] * df['total_rooms'] * (5 - df['locality_tier'])
    ) / (df['price'] + 0.01)
    
    # Price category
    price_quartiles = df['price'].quantile([0.25, 0.5, 0.75])
    df['price_category'] = pd.cut(
        df['price'],
        bins=[0, price_quartiles[0.25], price_quartiles[0.5], price_quartiles[0.75], float('inf')],
        labels=['Budget', 'Moderate', 'Premium', 'Luxury']
    )
    
    return df


def get_feature_columns():
    """
    Returns the list of derived feature column names (numeric only).
    Use this to know which features are available for modeling.
    """
    return [
        # Area-based
        'price_per_sqft', 'is_spacious', 'log_area',
        
        # Age-based
        'is_new_construction', 'is_established', 'is_old',
        
        # Room-based
        'bathroom_bhk_ratio', 'total_rooms', 'area_per_bhk', 
        'area_per_room', 'is_well_equipped',
        
        # Status-based
        'is_ready', 'is_under_construction',
        
        # Location Intelligence - Distances
        'distance_to_beach', 'distance_to_airport', 'distance_to_railway',
        'distance_to_it_corridor', 'distance_to_city_center', 'avg_distance',
        
        # Location Intelligence - Scores
        'connectivity_score', 'near_it_corridor', 'near_beach',
        'near_airport', 'is_central', 'locality_tier', 'is_prime_location',
        
        # Location-based
        'is_premium_location', 'is_suburban', 'is_it_hub',
        
        # Builder-based
        'is_premium_builder',
        
        # Interaction features
        'premium_score', 'value_score', 'is_luxury',
        'location_advantage', 'investment_score',
        
        # Advanced Geospatial Features
        'price_cluster', 'location_cluster', 'price_zone',
        
        # Property Characteristics
        'luxury_score', 'bedroom_to_sqft', 'is_spacious_per_bhk',
        'bedroom_density', 'bathroom_luxury', 'property_grade',
        'is_compact', 'is_family_home', 'is_studio_single', 'value_for_money'
    ]


def get_categorical_features():
    """Returns list of categorical derived features"""
    return ['area_category', 'age_category', 'bhk_category']


def get_location_features():
    """Returns list of location intelligence features"""
    return [
        'distance_to_beach', 'distance_to_airport', 'distance_to_railway',
        'distance_to_it_corridor', 'distance_to_city_center', 'avg_distance',
        'connectivity_score', 'near_it_corridor', 'near_beach',
        'near_airport', 'is_central', 'locality_tier', 'is_prime_location',
        'is_premium_location', 'is_suburban', 'is_it_hub',
        'location_advantage', 'investment_score'
    ]


def get_geospatial_features():
    """Returns list of advanced geospatial features"""
    return [
        'price_cluster', 'location_cluster', 'price_zone'
    ]


def get_property_characteristics():
    """Returns list of property characteristic features"""
    return [
        'luxury_score', 'bedroom_to_sqft', 'is_spacious_per_bhk',
        'bedroom_density', 'bathroom_luxury', 'property_grade',
        'is_compact', 'is_family_home', 'is_studio_single', 'value_for_money'
    ]


def get_all_feature_categories():
    """Returns a dictionary of all feature categories"""
    return {
        'Area-based': ['price_per_sqft', 'is_spacious', 'log_area'],
        'Age-based': ['is_new_construction', 'is_established', 'is_old'],
        'Room-based': ['bathroom_bhk_ratio', 'total_rooms', 'area_per_bhk', 'area_per_room', 'is_well_equipped'],
        'Status-based': ['is_ready', 'is_under_construction'],
        'Location Distances': ['distance_to_beach', 'distance_to_airport', 'distance_to_railway', 
                               'distance_to_it_corridor', 'distance_to_city_center', 'avg_distance'],
        'Location Scores': ['connectivity_score', 'near_it_corridor', 'near_beach', 'near_airport', 
                            'is_central', 'locality_tier', 'is_prime_location'],
        'Location Types': ['is_premium_location', 'is_suburban', 'is_it_hub'],
        'Builder-based': ['is_premium_builder'],
        'Interaction': ['premium_score', 'value_score', 'is_luxury', 'location_advantage', 'investment_score'],
        'Geospatial': ['price_cluster', 'location_cluster', 'price_zone'],
        'Property Characteristics': ['luxury_score', 'bedroom_to_sqft', 'is_spacious_per_bhk', 
                                     'bedroom_density', 'bathroom_luxury', 'property_grade',
                                     'is_compact', 'is_family_home', 'is_studio_single', 'value_for_money']
    }


# ============================================
# TEST THE FEATURES
# ============================================
if __name__ == "__main__":
    print("Testing Feature Engineering...")
    print("=" * 70)
    
    # Load sample data
    df = pd.read_csv('data/clean_data.csv')
    
    # Fill missing values for testing
    df['bhk'] = df['bhk'].fillna(df['bhk'].median())
    df['bathroom'] = df['bathroom'].fillna(df['bathroom'].median())
    df['age'] = df['age'].fillna(df['age'].median())
    
    # Create features
    df_featured = create_derived_features(df)
    
    print(f"\nOriginal columns: {len(df.columns)}")
    print(f"After feature engineering: {len(df_featured.columns)}")
    print(f"New features added: {len(df_featured.columns) - len(df.columns)}")
    
    print("\n[FEATURE CATEGORIES]")
    print("-" * 70)
    categories = get_all_feature_categories()
    for cat_name, features in categories.items():
        print(f"\n  {cat_name} ({len(features)} features):")
        for feat in features[:5]:
            if feat in df_featured.columns:
                print(f"    - {feat}")
        if len(features) > 5:
            print(f"    ... and {len(features) - 5} more")
    
    print("\n[GEOSPATIAL CLUSTERING]")
    print("-" * 70)
    if 'price_cluster' in df_featured.columns:
        cluster_dist = df_featured['price_cluster'].value_counts().sort_index()
        for cluster, count in cluster_dist.items():
            avg_price = df_featured[df_featured['price_cluster'] == cluster]['price'].mean()
            print(f"  Cluster {cluster}: {count:4d} properties, Avg Price: Rs.{avg_price:.2f} Lakhs")
    
    print("\n[PROPERTY CHARACTERISTICS SAMPLE]")
    print("-" * 70)
    prop_cols = ['location', 'luxury_score', 'property_grade', 'is_family_home', 
                 'is_compact', 'value_for_money']
    available = [c for c in prop_cols if c in df_featured.columns]
    print(df_featured[available].head(10).to_string())
    
    print("\n[LUXURY SCORE DISTRIBUTION]")
    print("-" * 70)
    if 'luxury_score' in df_featured.columns:
        luxury_dist = df_featured['luxury_score'].value_counts().sort_index()
        for score, count in luxury_dist.items():
            pct = count / len(df_featured) * 100
            bar = "#" * int(pct / 2)
            print(f"  Score {int(score)}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print("\n[PRICE CATEGORY DISTRIBUTION]")
    print("-" * 70)
    if 'price_category' in df_featured.columns:
        cat_dist = df_featured['price_category'].value_counts()
        for cat in ['Budget', 'Moderate', 'Premium', 'Luxury']:
            if cat in cat_dist.index:
                count = cat_dist[cat]
                pct = count / len(df_featured) * 100
                bar = "#" * int(pct / 2)
                print(f"  {cat:10s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    print("\n[TOTAL FEATURES SUMMARY]")
    print("-" * 70)
    total_features = len(get_feature_columns())
    print(f"  Total numeric features: {total_features}")
    print(f"  Total categorical features: {len(get_categorical_features())}")
    print(f"  Grand total derived features: {total_features + len(get_categorical_features()) + 1}")  # +1 for price_category
    
    print("\n[OK] Feature engineering test completed!")

