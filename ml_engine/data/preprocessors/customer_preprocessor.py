import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer
from geopy.distance import geodesic
from sklearn.cluster import KMeans, DBSCAN
import re
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerPreprocessor:
    """
    Advanced customer data preprocessing for geosales intelligence platform.
    Handles data standardization, geocoding, customer profiling, and segmentation.
    """
    
    def __init__(self, 
                 geocoding_validation: bool = True,
                 customer_segmentation: bool = True,
                 territory_analysis: bool = True):
        
        self.geocoding_validation = geocoding_validation
        self.customer_segmentation = customer_segmentation
        self.territory_analysis = territory_analysis
        
        # Initialize processors
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
        self.segmentation_models = {}
        
        # Sri Lanka geographic constraints
        self.sri_lanka_bounds = {
            'lat_min': 5.9, 'lat_max': 9.9,
            'lon_min': 79.6, 'lon_max': 81.9
        }
        
        # Major Sri Lankan cities and their coordinates
        self.major_cities = {
            'COLOMBO': (6.9271, 79.8612),
            'KANDY': (7.2906, 80.6337),
            'GALLE': (6.0329, 80.2168),
            'JAFFNA': (9.6615, 80.0255),
            'NEGOMBO': (7.2084, 79.8358),
            'ANURADHAPURA': (8.3114, 80.4037),
            'RATNAPURA': (6.6828, 80.3992),
            'BATTICALOA': (7.7102, 81.6924),
            'TRINCOMALEE': (8.5874, 81.2152),
            'KURUNEGALA': (7.4863, 80.3623)
        }
    
    def preprocess_customer_data(self, customer_df: pd.DataFrame, 
                               customer_location_df: Optional[pd.DataFrame] = None,
                               sales_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Main preprocessing pipeline for customer data.
        
        Args:
            customer_df: Customer data with columns: No., City, Contact, etc.
            customer_location_df: Customer location data with lat/lon
            sales_df: Sales data for customer profiling
        
        Returns:
            Preprocessed customer dataframe with enriched features
        """
        logger.info(f"Starting customer preprocessing for {len(customer_df)} customers")
        
        # Make a copy to avoid modifying original data
        processed_df = customer_df.copy()
        
        # Step 1: Basic data cleaning and standardization
        processed_df = self._clean_basic_data(processed_df)
        
        # Step 2: Merge with location data if available
        if customer_location_df is not None:
            processed_df = self._merge_location_data(processed_df, customer_location_df)
        
        # Step 3: Geocoding and location validation
        if self.geocoding_validation:
            processed_df = self._validate_and_enrich_locations(processed_df)
        
        # Step 4: Extract and standardize contact information
        processed_df = self._process_contact_information(processed_df)
        
        # Step 5: Create geographic features
        processed_df = self._create_geographic_features(processed_df)
        
        # Step 6: Customer profiling using sales data
        if sales_df is not None:
            processed_df = self._create_customer_profiles(processed_df, sales_df)
        
        # Step 7: Customer segmentation
        if self.customer_segmentation:
            processed_df = self._perform_customer_segmentation(processed_df)
        
        # Step 8: Territory analysis
        if self.territory_analysis:
            processed_df = self._analyze_territories(processed_df)
        
        # Step 9: Feature encoding and normalization
        processed_df = self._encode_and_normalize_features(processed_df)
        
        # Step 10: Data quality assessment
        processed_df = self._assess_data_quality(processed_df)
        
        logger.info(f"Customer preprocessing completed. {len(processed_df)} customers processed")
        
        return processed_df
    
    def _clean_basic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning and standardization."""
        
        # Standardize customer ID column name
        if 'No.' in df.columns:
            df = df.rename(columns={'No.': 'CustomerID'})
        elif 'Customer ID' in df.columns:
            df = df.rename(columns={'Customer ID': 'CustomerID'})
        
        # Clean and standardize text fields
        text_columns = ['City', 'Contact', 'CustomerID']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                # Remove extra spaces
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        
        # Remove duplicate customers
        df = df.drop_duplicates(subset=['CustomerID'])
        
        # Handle missing values in basic fields
        df['City'] = df['City'].fillna('UNKNOWN')
        df['Contact'] = df['Contact'].fillna('NO_CONTACT')
        
        # Clean city names (remove common prefixes/suffixes)
        if 'City' in df.columns:
            df['City_Clean'] = df['City'].str.replace(r'^\d+\s*', '', regex=True)  # Remove leading numbers
            df['City_Clean'] = df['City_Clean'].str.replace(r'\s+(CITY|TOWN|DISTRICT)$', '', regex=True)
        
        return df
    
    def _merge_location_data(self, customer_df: pd.DataFrame, 
                           location_df: pd.DataFrame) -> pd.DataFrame:
        """Merge customer data with location coordinates."""
        
        # Standardize the merge key
        if 'Customer_ID' in location_df.columns:
            location_df = location_df.rename(columns={'Customer_ID': 'CustomerID'})
        
        # Ensure CustomerID is string in both dataframes
        customer_df['CustomerID'] = customer_df['CustomerID'].astype(str)
        location_df['CustomerID'] = location_df['CustomerID'].astype(str)
        
        # Merge the dataframes
        merged_df = customer_df.merge(
            location_df[['CustomerID', 'Latitude', 'Longitude']], 
            on='CustomerID', 
            how='left'
        )
        
        logger.info(f"Location merge: {merged_df['Latitude'].notna().sum()} customers have coordinates")
        
        return merged_df
    
    def _validate_and_enrich_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and enrich location data."""
        
        # Validate existing coordinates
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            # Convert to numeric
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            
            # Validate coordinates are within Sri Lanka bounds
            valid_coords_mask = (
                df['Latitude'].between(self.sri_lanka_bounds['lat_min'], self.sri_lanka_bounds['lat_max']) &
                df['Longitude'].between(self.sri_lanka_bounds['lon_min'], self.sri_lanka_bounds['lon_max'])
            )
            
            # Flag invalid coordinates
            df['has_valid_coordinates'] = valid_coords_mask & df['Latitude'].notna() & df['Longitude'].notna()
        else:
            df['Latitude'] = np.nan
            df['Longitude'] = np.nan
            df['has_valid_coordinates'] = False
        
        # Estimate coordinates for customers without valid coordinates
        df = self._estimate_missing_coordinates(df)
        
        return df
    
    def _estimate_missing_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate coordinates for customers without valid location data."""
        
        missing_coords_mask = ~df['has_valid_coordinates']
        
        if missing_coords_mask.sum() == 0:
            return df
        
        logger.info(f"Estimating coordinates for {missing_coords_mask.sum()} customers")
        
        # Method 1: Match with major cities
        for idx in df[missing_coords_mask].index:
            city = df.loc[idx, 'City_Clean'] if 'City_Clean' in df.columns else df.loc[idx, 'City']
            
            if city in self.major_cities:
                df.loc[idx, 'Latitude'] = self.major_cities[city]
                df.loc[idx, 'Longitude'] = self.major_cities[city][1]
                df.loc[idx, 'has_valid_coordinates'] = True
                df.loc[idx, 'coordinate_source'] = 'city_match'
                continue
        
        # Method 2: Use city-based clustering for remaining customers
        df = self._cluster_by_city_for_coordinates(df)
        
        return df
    
    def _cluster_by_city_for_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use clustering to estimate coordinates for customers in the same city."""
        
        # Group customers by city
        city_groups = df.groupby('City_Clean')
        
        for city, group in city_groups:
            # Skip if city has no customers with valid coordinates
            valid_coords_in_city = group[group['has_valid_coordinates']]
            if len(valid_coords_in_city) == 0:
                continue
            
            # Calculate centroid for customers without coordinates
            centroid_lat = valid_coords_in_city['Latitude'].mean()
            centroid_lon = valid_coords_in_city['Longitude'].mean()
            
            # Add some random noise to avoid exact duplicates
            for idx in group[~group['has_valid_coordinates']].index:
                noise_lat = np.random.normal(0, 0.01)  # ~1km noise
                noise_lon = np.random.normal(0, 0.01)
                
                df.loc[idx, 'Latitude'] = centroid_lat + noise_lat
                df.loc[idx, 'Longitude'] = centroid_lon + noise_lon
                df.loc[idx, 'has_valid_coordinates'] = True
                df.loc[idx, 'coordinate_source'] = 'city_cluster'
        
        return df
    
    def _process_contact_information(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and standardize contact information."""
        
        if 'Contact' not in df.columns:
            return df
        
        # Extract phone numbers
        phone_pattern = r'(\+?94|0)?[1-9]\d{8}'
        df['phone_number'] = df['Contact'].str.extract(f'({phone_pattern})', expand=False)
        
        # Extract email addresses
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        df['email'] = df['Contact'].str.extract(f'({email_pattern})', expand=False)
        
        # Contact quality score
        df['contact_quality_score'] = 0
        df.loc[df['phone_number'].notna(), 'contact_quality_score'] += 1
        df.loc[df['email'].notna(), 'contact_quality_score'] += 1
        
        # Clean phone numbers
        df['phone_cleaned'] = df['phone_number'].str.replace(r'[^\d+]', '', regex=True)
        
        return df
    
    def _create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create geographic features for analysis."""
        
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            return df
        
        # Distance from Colombo (capital)
        colombo_coords = self.major_cities['COLOMBO']
        df['distance_from_colombo'] = df.apply(
            lambda row: geodesic(
                (row['Latitude'], row['Longitude']), 
                colombo_coords
            ).kilometers if pd.notna(row['Latitude']) else np.nan,
            axis=1
        )
        
        # Assign regions based on location
        df['region'] = df.apply(self._assign_region, axis=1)
        
        # Urban/Rural classification based on distance from major cities
        df['urban_rural'] = df.apply(self._classify_urban_rural, axis=1)
        
        # Coastal vs Inland
        df['coastal_inland'] = df.apply(self._classify_coastal_inland, axis=1)
        
        return df
    
    def _assign_region(self, row) -> str:
        """Assign region based on coordinates."""
        if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
            return 'UNKNOWN'
        
        lat, lon = row['Latitude'], row['Longitude']
        
        # Define regional boundaries for Sri Lanka
        if lat >= 8.5:
            return 'NORTHERN'
        elif lat >= 7.5:
            return 'NORTH_CENTRAL'
        elif lat >= 6.9 and lon >= 80.5:
            return 'EASTERN'
        elif lat >= 6.9:
            return 'CENTRAL'
        elif lon <= 80.0:
            return 'WESTERN'
        elif lat <= 6.3:
            return 'SOUTHERN'
        else:
            return 'CENTRAL'
    
    def _classify_urban_rural(self, row) -> str:
        """Classify as urban or rural based on proximity to major cities."""
        if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
            return 'UNKNOWN'
        
        customer_coords = (row['Latitude'], row['Longitude'])
        
        # Check distance to all major cities
        min_distance = float('inf')
        for city_coords in self.major_cities.values():
            distance = geodesic(customer_coords, city_coords).kilometers
            min_distance = min(min_distance, distance)
        
        if min_distance <= 10:  # Within 10km of major city
            return 'URBAN'
        elif min_distance <= 25:  # Within 25km
            return 'SUBURBAN'
        else:
            return 'RURAL'
    
    def _classify_coastal_inland(self, row) -> str:
        """Classify as coastal or inland based on proximity to coast."""
        if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
            return 'UNKNOWN'
        
        lat, lon = row['Latitude'], row['Longitude']
        
        # Simple coastal classification based on longitude
        # (This is a simplified approach - in practice, you'd use proper coastal boundaries)
        if lon <= 79.9 or lon >= 81.5:  # Western or Eastern coast
            return 'COASTAL'
        elif lat <= 6.2 and lon <= 80.5:  # Southern coast
            return 'COASTAL'
        else:
            return 'INLAND'
    
    def _create_customer_profiles(self, customer_df: pd.DataFrame, 
                                sales_df: pd.DataFrame) -> pd.DataFrame:
        """Create customer profiles based on sales data."""
        
        # Merge sales data with customer data
        sales_summary = self._create_sales_summary(sales_df)
        
        # Merge with customer data
        customer_df = customer_df.merge(
            sales_summary, 
            left_on='CustomerID', 
            right_on='DistributorCode', 
            how='left'
        )
        
        # Fill missing values for customers without sales
        sales_columns = ['total_sales', 'avg_order_value', 'order_frequency', 
                        'days_since_last_order', 'total_orders', 'sales_trend']
        
        for col in sales_columns:
            if col in customer_df.columns:
                customer_df[col] = customer_df[col].fillna(0)
        
        # Create customer value score
        customer_df['customer_value_score'] = self._calculate_customer_value_score(customer_df)
        
        return customer_df
    
    def _create_sales_summary(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Create sales summary statistics for each customer."""
        
        # Convert date columns
        if 'Date' in sales_df.columns:
            sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        
        # Group by customer (DistributorCode)
        sales_summary = sales_df.groupby('DistributorCode').agg({
            'FinalValue': ['sum', 'mean', 'count'],
            'Date': ['max', 'min']
        }).reset_index()
        
        # Flatten column names
        sales_summary.columns = ['DistributorCode', 'total_sales', 'avg_order_value', 
                               'total_orders', 'last_order_date', 'first_order_date']
        
        # Calculate additional metrics
        current_date = pd.Timestamp.now()
        sales_summary['days_since_last_order'] = (
            current_date - sales_summary['last_order_date']
        ).dt.days
        
        sales_summary['customer_lifetime_days'] = (
            sales_summary['last_order_date'] - sales_summary['first_order_date']
        ).dt.days
        
        sales_summary['order_frequency'] = (
            sales_summary['total_orders'] / 
            (sales_summary['customer_lifetime_days'] + 1)
        )
        
        # Calculate sales trend (simple linear trend)
        sales_summary['sales_trend'] = sales_df.groupby('DistributorCode').apply(
            self._calculate_sales_trend
        ).reset_index(drop=True)
        
        return sales_summary
    
    def _calculate_sales_trend(self, group_df: pd.DataFrame) -> float:
        """Calculate sales trend for a customer."""
        if len(group_df) < 2:
            return 0
        
        # Simple linear regression on sales over time
        group_df = group_df.sort_values('Date')
        x = np.arange(len(group_df))
        y = group_df['FinalValue'].values
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _calculate_customer_value_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate customer value score based on multiple factors."""
        
        # Normalize metrics
        scaler = StandardScaler()
        
        # Prepare features for scoring
        scoring_features = []
        feature_names = []
        
        if 'total_sales' in df.columns:
            scoring_features.append(df['total_sales'].fillna(0))
            feature_names.append('total_sales')
        
        if 'order_frequency' in df.columns:
            scoring_features.append(df['order_frequency'].fillna(0))
            feature_names.append('order_frequency')
        
        if 'days_since_last_order' in df.columns:
            # Inverse: recent customers get higher scores
            scoring_features.append(1 / (df['days_since_last_order'].fillna(365) + 1))
            feature_names.append('recency_score')
        
        if len(scoring_features) == 0:
            return pd.Series([0] * len(df))
        
        # Combine features
        features_matrix = np.column_stack(scoring_features)
        
        # Normalize and calculate weighted score
        features_normalized = scaler.fit_transform(features_matrix)
        
        # Weights for different aspects (Revenue, Frequency, Recency)
        weights = [0.5, 0.3, 0.2][:len(feature_names)]
        
        # Calculate weighted score
        customer_value_score = np.average(features_normalized, axis=1, weights=weights)
        
        return pd.Series(customer_value_score)
    
    def _perform_customer_segmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform customer segmentation using clustering."""
        
        # Prepare features for clustering
        clustering_features = []
        feature_names = []
        
        # Geographic features
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            clustering_features.extend([df['Latitude'].fillna(0), df['Longitude'].fillna(0)])
            feature_names.extend(['Latitude', 'Longitude'])
        
        # Sales features
        sales_features = ['total_sales', 'avg_order_value', 'order_frequency', 'customer_value_score']
        for feature in sales_features:
            if feature in df.columns:
                clustering_features.append(df[feature].fillna(0))
                feature_names.append(feature)
        
        if len(clustering_features) < 2:
            df['customer_segment'] = 'DEFAULT'
            return df
        
        # Prepare feature matrix
        X = np.column_stack(clustering_features)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        n_clusters = min(8, len(df) // 50)  # Adaptive number of clusters
        n_clusters = max(3, n_clusters)  # At least 3 clusters
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Assign cluster labels
        df['customer_segment'] = cluster_labels
        
        # Create meaningful segment names
        df['customer_segment_name'] = df['customer_segment'].map(
            self._create_segment_names(df, cluster_labels)
        )
        
        # Store clustering model
        self.segmentation_models['kmeans'] = kmeans
        self.segmentation_models['scaler'] = scaler
        self.segmentation_models['feature_names'] = feature_names
        
        return df
    
    def _create_segment_names(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> dict:
        """Create meaningful names for customer segments."""
        
        segment_names = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_df = df[cluster_mask]
            
            # Analyze cluster characteristics
            avg_sales = cluster_df['total_sales'].mean() if 'total_sales' in cluster_df.columns else 0
            avg_frequency = cluster_df['order_frequency'].mean() if 'order_frequency' in cluster_df.columns else 0
            avg_value_score = cluster_df['customer_value_score'].mean() if 'customer_value_score' in cluster_df.columns else 0
            
            # Create segment name based on characteristics
            if avg_value_score > 0.5:
                if avg_frequency > 0.1:
                    segment_names[cluster_id] = 'HIGH_VALUE_FREQUENT'
                else:
                    segment_names[cluster_id] = 'HIGH_VALUE_OCCASIONAL'
            elif avg_value_score > 0:
                if avg_frequency > 0.05:
                    segment_names[cluster_id] = 'MEDIUM_VALUE_REGULAR'
                else:
                    segment_names[cluster_id] = 'MEDIUM_VALUE_INFREQUENT'
            else:
                segment_names[cluster_id] = 'LOW_VALUE'
        
        return segment_names
    
    def _analyze_territories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze and assign territories to customers."""
        
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            return df
        
        # Use geographic clustering for territory assignment
        valid_coords_mask = df['has_valid_coordinates']
        
        if valid_coords_mask.sum() < 10:
            df['territory'] = 'DEFAULT'
            return df
        
        # Prepare coordinates for clustering
        coords = df[valid_coords_mask][['Latitude', 'Longitude']].values
        
        # Use DBSCAN for territory clustering (handles irregular shapes)
        dbscan = DBSCAN(eps=0.1, min_samples=10)  # ~11km radius
        territory_labels = dbscan.fit_predict(coords)
        
        # Assign territories
        df['territory'] = 'OUTLIER'
        df.loc[valid_coords_mask, 'territory'] = territory_labels
        
        # Create territory names
        df['territory_name'] = df['territory'].apply(
            lambda x: f'TERRITORY_{x}' if x >= 0 else 'OUTLIER'
        )
        
        return df
    
    def _encode_and_normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables and normalize numerical features."""
        
        # Categorical columns to encode
        categorical_columns = ['City', 'City_Clean', 'region', 'urban_rural', 
                             'coastal_inland', 'customer_segment_name', 'territory_name']
        
        for col in categorical_columns:
            if col in df.columns:
                # Label encoding for categorical variables
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
        
        # Numerical columns to normalize
        numerical_columns = ['distance_from_colombo', 'total_sales', 'avg_order_value', 
                           'order_frequency', 'customer_value_score']
        
        for col in numerical_columns:
            if col in df.columns:
                # Standard scaling for numerical variables
                scaler = StandardScaler()
                df[f'{col}_normalized'] = scaler.fit_transform(df[[col]].fillna(0))
                self.scalers[col] = scaler
        
        return df
    
    def _assess_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assess and score data quality for each customer record."""
        
        df['data_quality_score'] = 0
        
        # Check for valid coordinates
        if 'has_valid_coordinates' in df.columns:
            df['data_quality_score'] += df['has_valid_coordinates'].astype(int) * 0.3
        
        # Check for contact information
        if 'contact_quality_score' in df.columns:
            df['data_quality_score'] += (df['contact_quality_score'] / 2) * 0.2
        
        # Check for sales data availability
        if 'total_sales' in df.columns:
            df['data_quality_score'] += (df['total_sales'] > 0).astype(int) * 0.3
        
        # Check for complete city information
        if 'City' in df.columns:
            df['data_quality_score'] += (df['City'] != 'UNKNOWN').astype(int) * 0.2
        
        # Overall completeness
        df['data_completeness'] = df.notna().mean(axis=1)
        
        return df
    
    def get_preprocessing_summary(self, df: pd.DataFrame) -> dict:
        """Get summary of preprocessing results."""
        
        summary = {
            'total_customers': len(df),
            'customers_with_coordinates': df['has_valid_coordinates'].sum() if 'has_valid_coordinates' in df.columns else 0,
            'customers_with_sales_data': (df['total_sales'] > 0).sum() if 'total_sales' in df.columns else 0,
            'average_data_quality_score': df['data_quality_score'].mean() if 'data_quality_score' in df.columns else 0,
            'regional_distribution': df['region'].value_counts().to_dict() if 'region' in df.columns else {},
            'urban_rural_distribution': df['urban_rural'].value_counts().to_dict() if 'urban_rural' in df.columns else {},
            'customer_segments': df['customer_segment_name'].value_counts().to_dict() if 'customer_segment_name' in df.columns else {},
            'territories': df['territory_name'].nunique() if 'territory_name' in df.columns else 0
        }
        
        return summary
    
    def transform_new_customer(self, customer_data: dict) -> dict:
        """Transform a new customer record using fitted preprocessors."""
        
        # This method would be used for real-time preprocessing of new customers
        # using the fitted encoders and scalers
        
        transformed_data = customer_data.copy()
        
        # Apply saved encoders and scalers
        for col, encoder in self.encoders.items():
            if col in transformed_data:
                try:
                    transformed_data[f'{col}_encoded'] = encoder.transform([str(transformed_data[col])])[0]
                except ValueError:
                    # Handle unseen categories
                    transformed_data[f'{col}_encoded'] = -1
        
        for col, scaler in self.scalers.items():
            if col in transformed_data:
                transformed_data[f'{col}_normalized'] = scaler.transform([[transformed_data[col]]])[0][0]
        
        return transformed_data