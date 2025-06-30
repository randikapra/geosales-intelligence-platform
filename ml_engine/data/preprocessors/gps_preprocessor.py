import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime, timedelta
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPSPreprocessor:
    """
    Advanced GPS data preprocessing for geosales intelligence platform.
    Handles data cleaning, outlier detection, route extraction, and quality validation.
    """
    
    def __init__(self, 
                 speed_threshold: float = 200.0,  # km/h - unrealistic speed threshold
                 max_distance_jump: float = 50.0,  # km - max distance between consecutive points
                 min_time_gap: int = 5,  # seconds - minimum time between points
                 max_time_gap: int = 3600):  # seconds - maximum time gap for same route
        
        self.speed_threshold = speed_threshold
        self.max_distance_jump = max_distance_jump
        self.min_time_gap = min_time_gap
        self.max_time_gap = max_time_gap
        
        # Sri Lanka bounding box (approximate)
        self.sri_lanka_bounds = {
            'lat_min': 5.9,
            'lat_max': 9.9,
            'lon_min': 79.6,
            'lon_max': 81.9
        }
    
    def preprocess_gps_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing pipeline for GPS data.
        
        Args:
            df: Raw GPS dataframe with columns: DivisionCode, UserCode, UserName, 
                Latitude, Longitude, TourCode, RecievedDate
        
        Returns:
            Preprocessed GPS dataframe with additional features
        """
        logger.info(f"Starting GPS preprocessing for {len(df)} records")
        
        # Make a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Step 1: Basic data cleaning
        processed_df = self._clean_basic_data(processed_df)
        
        # Step 2: Coordinate validation and correction
        processed_df = self._validate_coordinates(processed_df)
        
        # Step 3: Remove duplicates and near-duplicates
        processed_df = self._remove_duplicates(processed_df)
        
        # Step 4: Detect and remove outliers
        processed_df = self._detect_outliers(processed_df)
        
        # Step 5: Calculate movement features
        processed_df = self._calculate_movement_features(processed_df)
        
        # Step 6: Extract route segments
        processed_df = self._extract_route_segments(processed_df)
        
        # Step 7: Detect stops and visits
        processed_df = self._detect_stops(processed_df)
        
        # Step 8: Quality scoring
        processed_df = self._calculate_quality_score(processed_df)
        
        logger.info(f"GPS preprocessing completed. {len(processed_df)} records remaining")
        
        return processed_df
    
    def _clean_basic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning and standardization."""
        
        # Convert date column to datetime
        df['RecievedDate'] = pd.to_datetime(df['RecievedDate'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['RecievedDate'])
        
        # Convert coordinates to numeric
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        
        # Remove rows with missing coordinates
        df = df.dropna(subset=['Latitude', 'Longitude'])
        
        # Standardize text fields
        df['UserCode'] = df['UserCode'].astype(str).str.strip()
        df['UserName'] = df['UserName'].astype(str).str.strip()
        df['TourCode'] = df['TourCode'].astype(str).str.strip()
        df['DivisionCode'] = df['DivisionCode'].astype(str).str.strip()
        
        # Sort by user and time
        df = df.sort_values(['UserCode', 'RecievedDate']).reset_index(drop=True)
        
        return df
    
    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and correct GPS coordinates."""
        
        initial_count = len(df)
        
        # Remove coordinates outside Sri Lanka bounds (with some buffer)
        df = df[
            (df['Latitude'].between(self.sri_lanka_bounds['lat_min'] - 0.1, 
                                   self.sri_lanka_bounds['lat_max'] + 0.1)) &
            (df['Longitude'].between(self.sri_lanka_bounds['lon_min'] - 0.1, 
                                    self.sri_lanka_bounds['lon_max'] + 0.1))
        ]
        
        # Remove invalid coordinates (zeros, extreme values)
        df = df[
            (df['Latitude'] != 0) & 
            (df['Longitude'] != 0) &
            (df['Latitude'].abs() < 90) &
            (df['Longitude'].abs() < 180)
        ]
        
        # Flag potential coordinate precision issues
        df['coord_precision'] = df.apply(
            lambda row: len(str(row['Latitude']).split('.')[-1]) + 
                        len(str(row['Longitude']).split('.')[-1]), 
            axis=1
        )
        
        logger.info(f"Coordinate validation: {initial_count - len(df)} invalid points removed")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact and near-duplicate GPS points."""
        
        initial_count = len(df)
        
        # Remove exact duplicates
        df = df.drop_duplicates(
            subset=['UserCode', 'Latitude', 'Longitude', 'RecievedDate']
        )
        
        # Remove near-duplicates (same user, location within 10m, time within 30 seconds)
        df_clean = []
        
        for user_code in df['UserCode'].unique():
            user_df = df[df['UserCode'] == user_code].copy()
            
            if len(user_df) <= 1:
                df_clean.append(user_df)
                continue
            
            # Calculate time differences
            user_df['time_diff'] = user_df['RecievedDate'].diff().dt.total_seconds()
            
            # Keep first point and points that are far enough in time/space
            keep_indices = [0]  # Always keep first point
            
            for i in range(1, len(user_df)):
                current_point = (user_df.iloc[i]['Latitude'], user_df.iloc[i]['Longitude'])
                prev_point = (user_df.iloc[i-1]['Latitude'], user_df.iloc[i-1]['Longitude'])
                
                distance = geodesic(current_point, prev_point).meters
                time_diff = user_df.iloc[i]['time_diff']
                
                # Keep point if it's far enough or enough time has passed
                if distance > 10 or time_diff > 30:
                    keep_indices.append(i)
            
            df_clean.append(user_df.iloc[keep_indices])
        
        df = pd.concat(df_clean, ignore_index=True)
        
        logger.info(f"Duplicate removal: {initial_count - len(df)} duplicates removed")
        
        return df
    
    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and remove GPS outliers using multiple methods."""
        
        initial_count = len(df)
        outlier_flags = pd.Series(False, index=df.index)
        
        # Method 1: Speed-based outlier detection
        for user_code in df['UserCode'].unique():
            user_df = df[df['UserCode'] == user_code].copy()
            
            if len(user_df) <= 1:
                continue
            
            # Calculate speeds between consecutive points
            speeds = []
            for i in range(1, len(user_df)):
                point1 = (user_df.iloc[i-1]['Latitude'], user_df.iloc[i-1]['Longitude'])
                point2 = (user_df.iloc[i]['Latitude'], user_df.iloc[i]['Longitude'])
                
                distance = geodesic(point1, point2).kilometers
                time_diff = (user_df.iloc[i]['RecievedDate'] - 
                           user_df.iloc[i-1]['RecievedDate']).total_seconds() / 3600
                
                if time_diff > 0:
                    speed = distance / time_diff
                    speeds.append(speed)
                else:
                    speeds.append(0)
            
            # Flag points with unrealistic speeds
            for i, speed in enumerate(speeds):
                if speed > self.speed_threshold:
                    outlier_flags.loc[user_df.index[i+1]] = True
        
        # Method 2: Distance jump detection
        for user_code in df['UserCode'].unique():
            user_df = df[df['UserCode'] == user_code].copy()
            
            if len(user_df) <= 1:
                continue
            
            for i in range(1, len(user_df)):
                point1 = (user_df.iloc[i-1]['Latitude'], user_df.iloc[i-1]['Longitude'])
                point2 = (user_df.iloc[i]['Latitude'], user_df.iloc[i]['Longitude'])
                
                distance = geodesic(point1, point2).kilometers
                
                if distance > self.max_distance_jump:
                    outlier_flags.loc[user_df.index[i]] = True
        
        # Method 3: DBSCAN clustering for spatial outliers
        for user_code in df['UserCode'].unique():
            user_df = df[df['UserCode'] == user_code].copy()
            
            if len(user_df) < 10:  # Skip if too few points
                continue
            
            coords = user_df[['Latitude', 'Longitude']].values
            
            # Apply DBSCAN
            dbscan = DBSCAN(eps=0.01, min_samples=3)  # eps in degrees (~1km)
            clusters = dbscan.fit_predict(coords)
            
            # Flag points in small clusters or noise
            for i, cluster in enumerate(clusters):
                if cluster == -1:  # Noise points
                    outlier_flags.loc[user_df.index[i]] = True
        
        df['is_outlier'] = outlier_flags
        df_clean = df[~outlier_flags].copy()
        
        logger.info(f"Outlier detection: {initial_count - len(df_clean)} outliers removed")
        
        return df_clean
    
    def _calculate_movement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate movement-related features."""
        
        movement_features = []
        
        for user_code in df['UserCode'].unique():
            user_df = df[df['UserCode'] == user_code].copy()
            
            if len(user_df) <= 1:
                # Single point - set default values
                for idx in user_df.index:
                    movement_features.append({
                        'index': idx,
                        'speed_kmh': 0,
                        'distance_to_prev_m': 0,
                        'time_gap_seconds': 0,
                        'bearing_degrees': 0,
                        'acceleration_ms2': 0
                    })
                continue
            
            # Sort by time
            user_df = user_df.sort_values('RecievedDate')
            
            for i, (idx, row) in enumerate(user_df.iterrows()):
                if i == 0:
                    # First point
                    movement_features.append({
                        'index': idx,
                        'speed_kmh': 0,
                        'distance_to_prev_m': 0,
                        'time_gap_seconds': 0,
                        'bearing_degrees': 0,
                        'acceleration_ms2': 0
                    })
                    continue
                
                # Current and previous points
                curr_point = (row['Latitude'], row['Longitude'])
                prev_row = user_df.iloc[i-1]
                prev_point = (prev_row['Latitude'], prev_row['Longitude'])
                
                # Distance and time calculations
                distance_m = geodesic(prev_point, curr_point).meters
                time_diff = (row['RecievedDate'] - prev_row['RecievedDate']).total_seconds()
                
                # Speed calculation
                speed_kmh = (distance_m / 1000) / (time_diff / 3600) if time_diff > 0 else 0
                
                # Bearing calculation
                bearing = self._calculate_bearing(prev_point, curr_point)
                
                # Acceleration calculation (if we have previous speed)
                acceleration = 0
                if i > 1:
                    prev_speed_ms = movement_features[-1]['speed_kmh'] * 1000 / 3600
                    curr_speed_ms = speed_kmh * 1000 / 3600
                    acceleration = (curr_speed_ms - prev_speed_ms) / time_diff if time_diff > 0 else 0
                
                movement_features.append({
                    'index': idx,
                    'speed_kmh': min(speed_kmh, 300),  # Cap at reasonable maximum
                    'distance_to_prev_m': distance_m,
                    'time_gap_seconds': time_diff,
                    'bearing_degrees': bearing,
                    'acceleration_ms2': acceleration
                })
        
        # Convert to DataFrame and merge
        features_df = pd.DataFrame(movement_features).set_index('index')
        df = df.join(features_df)
        
        return df
    
    def _calculate_bearing(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """Calculate bearing between two GPS points."""
        
        lat1, lon1 = np.radians(point1[0]), np.radians(point1[1])
        lat2, lon2 = np.radians(point2[0]), np.radians(point2[1])
        
        dlon = lon2 - lon1
        
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        
        bearing = np.degrees(np.arctan2(y, x))
        return (bearing + 360) % 360
    
    def _extract_route_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract meaningful route segments from GPS data."""
        
        df['route_segment_id'] = 0
        segment_id = 1
        
        for user_code in df['UserCode'].unique():
            user_mask = df['UserCode'] == user_code
            user_df = df[user_mask].copy()
            
            if len(user_df) <= 1:
                continue
            
            current_segment = segment_id
            df.loc[user_df.index[0], 'route_segment_id'] = current_segment
            
            for i in range(1, len(user_df)):
                curr_idx = user_df.index[i]
                prev_idx = user_df.index[i-1]
                
                time_gap = df.loc[curr_idx, 'time_gap_seconds']
                distance = df.loc[curr_idx, 'distance_to_prev_m']
                
                # Start new segment if large time gap or distance jump
                if time_gap > self.max_time_gap or distance > self.max_distance_jump * 1000:
                    segment_id += 1
                    current_segment = segment_id
                
                df.loc[curr_idx, 'route_segment_id'] = current_segment
            
            segment_id += 1
        
        return df
    
    def _detect_stops(self, df: pd.DataFrame, 
                     stop_radius_m: float = 50, 
                     min_stop_duration_s: int = 300) -> pd.DataFrame:
        """Detect stops and visits from GPS data."""
        
        df['is_stop'] = False
        df['stop_duration_s'] = 0
        df['stop_id'] = 0
        
        stop_id = 1
        
        for user_code in df['UserCode'].unique():
            user_mask = df['UserCode'] == user_code
            user_df = df[user_mask].copy().sort_values('RecievedDate')
            
            if len(user_df) <= 1:
                continue
            
            # Detect stops using speed and proximity
            potential_stops = user_df[
                (user_df['speed_kmh'] < 5) &  # Low speed
                (user_df['distance_to_prev_m'] < stop_radius_m)  # Small movement
            ]
            
            if len(potential_stops) == 0:
                continue
            
            # Group consecutive stop points
            stop_groups = []
            current_group = [potential_stops.index[0]]
            
            for i in range(1, len(potential_stops)):
                curr_idx = potential_stops.index[i]
                prev_idx = potential_stops.index[i-1]
                
                curr_time = df.loc[curr_idx, 'RecievedDate']
                prev_time = df.loc[prev_idx, 'RecievedDate']
                
                time_diff = (curr_time - prev_time).total_seconds()
                
                if time_diff < 600:  # Within 10 minutes
                    current_group.append(curr_idx)
                else:
                    if len(current_group) > 1:
                        stop_groups.append(current_group)
                    current_group = [curr_idx]
            
            if len(current_group) > 1:
                stop_groups.append(current_group)
            
            # Process each stop group
            for group in stop_groups:
                if len(group) < 3:  # Require at least 3 points
                    continue
                
                start_time = df.loc[group[0], 'RecievedDate']
                end_time = df.loc[group[-1], 'RecievedDate']
                duration = (end_time - start_time).total_seconds()
                
                if duration >= min_stop_duration_s:
                    # Mark as stop
                    df.loc[group, 'is_stop'] = True
                    df.loc[group, 'stop_duration_s'] = duration
                    df.loc[group, 'stop_id'] = stop_id
                    stop_id += 1
        
        return df
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate data quality score for each GPS record."""
        
        df['quality_score'] = 100.0  # Start with perfect score
        
        # Reduce score based on various factors
        
        # Time gap penalty
        df.loc[df['time_gap_seconds'] > 300, 'quality_score'] -= 10
        df.loc[df['time_gap_seconds'] > 1800, 'quality_score'] -= 20
        
        # Speed penalty (too high or too low for too long)
        df.loc[df['speed_kmh'] > 100, 'quality_score'] -= 15
        df.loc[df['speed_kmh'] > 150, 'quality_score'] -= 25
        
        # Coordinate precision penalty
        df.loc[df['coord_precision'] < 10, 'quality_score'] -= 10
        
        # Distance jump penalty
        df.loc[df['distance_to_prev_m'] > 5000, 'quality_score'] -= 15
        
        # Ensure score doesn't go below 0
        df['quality_score'] = df['quality_score'].clip(lower=0)
        
        return df
    
    def generate_preprocessing_report(self, original_df: pd.DataFrame, 
                                    processed_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a comprehensive preprocessing report."""
        
        report = {
            'summary': {
                'original_records': len(original_df),
                'processed_records': len(processed_df),
                'records_removed': len(original_df) - len(processed_df),
                'removal_rate': (len(original_df) - len(processed_df)) / len(original_df) * 100
            },
            'quality_metrics': {
                'avg_quality_score': processed_df['quality_score'].mean(),
                'high_quality_records': len(processed_df[processed_df['quality_score'] >= 80]),
                'low_quality_records': len(processed_df[processed_df['quality_score'] < 50])
            },
            'movement_analysis': {
                'avg_speed_kmh': processed_df['speed_kmh'].mean(),
                'max_speed_kmh': processed_df['speed_kmh'].max(),
                'total_distance_km': processed_df['distance_to_prev_m'].sum() / 1000,
                'stops_detected': processed_df['is_stop'].sum(),
                'unique_routes': processed_df['route_segment_id'].nunique()
            },
            'coverage_analysis': {
                'unique_users': processed_df['UserCode'].nunique(),
                'unique_tours': processed_df['TourCode'].nunique(),
                'date_range': {
                    'start': processed_df['RecievedDate'].min(),
                    'end': processed_df['RecievedDate'].max()
                },
                'geographic_bounds': {
                    'lat_min': processed_df['Latitude'].min(),
                    'lat_max': processed_df['Latitude'].max(),
                    'lon_min': processed_df['Longitude'].min(),
                    'lon_max': processed_df['Longitude'].max()
                }
            }
        }
        
        return report

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = GPSPreprocessor()
    
    # Load sample data (replace with actual data loading)
    sample_data = {
        'DivisionCode': ['202'] * 5,
        'UserCode': ['AT-001651'] * 5,
        'UserName': ['A Thananshayan'] * 5,
        'Latitude': [9.70872, 9.70868, 9.70875, 9.70880, 9.70885],
        'Longitude': [80.07447, 80.07445, 80.07450, 80.07455, 80.07460],
        'TourCode': ['TU202U26523001'] * 5,
        'RecievedDate': pd.date_range('2023-01-27 09:38:22', periods=5, freq='2min')
    }
    
    df = pd.DataFrame(sample_data)
    
    # Preprocess the data
    processed_df = preprocessor.preprocess_gps_data(df)
    
    # Generate report
    report = preprocessor.generate_preprocessing_report(df, processed_df)
    
    print("GPS Preprocessing Report:")
    print(f"Original records: {report['summary']['original_records']}")
    print(f"Processed records: {report['summary']['processed_records']}")
    print(f"Average quality score: {report['quality_metrics']['avg_quality_score']:.2f}")