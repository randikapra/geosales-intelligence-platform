import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgb

# Geospatial
from geopy.distance import geodesic
from sklearn.neighbors import NearestNeighbors

# Time Series
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import holidays

# Utilities
import joblib
import json
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from pathlib import Path
import folium
import plotly.express as px
import plotly.graph_objects as go

@dataclass
class DemandConfig:
    """Configuration for demand forecasting"""
    forecast_horizon: int = 30
    location_radius_km: int = 5
    min_historical_data: int = 30
    validation_split: float = 0.2
    model_save_path: str = "models/demand/"
    confidence_level: float = 0.95

class DemandForecaster:
    """
    Advanced Demand Forecasting System with Geospatial Intelligence
    Predicts demand by location, product, and temporal patterns
    """
    
    def __init__(self, config: DemandConfig = None):
        self.config = config or DemandConfig()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.location_clusters = None
        self.customer_locations = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create model directory
        Path(self.config.model_save_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize Sri Lankan holidays
        self.sri_lanka_holidays = holidays.SriLanka()
    
    def load_customer_locations(self, customer_df: pd.DataFrame):
        """
        Load customer location data
        """
        self.customer_locations = customer_df.copy()
        self.logger.info(f"Loaded {len(customer_df)} customer locations")
    
    def prepare_demand_data(self, orders_df: pd.DataFrame, gps_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Prepare comprehensive demand dataset with spatial and temporal features
        """
        self.logger.info("Preparing demand data...")
        
        # Convert date columns
        orders_df['Date'] = pd.to_datetime(orders_df['Date'])
        orders_df['CreationDate'] = pd.to_datetime(orders_df['CreationDate'])
        
        # Merge with customer locations if available
        if self.customer_locations is not None:
            orders_df = orders_df.merge(
                self.customer_locations[['DistributorCode', 'Latitude', 'Longitude']],
                left_on='DistributorCode',
                right_on='DistributorCode',
                how='left'
            )
        
        # Create comprehensive features
        demand_data = self._create_demand_features(orders_df, gps_df)
        
        return demand_data
    
    def _create_demand_features(self, orders_df: pd.DataFrame, gps_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create comprehensive feature set for demand prediction
        """
        df = orders_df.copy()
        
        # === TEMPORAL FEATURES ===
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for temporal features
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfMonth_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
        df['DayOfMonth_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
        
        # Holiday features
        df['IsHoliday'] = df['Date'].apply(lambda x: x in self.sri_lanka_holidays).astype(int)
        df['DaysToHoliday'] = df['Date'].apply(self._days_to_next_holiday)
        df['DaysFromHoliday'] = df['Date'].apply(self._days_from_last_holiday)
        
        # === GEOSPATIAL FEATURES ===
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            # Distance from major cities (Sri Lanka)
            major_cities = {
                'Colombo': (6.9271, 79.8612),
                'Kandy': (7.2906, 80.6337),
                'Galle': (6.0329, 80.217),
                'Jaffna': (9.6615, 80.0255),
                'Anuradhapura': (8.3114, 80.4037)
            }
            
            for city, (lat, lon) in major_cities.items():
                df[f'Distance_to_{city}'] = df.apply(
                    lambda row: self._calculate_distance(
                        (row['Latitude'], row['Longitude']), (lat, lon)
                    ) if pd.notna(row['Latitude']) else np.nan, axis=1
                )
            
            # Location clustering
            df = self._add_location_clusters(df)
            
            # Nearby demand features
            df = self._add_nearby_demand_features(df)
        
        # === DEALER PERFORMANCE FEATURES ===
        dealer_stats = self._calculate_dealer_statistics(df)
        df = df.merge(dealer_stats, on='UserCode', how='left')
        
        # === CUSTOMER FEATURES ===
        customer_stats = self._calculate_customer_statistics(df)
        df = df.merge(customer_stats, on='DistributorCode', how='left')
        
        # === DEMAND PATTERNS ===
        df = self._add_demand_pattern_features(df)
        
        # === EXTERNAL FACTORS ===
        df = self._add_external_factors(df, gps_df)
        
        # Fill missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        return df
    
    def _calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates in kilometers"""
        try:
            return geodesic(coord1, coord2).kilometers
        except:
            return np.nan
    
    def _days_to_next_holiday(self, date: datetime) -> int:
        """Calculate days to next holiday"""
        for i in range(1, 366):  # Check next year
            future_date = date + timedelta(days=i)
            if future_date in self.sri_lanka_holidays:
                return i
        return 365
    
    def _days_from_last_holiday(self, date: datetime) -> int:
        """Calculate days from last holiday"""
        for i in range(1, 366):  # Check past year
            past_date = date - timedelta(days=i)
            if past_date in self.sri_lanka_holidays:
                return i
        return 365
    
    def _add_location_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add location-based clustering features"""
        if self.location_clusters is None:
            # Create location clusters
            location_data = df[['Latitude', 'Longitude']].dropna()
            if len(location_data) > 0:
                kmeans = KMeans(n_clusters=min(10, len(location_data)//10), random_state=42)
                self.location_clusters = kmeans.fit(location_data)
        
        # Assign clusters
        df['LocationCluster'] = np.nan
        valid_coords = df[['Latitude', 'Longitude']].notna().all(axis=1)
        if valid_coords.any():
            df.loc[valid_coords, 'LocationCluster'] = self.location_clusters.predict(
                df.loc[valid_coords, ['Latitude', 'Longitude']]
            )
        
        return df
    
    def _add_nearby_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on nearby demand patterns"""
        df['NearbyDemand_7days'] = 0
        df['NearbyDemand_30days'] = 0
        df['NearbyCustomers'] = 0
        
        # For each location, find nearby demand
        for idx, row in df.iterrows():
            if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                current_location = (row['Latitude'], row['Longitude'])
                current_date = row['Date']
                
                # Find nearby orders
                nearby_mask = df.apply(
                    lambda x: (
                        pd.notna(x['Latitude']) and pd.notna(x['Longitude']) and
                        self._calculate_distance(
                            (x['Latitude'], x['Longitude']), current_location
                        ) <= self.config.location_radius_km
                    ), axis=1
                )
                
                nearby_orders = df[nearby_mask]
                
                # Calculate nearby demand in different time windows
                date_7days = (nearby_orders['Date'] >= current_date - timedelta(days=7)) & \
                           (nearby_orders['Date'] < current_date)
                date_30days = (nearby_orders['Date'] >= current_date - timedelta(days=30)) & \
                            (nearby_orders['Date'] < current_date)
                
                df.at[idx, 'NearbyDemand_7days'] = nearby_orders[date_7days]['FinalValue'].sum()
                df.at[idx, 'NearbyDemand_30days'] = nearby_orders[date_30days]['FinalValue'].sum()
                df.at[idx, 'NearbyCustomers'] = nearby_orders['DistributorCode'].nunique()
        
        return df
    
    def _calculate_dealer_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate dealer performance statistics"""
        dealer_stats = df.groupby('UserCode').agg({
            'FinalValue': ['mean', 'std', 'sum', 'count'],
            'Date': ['min', 'max'],
            'DistributorCode': 'nunique'
        }).reset_index()
        
        # Flatten column names
        dealer_stats.columns = ['UserCode', 'DealerAvgSale', 'DealerStdSale', 'DealerTotalSales',
                               'DealerOrderCount', 'DealerFirstSale', 'DealerLastSale', 'DealerUniqueCustomers']
        
        # Calculate additional metrics
        dealer_stats['DealerExperience'] = (
            dealer_stats['DealerLastSale'] - dealer_stats['DealerFirstSale']
        ).dt.days
        
        dealer_stats['DealerSalesVelocity'] = dealer_stats['DealerTotalSales'] / (
            dealer_stats['DealerExperience'] + 1
        )
        
        return dealer_stats
    
    def _calculate_customer_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate customer behavior statistics"""
        customer_stats = df.groupby('DistributorCode').agg({
            'FinalValue': ['mean', 'std', 'sum', 'count'],
            'Date': ['min', 'max'],
            'UserCode': 'nunique'
        }).reset_index()
        
        # Flatten column names
        customer_stats.columns = ['DistributorCode', 'CustomerAvgOrder', 'CustomerStdOrder',
                                 'CustomerTotalPurchases', 'CustomerOrderCount', 'CustomerFirstOrder',
                                 'CustomerLastOrder', 'CustomerDealerCount']
        
        # Calculate additional metrics
        customer_stats['CustomerLifetime'] = (
            customer_stats['CustomerLastOrder'] - customer_stats['CustomerFirstOrder']
        ).dt.days
        
        customer_stats['CustomerLoyalty'] = customer_stats['CustomerOrderCount'] / (
            customer_stats['CustomerLifetime'] + 1
        )
        
        return customer_stats
    
    def _add_demand_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add demand pattern and seasonality features"""
        # Sort by date for lag features
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Create lag features by dealer
        for dealer in df['UserCode'].unique():
            dealer_mask = df['UserCode'] == dealer
            dealer_data = df[dealer_mask].copy()
            
            # Lag features
            for lag in [1, 3, 7, 14, 30]:
                lag_col = f'DemandLag_{lag}d'
                df.loc[dealer_mask, lag_col] = dealer_data['FinalValue'].shift(lag)
            
            # Rolling statistics
            for window in [7, 14, 30]:
                rolling_mean_col = f'DemandRollingMean_{window}d'
                rolling_std_col = f'DemandRollingStd_{window}d'
                df.loc[dealer_mask, rolling_mean_col] = dealer_data['FinalValue'].rolling(window=window).mean()
                df.loc[dealer_mask, rolling_std_col] = dealer_data['FinalValue'].rolling(window=window).std()
        
        # Exponential moving averages
        df['DemandEMA_7'] = df.groupby('UserCode')['FinalValue'].transform(lambda x: x.ewm(span=7).mean())
        df['DemandEMA_30'] = df.groupby('UserCode')['FinalValue'].transform(lambda x: x.ewm(span=30).mean())
        
        # Trend features
        df['DemandTrend'] = df.groupby('UserCode')['FinalValue'].transform(lambda x: x.pct_change())
        df['DemandVolatility'] = df.groupby('UserCode')['DemandTrend'].transform(lambda x: x.rolling(7).std())
        
        return df
    
    def _add_external_factors(self, df: pd.DataFrame, gps_df: pd.DataFrame = None) -> pd.DataFrame:
        """Add external factors that might influence demand"""
        # Economic indicators (placeholder - you could add actual economic data)
        df['EconomicIndex'] = 100  # Base economic index
        
        # Weather influence (placeholder - you could add weather data)
        df['WeatherIndex'] = df['Month'].apply(
            lambda x: 0.8 if x in [4, 5, 10, 11] else 1.0  # Monsoon months
        )
        
        # Market competition (based on dealer density)
        if gps_df is not None:
            dealer_density = self._calculate_dealer_density(df, gps_df)
            df = df.merge(dealer_density, on=['UserCode', 'Date'], how='left')
        
        # Supply chain factors
        df['SupplyChainDelay'] = df['Date'].apply(
            lambda x: 1 if x.weekday() in [5, 6] else 0  # Weekend delays
        )
        
        return df
    
    def _calculate_dealer_density(self, df: pd.DataFrame, gps_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate dealer density in area"""
        # Simplify GPS data to daily locations
        gps_df['Date'] = pd.to_datetime(gps_df['RecievedDate']).dt.date
        daily_locations = gps_df.groupby(['UserCode', 'Date']).agg({
            'Latitude': 'mean',
            'Longitude': 'mean'
        }).reset_index()
        
        # Calculate dealer density for each dealer-date combination
        density_data = []
        
        for _, row in daily_locations.iterrows():
            current_location = (row['Latitude'], row['Longitude'])
            current_date = row['Date']
            
            # Count other dealers within radius on same date
            other_dealers = daily_locations[
                (daily_locations['Date'] == current_date) & 
                (daily_locations['UserCode'] != row['UserCode'])
            ]
            
            nearby_count = 0
            for _, other in other_dealers.iterrows():
                other_location = (other['Latitude'], other['Longitude'])
                distance = self._calculate_distance(current_location, other_location)
                if distance <= self.config.location_radius_km:
                    nearby_count += 1
            
            density_data.append({
                'UserCode': row['UserCode'],
                'Date': pd.to_datetime(current_date),
                'DealerDensity': nearby_count
            })
        
        return pd.DataFrame(density_data)
    
    def train_demand_models(self, demand_data: pd.DataFrame) -> Dict:
        """
        Train multiple demand forecasting models
        """
        self.logger.info("Training demand forecasting models...")
        
        # Prepare features and target
        feature_columns = [col for col in demand_data.columns 
                          if col not in ['Date', 'FinalValue', 'Code', 'CreationDate', 'SubmittedDate', 'ERPOrderNumber']]
        
        X = demand_data[feature_columns].copy()
        y = demand_data['FinalValue'].copy()
        
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.encoders[col].transform(X[col].astype(str))
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.fillna(0))
        self.scalers['demand_features'] = scaler
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config.validation_split, random_state=42
        )
        
        # Initialize models
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=42,
                verbose=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            self.logger.info(f"Training {name} model...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_pred_train)
            test_metrics = self._calculate_metrics(y_test, y_pred_test)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_columns, model.feature_importances_))
                feature_importance = dict(sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True))
            else:
                feature_importance = {}
            
            results[name] = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance
            }
            
            self.models[name] = model
        
        # Create ensemble model
        ensemble_model = self._create_ensemble_model(results, X_train, y_train, X_test, y_test)
        results['ensemble'] = ensemble_model
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def _create_ensemble_model(self, results: Dict, X_train: np.ndarray, y_train: np.ndarray, 
                              X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Create ensemble model from individual models"""
        # Calculate weights based on test performance
        weights = {}
        total_weight = 0
        
        for name, result in results.items():
            if 'test_metrics' in result:
                # Use inverse of RMSE as weight
                rmse = result['test_metrics']['rmse']
                weight = 1 / (rmse + 1e-8)
                weights[name] = weight
                total_weight += weight
        
        # Normalize weights
        for name in weights:
            weights[name] /= total_weight
        
        # Create ensemble predictions
        ensemble_pred_train = np.zeros(len(y_train))
        ensemble_pred_test = np.zeros(len(y_test))
        
        for name, weight in weights.items():
            model = results[name]['model']
            ensemble_pred_train += weight * model.predict(X_train)
            ensemble_pred_test += weight * model.predict(X_test)
        
        # Calculate ensemble metrics
        train_metrics = self._calculate_metrics(y_train, ensemble_pred_train)
        test_metrics = self._calculate_metrics(y_test, ensemble_pred_test)
        
        return {
            'weights': weights,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'predictions': {
                'train': ensemble_pred_train,
                'test': ensemble_pred_test
            }
        }
    
    def predict_demand(self, location: Tuple[float, float], date: datetime, 
                      dealer_code: str = None, model_name: str = 'ensemble') -> Dict:
        """
        Predict demand for specific location and date
        """
        if model_name not in self.models and model_name != 'ensemble':
            raise ValueError(f"Model {model_name} not found")
        
        # Create feature vector for prediction
        features = self._create_prediction_features(location, date, dealer_code)
        
        if model_name == 'ensemble':
            # Use ensemble prediction
            prediction = self._ensemble_predict(features)
        else:
            # Use single model
            model = self.models[model_name]
            prediction = model.predict(features.reshape(1, -1))[0]
        
        # Calculate confidence interval (simplified)
        confidence_interval = self._calculate_confidence_interval(prediction, model_name)
        
        return {
            'predicted_demand': prediction,
            'confidence_interval': confidence_interval,
            'location': location,
            'date': date,
            'model_used': model_name
        }
    
    def _create_prediction_features(self, location: Tuple[float, float], 
                                   date: datetime, dealer_code: str = None) -> np.ndarray:
        """
        Create feature vector for a single prediction
        """
        # This is a simplified version - you'd need to implement full feature creation
        # based on your specific feature engineering pipeline
        
        features = []
        
        # Temporal features
        features.extend([
            date.year,
            date.month,
            date.day,
            date.weekday(),
            date.isocalendar()[1],  # week of year
            (date.month - 1) // 3 + 1,  # quarter
            1 if date.weekday() >= 5 else 0,  # is weekend
        ])
        
        # Cyclical encoding
        features.extend([
            np.sin(2 * np.pi * date.month / 12),
            np.cos(2 * np.pi * date.month / 12),
            np.sin(2 * np.pi * date.weekday() / 7),
            np.cos(2 * np.pi * date.weekday() / 7),
        ])
        
        # Location features
        if location:
            lat, lon = location
            features.extend([lat, lon])
            
            # Distance to major cities
            major_cities = {
                'Colombo': (6.9271, 79.8612),
                'Kandy': (7.2906, 80.6337),
                'Galle': (6.0329, 80.217),
            }
            
            for city, (city_lat, city_lon) in major_cities.items():
                distance = self._calculate_distance((lat, lon), (city_lat, city_lon))
                features.append(distance)
        
        # Pad features to match training dimensions (simplified)
        while len(features) < 50:  # Adjust based on actual feature count
            features.append(0)
        
        return np.array(features[:50])  # Truncate if too long
    
    def _ensemble_predict(self, features: np.ndarray) -> float:
        """
        Make ensemble prediction
        """
        if 'ensemble' not in self.models or not hasattr(self.models['ensemble'], 'weights'):
            # Fallback to simple average
            predictions = []
            for name, model in self.models.items():
                if name != 'ensemble':
                    pred = model.predict(features.reshape(1, -1))[0]
                    predictions.append(pred)
            return np.mean(predictions) if predictions else 0
        
        # Weighted ensemble prediction
        ensemble_info = self.models['ensemble']
        weights = ensemble_info['weights']
        
        weighted_sum = 0
        for name, weight in weights.items():
            if name in self.models:
                model = self.models[name]
                pred = model.predict(features.reshape(1, -1))[0]
                weighted_sum += weight * pred
        
        return weighted_sum
    
    def _calculate_confidence_interval(self, prediction: float, model_name: str) -> Tuple[float, float]:
        """
        Calculate confidence interval for prediction (simplified)
        """
        # This is a simplified approach - in practice, you'd use model-specific methods
        std_error = prediction * 0.1  # Assume 10% standard error
        z_score = stats.norm.ppf((1 + self.config.confidence_level) / 2)
        
        lower_bound = prediction - z_score * std_error
        upper_bound = prediction + z_score * std_error
        
        return (max(0, lower_bound), upper_bound)
    
    def generate_demand_heatmap(self, date: datetime, region_bounds: Dict = None) -> folium.Map:
        """
        Generate demand heatmap for a specific date
        """
        # Default to Sri Lanka bounds
        if region_bounds is None:
            region_bounds = {
                'min_lat': 5.9, 'max_lat': 9.9,
                'min_lon': 79.5, 'max_lon': 81.9
            }
        
        # Create grid of locations
        lat_range = np.linspace(region_bounds['min_lat'], region_bounds['max_lat'], 20)
        lon_range = np.linspace(region_bounds['min_lon'], region_bounds['max_lon'], 20)
        
        # Generate demand predictions for grid
        demand_data = []
        for lat in lat_range:
            for lon in lon_range:
                try:
                    demand_pred = self.predict_demand((lat, lon), date)
                    demand_data.append([lat, lon, demand_pred['predicted_demand']])
                except:
                    demand_data.append([lat, lon, 0])
        
        # Create map
        map_center = [
            (region_bounds['min_lat'] + region_bounds['max_lat']) / 2,
            (region_bounds['min_lon'] + region_bounds['max_lon']) / 2
        ]
        
        m = folium.Map(location=map_center, zoom_start=8)
        
        # Add heatmap
        from folium.plugins import HeatMap
        HeatMap(demand_data).add_to(m)
        
        return m
    
    def save_models(self):
        """Save trained models"""
        for name, model in self.models.items():
            filename = f"{self.config.model_save_path}/demand_{name}_model.pkl"
            joblib.dump(model, filename)
            self.logger.info(f"Saved {name} model to {filename}")
        
        # Save encoders and scalers
        joblib.dump(self.encoders, f"{self.config.model_save_path}/demand_encoders.pkl")
        joblib.dump(self.scalers, f"{self.config.model_save_path}/demand_scalers.pkl")
    
    def load_models(self):
        """Load trained models"""
        try:
            self.encoders = joblib.load(f"{self.config.model_save_path}/demand_encoders.pkl")
            self.scalers = joblib.load(f"{self.config.model_save_path}/demand_scalers.pkl")
            
            model_files = Path(self.config.model_save_path).glob("demand_*_model.pkl")
            for file_path in model_files:
                model_name = file_path.stem.replace("demand_", "").replace("_model", "")
                self.models[model_name] = joblib.load(file_path)
                self.logger.info(f"Loaded {model_name} model")
        except FileNotFoundError as e:
            self.logger.error(f"Model files not found: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize demand forecaster
    config = DemandConfig(
        forecast_horizon=30,
        location_radius_km=5,
        model_save_path="models/demand/"
    )
    
    forecaster = DemandForecaster(config)
    
    # Example usage (you would load your actual data)
    # customer_df = pd.read_csv("customer.csv")
    # orders_df = pd.read_excel("SFA_Orders.xlsx", sheet_name="Jan")
    # gps_df = pd.read_csv("SFA_GPSData.csv")
    
    # forecaster.load_customer_locations(customer_df)
    # demand_data = forecaster.prepare_demand_data(orders_df, gps_df)
    # results = forecaster.train_demand_models(demand_data)
    
    print("Demand Forecaster initialized successfully!")