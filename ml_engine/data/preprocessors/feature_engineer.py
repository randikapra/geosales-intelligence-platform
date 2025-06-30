import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Advanced feature engineering for geosales intelligence platform.
    Creates temporal, spatial, behavioral, and interaction features from raw data.
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_cache = {}
        
        # Sri Lanka specific constants
        self.working_hours = (8, 18)  # 8 AM to 6 PM
        self.business_days = [0, 1, 2, 3, 4]  # Monday to Friday
        
        # Seasonal patterns for Sri Lanka
        self.monsoon_months = {
            'southwest': [5, 6, 7, 8, 9],  # May-September
            'northeast': [10, 11, 12, 1, 2]  # October-February
        }
        
        self.festive_seasons = {
            'new_year': [(4, 13), (4, 14)],  # Sinhala Tamil New Year
            'vesak': [(5, 15)],  # Vesak (approximate)
            'christmas': [(12, 25)],
            'deepavali': [(10, 15)]  # Approximate
        }
    
    def create_comprehensive_features(self, 
                                    customer_df: pd.DataFrame,
                                    sales_df: pd.DataFrame,
                                    gps_df: pd.DataFrame,
                                    po_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create comprehensive feature set from all available data sources.
        
        Args:
            customer_df: Processed customer data
            sales_df: Sales transaction data
            gps_df: GPS tracking data
            po_df: Purchase order data (optional)
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting comprehensive feature engineering")
        
        # Start with customer data as base
        features_df = customer_df.copy()
        
        # 1. Temporal Features
        features_df = self._create_temporal_features(features_df, sales_df)
        
        # 2. Spatial Features
        features_df = self._create_spatial_features(features_df, gps_df)
        
        # 3. Behavioral Features
        features_df = self._create_behavioral_features(features_df, sales_df)
        
        # 4. Sales Performance Features
        features_df = self._create_sales_performance_features(features_df, sales_df)
        
        # 5. Route and Movement Features
        features_df = self._create_movement_features(features_df, gps_df)
        
        # 6. Interaction Features
        features_df = self._create_interaction_features(features_df)
        
        # 7. Aggregated Features
        features_df = self._create_aggregated_features(features_df, sales_df)
        
        # 8. Rolling Window Features
        features_df = self._create_rolling_window_features(features_df, sales_df)
        
        # 9. Seasonal Features
        features_df = self._create_seasonal_features(features_df, sales_df)
        
        # 10. PO-based Features if available
        if po_df is not None:
            features_df = self._create_po_features(features_df, po_df)
        
        logger.info(f"Feature engineering completed. Created {len(features_df.columns)} features")
        
        return features_df
    
    def _create_temporal_features(self, customer_df: pd.DataFrame, 
                                sales_df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from sales data."""
        
        if sales_df.empty:
            return customer_df
        
        # Ensure date columns are datetime
        if 'Date' in sales_df.columns:
            sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        
        # Group by customer
        temporal_features = []
        
        for customer_id in customer_df['CustomerID'].unique():
            customer_sales = sales_df[sales_df['DistributorCode'] == customer_id]
            
            if customer_sales.empty:
                temporal_features.append(self._create_empty_temporal_features(customer_id))
                continue
            
            # Sort by date
            customer_sales = customer_sales.sort_values('Date')
            
            features = {
                'CustomerID': customer_id,
                
                # Basic temporal features
                'first_purchase_date': customer_sales['Date'].min(),
                'last_purchase_date': customer_sales['Date'].max(),
                'days_as_customer': (customer_sales['Date'].max() - customer_sales['Date'].min()).days,
                'days_since_last_purchase': (datetime.now() - customer_sales['Date'].max()).days,
                'days_since_first_purchase': (datetime.now() - customer_sales['Date'].min()).days,
                
                # Purchase frequency features
                'total_purchases': len(customer_sales),
                'avg_days_between_purchases': self._calculate_avg_days_between_purchases(customer_sales['Date']),
                'purchase_frequency_per_month': len(customer_sales) / max(1, (customer_sales['Date'].max() - customer_sales['Date'].min()).days / 30),
                
                # Temporal patterns
                'most_common_purchase_day': customer_sales['Date'].dt.dayofweek.mode().iloc[0] if len(customer_sales) > 0 else 0,
                'most_common_purchase_hour': customer_sales['Date'].dt.hour.mode().iloc[0] if len(customer_sales) > 0 else 12,
                'weekend_purchase_ratio': (customer_sales['Date'].dt.dayofweek >= 5).mean(),
                'business_hours_purchase_ratio': ((customer_sales['Date'].dt.hour >= 8) & (customer_sales['Date'].dt.hour <= 18)).mean(),
                
                # Seasonal patterns
                'purchases_per_season': self._calculate_seasonal_purchases(customer_sales['Date']),
                'peak_season_months': self._identify_peak_season(customer_sales['Date']),
                
                # Recency, Frequency, Monetary (RFM) features
                'recency_score': min(100, max(1, 100 - (datetime.now() - customer_sales['Date'].max()).days)),
                'frequency_score': min(100, len(customer_sales) * 10),
                'monetary_score': min(100, customer_sales['FinalValue'].sum() / 1000) if 'FinalValue' in customer_sales.columns else 0,
            }
            
            temporal_features.append(features)
        
        # Convert to DataFrame and merge
        temporal_df = pd.DataFrame(temporal_features)
        
        # Merge with customer data
        result_df = customer_df.merge(temporal_df, on='CustomerID', how='left')
        
        # Fill missing values
        temporal_columns = [col for col in temporal_df.columns if col != 'CustomerID']
        for col in temporal_columns:
            if col in result_df.columns:
                if result_df[col].dtype in ['int64', 'float64']:
                    result_df[col] = result_df[col].fillna(0)
                else:
                    result_df[col] = result_df[col].fillna('unknown')
        
        return result_df
    
    def _create_spatial_features(self, customer_df: pd.DataFrame, 
                               gps_df: pd.DataFrame) -> pd.DataFrame:
        """Create spatial features from GPS data."""
        
        if gps_df.empty or 'Latitude' not in customer_df.columns:
            return customer_df
        
        # Create spatial features for each customer
        spatial_features = []
        
        for customer_id in customer_df['CustomerID'].unique():
            customer_row = customer_df[customer_df['CustomerID'] == customer_id].iloc[0]
            
            if pd.isna(customer_row['Latitude']) or pd.isna(customer_row['Longitude']):
                spatial_features.append(self._create_empty_spatial_features(customer_id))
                continue
            
            customer_coords = (customer_row['Latitude'], customer_row['Longitude'])
            
            # Find GPS data for dealers who visited this customer
            # This would require linking GPS data to customer visits
            # For now, we'll create features based on customer location
            
            features = {
                'CustomerID': customer_id,
                
                # Location-based features
                'is_urban': customer_row.get('urban_rural', 'UNKNOWN') == 'URBAN',
                'is_coastal': customer_row.get('coastal_inland', 'UNKNOWN') == 'COASTAL',
                'distance_from_capital': customer_row.get('distance_from_colombo', 0),
                
                # Accessibility features
                'accessibility_score': self._calculate_accessibility_score(customer_coords),
                'population_density_score': self._estimate_population_density(customer_coords),
                'commercial_activity_score': self._estimate_commercial_activity(customer_coords),
                
                # Nearest neighbor features
                'nearest_customer_distance': self._find_nearest_customer_distance(customer_coords, customer_df),
                'customer_density_5km': self._count_customers_within_radius(customer_coords, customer_df, 5),
                'customer_density_10km': self._count_customers_within_radius(customer_coords, customer_df, 10),
                'customer_density_20km': self._count_customers_within_radius(customer_coords, customer_df, 20),
                
                # Territory features
                'territory_centrality': self._calculate_territory_centrality(customer_coords, customer_df),
                'market_penetration_score': self._calculate_market_penetration(customer_coords, customer_df),
            }
            
            spatial_features.append(features)
        
        # Convert to DataFrame and merge
        spatial_df = pd.DataFrame(spatial_features)
        result_df = customer_df.merge(spatial_df, on='CustomerID', how='left')
        
        return result_df
    
    def _create_behavioral_features(self, customer_df: pd.DataFrame, 
                                  sales_df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral features from sales patterns."""
        
        if sales_df.empty:
            return customer_df
        
        behavioral_features = []
        
        for customer_id in customer_df['CustomerID'].unique():
            customer_sales = sales_df[sales_df['DistributorCode'] == customer_id]
            
            if customer_sales.empty:
                behavioral_features.append(self._create_empty_behavioral_features(customer_id))
                continue
            
            # Convert date column
            if 'Date' in customer_sales.columns:
                customer_sales['Date'] = pd.to_datetime(customer_sales['Date'])
            
            features = {
                'CustomerID': customer_id,
                
                # Order behavior
                'avg_order_value': customer_sales['FinalValue'].mean() if 'FinalValue' in customer_sales.columns else 0,
                'order_value_std': customer_sales['FinalValue'].std() if 'FinalValue' in customer_sales.columns else 0,
                'order_value_cv': (customer_sales['FinalValue'].std() / customer_sales['FinalValue'].mean()) if 'FinalValue' in customer_sales.columns and customer_sales['FinalValue'].mean() > 0 else 0,
                'max_order_value': customer_sales['FinalValue'].max() if 'FinalValue' in customer_sales.columns else 0,
                'min_order_value': customer_sales['FinalValue'].min() if 'FinalValue' in customer_sales.columns else 0,
                
                # Purchase patterns
                'orders_per_dealer': len(customer_sales['UserCode'].unique()) if 'UserCode' in customer_sales.columns else 0,
                'preferred_dealer_loyalty': (customer_sales['UserCode'].value_counts().iloc[0] / len(customer_sales)) if 'UserCode' in customer_sales.columns and len(customer_sales) > 0 else 0,
                'dealer_switching_frequency': len(customer_sales['UserCode'].unique()) / len(customer_sales) if 'UserCode' in customer_sales.columns and len(customer_sales) > 0 else 0,
                
                # Temporal behavior
                'purchase_regularity': self._calculate_purchase_regularity(customer_sales['Date']),
                'seasonal_variation': self._calculate_seasonal_variation(customer_sales),
                'trend_direction': self._calculate_trend_direction(customer_sales),
                
                # Value behavior
                'value_growth_rate': self._calculate_value_growth_rate(customer_sales),
                'price_sensitivity': self._calculate_price_sensitivity(customer_sales),
                'bulk_purchase_ratio': self._calculate_bulk_purchase_ratio(customer_sales),
            }
            
            behavioral_features.append(features)
        
        # Convert to DataFrame and merge
        behavioral_df = pd.DataFrame(behavioral_features)
        result_df = customer_df.merge(behavioral_df, on='CustomerID', how='left')
        
        return result_df
    
    def _create_sales_performance_features(self, customer_df: pd.DataFrame, 
                                         sales_df: pd.DataFrame) -> pd.DataFrame:
        """Create sales performance features."""
        
        if sales_df.empty:
            return customer_df
        
        performance_features = []
        
        # Calculate overall statistics for comparison
        total_sales = sales_df['FinalValue'].sum() if 'FinalValue' in sales_df.columns else 1
        avg_customer_value = sales_df.groupby('DistributorCode')['FinalValue'].sum().mean() if 'FinalValue' in sales_df.columns else 1
        
        for customer_id in customer_df['CustomerID'].unique():
            customer_sales = sales_df[sales_df['DistributorCode'] == customer_id]
            
            if customer_sales.empty:
                performance_features.append(self._create_empty_performance_features(customer_id))
                continue
            
            total_value = customer_sales['FinalValue'].sum() if 'FinalValue' in customer_sales.columns else 0
            
            features = {
                'CustomerID': customer_id,
                
                # Revenue features
                'total_revenue': total_value,
                'revenue_rank': self._calculate_revenue_rank(customer_id, sales_df),
                'revenue_percentile': (total_value / total_sales) * 100 if total_sales > 0 else 0,
                'revenue_vs_avg': total_value / avg_customer_value if avg_customer_value > 0 else 0,
                
                # Growth features
                'revenue_growth_3m': self._calculate_period_growth(customer_sales, 3),
                'revenue_growth_6m': self._calculate_period_growth(customer_sales, 6),
                'revenue_growth_12m': self._calculate_period_growth(customer_sales, 12),
                
                # Consistency features
                'revenue_consistency': self._calculate_revenue_consistency(customer_sales),
                'monthly_active_ratio': self._calculate_monthly_active_ratio(customer_sales),
                'purchase_momentum': self._calculate_purchase_momentum(customer_sales),
                
                # Efficiency features
                'orders_to_revenue_ratio': len(customer_sales) / total_value if total_value > 0 else 0,
                'avg_fulfillment_time': self._calculate_avg_fulfillment_time(customer_sales),
                'order_completion_rate': self._calculate_completion_rate(customer_sales),
            }
            
            performance_features.append(features)
        
        # Convert to DataFrame and merge
        performance_df = pd.DataFrame(performance_features)
        result_df = customer_df.merge(performance_df, on='CustomerID', how='left')
        
        return result_df
    
    def _create_movement_features(self, customer_df: pd.DataFrame, 
                                gps_df: pd.DataFrame) -> pd.DataFrame:
        """Create movement and route features from GPS data."""
        
        if gps_df.empty:
            return customer_df
        
        movement_features = []
        
        for customer_id in customer_df['CustomerID'].unique():
            # Find GPS records related to this customer (would need visit linking)
            # For now, we'll create aggregate features from nearby GPS activity
            
            customer_row = customer_df[customer_df['CustomerID'] == customer_id].iloc[0]
            
            if pd.isna(customer_row.get('Latitude')) or pd.isna(customer_row.get('Longitude')):
                movement_features.append(self._create_empty_movement_features(customer_id))
                continue
            
            customer_coords = (customer_row['Latitude'], customer_row['Longitude'])
            
            # Find GPS activity within reasonable distance (e.g., 1km)
            nearby_gps = self._find_nearby_gps_activity(customer_coords, gps_df, radius_km=1)
            
            features = {
                'CustomerID': customer_id,
                
                # Accessibility features
                'dealer_visits_nearby': len(nearby_gps['UserCode'].unique()) if len(nearby_gps) > 0 else 0,
                'avg_visit_frequency': self._calculate_visit_frequency(nearby_gps),
                'peak_visit_hours': self._identify_peak_visit_hours(nearby_gps),
                'visit_duration_avg': self._calculate_avg_visit_duration(nearby_gps),
                
                # Route features
                'route_accessibility': self._calculate_route_accessibility(customer_coords, gps_df),
                'traffic_pattern_score': self._calculate_traffic_pattern(customer_coords, gps_df),
                'route_efficiency_score': self._calculate_route_efficiency(customer_coords, gps_df),
                
                # Network features
                'dealer_network_density': self._calculate_dealer_network_density(customer_coords, gps_df),
                'service_coverage_score': self._calculate_service_coverage(customer_coords, gps_df),
            }
            
            movement_features.append(features)
        
        # Convert to DataFrame and merge
        movement_df = pd.DataFrame(movement_features)
        result_df = customer_df.merge(movement_df, on='CustomerID', how='left')
        
        return result_df
    
    def _create_interaction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between existing features."""
        
        # Revenue × Frequency interactions
        if 'total_revenue' in features_df.columns and 'total_purchases' in features_df.columns:
            features_df['revenue_frequency_interaction'] = features_df['total_revenue'] * features_df['total_purchases']
        
        # Distance × Value interactions
        if 'distance_from_capital' in features_df.columns and 'avg_order_value' in features_df.columns:
            features_df['distance_value_interaction'] = features_df['distance_from_capital'] * features_df['avg_order_value']
        
        # Loyalty × Performance interactions
        if 'preferred_dealer_loyalty' in features_df.columns and 'revenue_growth_3m' in features_df.columns:
            features_df['loyalty_growth_interaction'] = features_df['preferred_dealer_loyalty'] * features_df['revenue_growth_3m']
        
        # Temporal × Spatial interactions
        if 'purchase_frequency_per_month' in features_df.columns and 'customer_density_5km' in features_df.columns:
            features_df['temporal_spatial_interaction'] = features_df['purchase_frequency_per_month'] * features_df['customer_density_5km']
        
        # Create polynomial features for key metrics
        key_features = ['total_revenue', 'avg_order_value', 'purchase_frequency_per_month']
        for feature in key_features:
            if feature in features_df.columns:
                features_df[f'{feature}_squared'] = features_df[feature] ** 2
                features_df[f'{feature}_log'] = np.log1p(features_df[feature])
        
        return features_df
    
    def _create_aggregated_features(self, features_df: pd.DataFrame, 
                                  sales_df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features across different dimensions."""
        
        if sales_df.empty:
            return features_df
        
        # City-level aggregations
        if 'City' in features_df.columns:
            city_stats = sales_df.groupby('DistributorCode').agg({
                'FinalValue': ['mean', 'sum', 'std', 'count']
            }).reset_index() if 'FinalValue' in sales_df.columns else pd.DataFrame()
            
            if not city_stats.empty:
                city_stats.columns = ['CustomerID', 'city_avg_order_value', 'city_total_revenue', 
                                    'city_order_std', 'city_order_count']
                features_df = features_df.merge(city_stats, on='CustomerID', how='left')
        
        # Dealer-level aggregations
        if 'UserCode' in sales_df.columns:
            dealer_performance = sales_df.groupby('UserCode').agg({
                'FinalValue': ['mean', 'sum', 'count'],
                'DistributorCode': 'nunique'
            }).reset_index() if 'FinalValue' in sales_df.columns else pd.DataFrame()
            
            if not dealer_performance.empty:
                dealer_performance.columns = ['UserCode', 'dealer_avg_order', 'dealer_total_sales',
                                            'dealer_order_count', 'dealer_customer_count']
                
                # Join with customer-dealer relationships
                customer_dealer = sales_df.groupby('DistributorCode')['UserCode'].first().reset_index()
                customer_dealer = customer_dealer.merge(dealer_performance, on='UserCode', how='left')
                features_df = features_df.merge(customer_dealer[['DistributorCode', 'dealer_avg_order', 
                                                               'dealer_total_sales']], 
                                              left_on='CustomerID', right_on='DistributorCode', how='left')
        
        return features_df
    
    def _create_rolling_window_features(self, features_df: pd.DataFrame, 
                                      sales_df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features for trend analysis."""
        
        if sales_df.empty or 'Date' not in sales_df.columns:
            return features_df
        
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        rolling_features = []
        
        for customer_id in features_df['CustomerID'].unique():
            customer_sales = sales_df[sales_df['DistributorCode'] == customer_id].copy()
            
            if customer_sales.empty:
                rolling_features.append({'CustomerID': customer_id})
                continue
            
            customer_sales = customer_sales.sort_values('Date')
            customer_sales.set_index('Date', inplace=True)
            
            # Create monthly aggregations
            monthly_sales = customer_sales.resample('M')['FinalValue'].sum() if 'FinalValue' in customer_sales.columns else pd.Series()
            
            features = {'CustomerID': customer_id}
            
            if len(monthly_sales) > 0:
                # Rolling averages
                features['sales_3m_avg'] = monthly_sales.rolling(3, min_periods=1).mean().iloc[-1]
                features['sales_6m_avg'] = monthly_sales.rolling(6, min_periods=1).mean().iloc[-1]
                features['sales_12m_avg'] = monthly_sales.rolling(12, min_periods=1).mean().iloc[-1]
                
                # Rolling trends
                features['sales_3m_trend'] = self._calculate_rolling_trend(monthly_sales, 3)
                features['sales_6m_trend'] = self._calculate_rolling_trend(monthly_sales, 6)
                
                # Rolling volatility
                features['sales_3m_volatility'] = monthly_sales.rolling(3, min_periods=1).std().iloc[-1]
                features['sales_6m_volatility'] = monthly_sales.rolling(6, min_periods=1).std().iloc[-1]
            
            rolling_features.append(features)
        
        # Convert to DataFrame and merge
        rolling_df = pd.DataFrame(rolling_features)
        result_df = features_df.merge(rolling_df, on='CustomerID', how='left')
        
        return result_df
    
    def _create_seasonal_features(self, features_df: pd.DataFrame, 
                                sales_df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal and cyclical features."""
        
        if sales_df.empty or 'Date' not in sales_df.columns:
            return features_df
        
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        seasonal_features = []
        
        for customer_id in features_df['CustomerID'].unique():
            customer_sales = sales_df[sales_df['DistributorCode'] == customer_id]
            
            if customer_sales.empty:
                seasonal_features.append(self._create_empty_seasonal_features(customer_id))
                continue
            
            features = {
                'CustomerID': customer_id,
                
                # Seasonal patterns
                'monsoon_sales_ratio': self._calculate_monsoon_sales_ratio(customer_sales),
                'festive_season_boost': self._calculate_festive_season_boost(customer_sales),
                'quarter_seasonality': self._calculate_quarter_seasonality(customer_sales),
                
                # Cyclical patterns
                'weekly_pattern_strength': self._calculate_weekly_pattern_strength(customer_sales),
                'monthly_pattern_strength': self._calculate_monthly_pattern_strength(customer_sales),
                
                # Weather correlation (if available)
                'weather_sensitivity': self._estimate_weather_sensitivity(customer_sales),
            }
            
            seasonal_features.append(features)
        
        # Convert to DataFrame and merge
        seasonal_df = pd.DataFrame(seasonal_features)
        result_df = features_df.merge(seasonal_df, on='CustomerID', how='left')
        
        return result_df
    
    def _create_po_features(self, features_df: pd.DataFrame, 
                          po_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from purchase order data."""
        
        if po_df.empty:
            return features_df
        
        po_features = []
        
        for customer_id in features_df['CustomerID'].unique():
            customer_pos = po_df[po_df['DistributorCode'] == customer_id] if 'DistributorCode' in po_df.columns else pd.DataFrame()
            
            if customer_pos.empty:
                po_features.append(self._create_empty_po_features(customer_id))
                continue
            
            features = {
                'CustomerID': customer_id,
                
                # PO patterns
                'total_pos': len(customer_pos),
                'avg_po_value': customer_pos['Total'].mean() if 'Total' in customer_pos.columns else 0,
                'po_value_std': customer_pos['Total'].std() if 'Total' in customer_pos.columns else 0,
                'max_po_value': customer_pos['Total'].max() if 'Total' in customer_pos.columns else 0,
                
                # Delivery patterns
                'avg_delivery_days': customer_pos['Days'].mean() if 'Days' in customer_pos.columns else 0,
                'delivery_consistency': 1 / (1 + customer_pos['Days'].std()) if 'Days' in customer_pos.columns and customer_pos['Days'].std() > 0 else 1,
                'on_time_delivery_rate': (customer_pos['Days'] <= 7).mean() if 'Days' in customer_pos.columns else 0,
                
                # Location patterns
                'po_location_consistency': self._calculate_po_location_consistency(customer_pos),
                'delivery_distance_avg': self._calculate_avg_delivery_distance(customer_pos),
            }
            
            po_features.append(features)
        
        # Convert to DataFrame and merge
        po_df_features = pd.DataFrame(po_features)
        result_df = features_df.merge(po_df_features, on='CustomerID', how='left')
        
        return result_df
    
    # Helper methods for feature calculations
    def _calculate_avg_days_between_purchases(self, dates: pd.Series) -> float:
        """Calculate average days between purchases."""
        if len(dates) < 2:
            return 0
        
        dates_sorted = dates.sort_values()
        diff_days = dates_sorted.diff().dt.days.dropna()
        return diff_days.mean() if len(diff_days) > 0 else 0
    
    def _calculate_seasonal_purchases(self, dates: pd.Series) -> Dict:
        """Calculate purchases per season."""
        seasons = {
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'autumn': [9, 10, 11],
            'winter': [12, 1, 2]
        }
        
        seasonal_counts = {}
        for season, months in seasons.items():
            seasonal_counts[season] = sum(dates.dt.month.isin(months))
        
        return seasonal_counts
    
    def _identify_peak_season(self, dates: pd.Series) -> str:
        """Identify the peak season for purchases."""
        seasonal_purchases = self._calculate_seasonal_purchases(dates)
        return max(seasonal_purchases, key=seasonal_purchases.get)
    
    def _calculate_accessibility_score(self, coords: Tuple[float, float]) -> float:
        """Calculate accessibility score based on location."""
        # Simple heuristic - closer to major cities = higher accessibility
        major_cities = [
            (6.9271, 79.8612),  # Colombo
            (7.2906, 80.6337),  # Kandy
            (6.0535, 80.2210),  # Galle
        ]
        
        min_distance = min([geodesic(coords, city).kilometers for city in major_cities])
        return max(0, 100 - min_distance)  # Score decreases with distance
    
    def _estimate_population_density(self, coords: Tuple[float, float]) -> float:
        """Estimate population density score."""
        # Simple heuristic based on proximity to Colombo
        colombo = (6.9271, 79.8612)
        distance = geodesic(coords, colombo).kilometers
        return max(0, 100 - distance * 2)  # Higher score for areas closer to Colombo
    
    def _estimate_commercial_activity(self, coords: Tuple[float, float]) -> float:
        """Estimate commercial activity score."""
        # This would ideally use external data sources
        # For now, use distance from major commercial centers
        commercial_centers = [
            (6.9271, 79.8612),  # Colombo
            (7.2906, 80.6337),  # Kandy
            (6.9319, 79.8478),  # Mount Lavinia
        ]
        
        min_distance = min([geodesic(coords, center).kilometers for center in commercial_centers])
        return max(0, 80 - min_distance * 3)
    
    def _find_nearest_customer_distance(self, coords: Tuple[float, float], 
                                      customer_df: pd.DataFrame) -> float:
        """Find distance to nearest customer."""
        distances = []
        for _, customer in customer_df.iterrows():
            if pd.notna(customer.get('Latitude')) and pd.notna(customer.get('Longitude')):
                other_coords = (customer['Latitude'], customer['Longitude'])
                if coords != other_coords:
                    distances.append(geodesic(coords, other_coords).kilometers)
        
        return min(distances) if distances else float('inf')
    
    def _count_customers_within_radius(self, coords: Tuple[float, float], 
                                     customer_df: pd.DataFrame, radius_km: float) -> int:
        """Count customers within given radius."""
        count = 0
        for _, customer in customer_df.iterrows():
            if pd.notna(customer.get('Latitude')) and pd.notna(customer.get('Longitude')):
                other_coords = (customer['Latitude'], customer['Longitude'])
                if coords != other_coords:
                    distance = geodesic(coords, other_coords).kilometers
                    if distance <= radius_km:
                        count += 1
        return count
    
    def _calculate_territory_centrality(self, coords: Tuple[float, float], 
                                      customer_df: pd.DataFrame) -> float:
        """Calculate how central a customer is within their territory."""
        # This is a simplified calculation
        nearby_customers = self._count_customers_within_radius(coords, customer_df, 20)
        return min(100, nearby_customers * 10)
    
    def _calculate_market_penetration(self, coords: Tuple[float, float], 
                                    customer_df: pd.DataFrame) -> float:
        """Calculate market penetration score."""
        # Simple heuristic based on customer density
        density_5km = self._count_customers_within_radius(coords, customer_df, 5)
        density_20km = self._count_customers_within_radius(coords, customer_df, 20)
        
        if density_20km == 0:
            return 0
        
        return (density_5km / density_20km) * 100
    
    # Create empty feature dictionaries for missing data
    def _create_empty_temporal_features(self, customer_id: str) -> Dict:
        """Create empty temporal features for customers with no sales."""
        return {
            'CustomerID': customer_id,
            'first_purchase_date': None,
            'last_purchase_date': None,
            'days_as_customer': 0,
            'days_since_last_purchase': float('inf'),
            'days_since_first_purchase': float('inf'),
            'total_purchases': 0,
            'avg_days_between_purchases': 0,
            'purchase_frequency_per_month': 0,
            'most_common_purchase_day': 0,
            'most_common_purchase_hour': 12,
            'weekend_purchase_ratio': 0,
            'business_hours_purchase_ratio': 0,
            'purchases_per_season': {'spring': 0, 'summer': 0, 'autumn': 0, 'winter': 0},
            'peak_season_months': 'unknown',
            'recency_score': 0,
            'frequency_score': 0,
            'monetary_score': 0,
        }
    
    def _create_empty_spatial_features(self, customer_id: str) -> Dict:
        """Create empty spatial features."""
        return {
            'CustomerID': customer_id,
            'is_urban': False,
            'is_coastal': False,
            'distance_from_capital': 0,
            'accessibility_score': 0,
            'population_density_score': 0,
            'commercial_activity_score': 0,
            'nearest_customer_distance': float('inf'),
            'customer_density_5km': 0,
            'customer_density_10km': 0,
            'customer_density_20km': 0,
            'territory_centrality': 0,
            'market_penetration_score': 0,
        }
    
    def _create_empty_behavioral_features(self, customer_id: str) -> Dict:
        """Create empty behavioral features."""
        return {
            'CustomerID': customer_id,
            'avg_order_value': 0,
            'order_value_std': 0,
            'order_value_cv': 0,
            'max_order_value': 0,
            'min_order_value': 0,
            'orders_per_dealer': 0,
            'preferred_dealer_loyalty': 0,
            'dealer_switching_frequency': 0,
            'purchase_regularity': 0,
            'seasonal_variation': 0,
            'trend_direction': 0,
            'value_growth_rate': 0,
            'price_sensitivity': 0,
            'bulk_purchase_ratio': 0,
        }
    
    def _create_empty_performance_features(self, customer_id: str) -> Dict:
        """Create empty performance features."""
        return {
            'CustomerID': customer_id,
            'total_revenue': 0,
            'revenue_rank': 0,
            'revenue_percentile': 0,
            'revenue_vs_avg': 0,
            'revenue_growth_3m': 0,
            'revenue_growth_6m': 0,
            'revenue_growth_12m': 0,
            'revenue_consistency': 0,
            'monthly_active_ratio': 0,
            'purchase_momentum': 0,
            'orders_to_revenue_ratio': 0,
            'avg_fulfillment_time': 0,
            'order_completion_rate': 0,
        }
    
    def _create_empty_movement_features(self, customer_id: str) -> Dict:
        """Create empty movement features."""
        return {
            'CustomerID': customer_id,
            'dealer_visits_nearby': 0,
            'avg_visit_frequency': 0,
            'peak_visit_hours': 12,
            'visit_duration_avg': 0,
            'route_accessibility': 0,
            'traffic_pattern_score': 0,
            'route_efficiency_score': 0,
            'dealer_network_density': 0,
            'service_coverage_score': 0,
        }
    
    def _create_empty_seasonal_features(self, customer_id: str) -> Dict:
        """Create empty seasonal features."""
        return {
            'CustomerID': customer_id,
            'monsoon_sales_ratio': 0,
            'festive_season_boost': 0,
            'quarter_seasonality': 0,
            'weekly_pattern_strength': 0,
            'monthly_pattern_strength': 0,
            'weather_sensitivity': 0,
        }
    
    def _create_empty_po_features(self, customer_id: str) -> Dict:
        """Create empty PO features."""
        return {
            'CustomerID': customer_id,
            'total_pos': 0,
            'avg_po_value': 0,
            'po_value_std': 0,
            'max_po_value': 0,
            'avg_delivery_days': 0,
            'delivery_consistency': 0,
            'on_time_delivery_rate': 0,
            'po_location_consistency': 0,
            'delivery_distance_avg': 0,
        }
    
    # Additional helper methods with simplified implementations
    def _calculate_purchase_regularity(self, dates: pd.Series) -> float:
        """Calculate purchase regularity score."""
        if len(dates) < 2:
            return 0
        
        intervals = dates.sort_values().diff().dt.days.dropna()
        if len(intervals) == 0:
            return 0
        
        cv = intervals.std() / intervals.mean() if intervals.mean() > 0 else float('inf')
        return max(0, 100 - cv * 10)  # Lower CV = higher regularity
    
    def _calculate_seasonal_variation(self, customer_sales: pd.DataFrame) -> float:
        """Calculate seasonal variation in sales."""
        if 'Date' not in customer_sales.columns or len(customer_sales) < 4:
            return 0
        
        monthly_sales = customer_sales.groupby(customer_sales['Date'].dt.month)['FinalValue'].sum()
        if len(monthly_sales) < 2:
            return 0
        
        cv = monthly_sales.std() / monthly_sales.mean() if monthly_sales.mean() > 0 else 0
        return min(100, cv * 50)  # Normalize to 0-100 scale

    def _calculate_trend_direction(self, customer_sales: pd.DataFrame) -> float:
        """Calculate trend direction in sales over time."""
        if 'Date' not in customer_sales.columns or len(customer_sales) < 3:
            return 0
        
        # Sort by date and create time series
        sales_ts = customer_sales.sort_values('Date')
        if 'FinalValue' not in sales_ts.columns:
            return 0
        
        # Calculate simple linear trend
        x = np.arange(len(sales_ts))
        y = sales_ts['FinalValue'].values
        
        if len(y) < 2:
            return 0
        
        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return np.tanh(slope / sales_ts['FinalValue'].mean()) * 100 if sales_ts['FinalValue'].mean() > 0 else 0
    
    def _calculate_value_growth_rate(self, customer_sales: pd.DataFrame) -> float:
        """Calculate value growth rate over time."""
        if 'Date' not in customer_sales.columns or 'FinalValue' not in customer_sales.columns:
            return 0
        
        # Sort by date
        sales_ts = customer_sales.sort_values('Date')
        if len(sales_ts) < 2:
            return 0
        
        # Compare first and last month
        first_month_sales = sales_ts.head(max(1, len(sales_ts)//4))['FinalValue'].sum()
        last_month_sales = sales_ts.tail(max(1, len(sales_ts)//4))['FinalValue'].sum()
        
        if first_month_sales == 0:
            return 0
        
        return ((last_month_sales - first_month_sales) / first_month_sales) * 100
    
    def _calculate_price_sensitivity(self, customer_sales: pd.DataFrame) -> float:
        """Calculate price sensitivity based on order value variations."""
        if 'FinalValue' not in customer_sales.columns or len(customer_sales) < 3:
            return 0
        
        # Calculate coefficient of variation as proxy for price sensitivity
        cv = customer_sales['FinalValue'].std() / customer_sales['FinalValue'].mean()
        return min(100, cv * 50)  # Normalize to 0-100 scale
    
    def _calculate_bulk_purchase_ratio(self, customer_sales: pd.DataFrame) -> float:
        """Calculate ratio of bulk purchases."""
        if 'FinalValue' not in customer_sales.columns or len(customer_sales) == 0:
            return 0
        
        # Define bulk purchase as orders above 75th percentile
        threshold = customer_sales['FinalValue'].quantile(0.75)
        bulk_orders = (customer_sales['FinalValue'] >= threshold).sum()
        
        return (bulk_orders / len(customer_sales)) * 100
    
    def _calculate_revenue_rank(self, customer_id: str, sales_df: pd.DataFrame) -> int:
        """Calculate revenue rank among all customers."""
        if 'FinalValue' not in sales_df.columns:
            return 0
        
        customer_revenues = sales_df.groupby('DistributorCode')['FinalValue'].sum().sort_values(ascending=False)
        rank = list(customer_revenues.index).index(customer_id) + 1 if customer_id in customer_revenues.index else len(customer_revenues)
        return rank
    
    def _calculate_period_growth(self, customer_sales: pd.DataFrame, months: int) -> float:
        """Calculate growth rate for specific period."""
        if 'Date' not in customer_sales.columns or 'FinalValue' not in customer_sales.columns:
            return 0
        
        # Get data for the specified period
        cutoff_date = datetime.now() - timedelta(days=months*30)
        recent_sales = customer_sales[customer_sales['Date'] >= cutoff_date]
        older_sales = customer_sales[customer_sales['Date'] < cutoff_date]
        
        if len(recent_sales) == 0 or len(older_sales) == 0:
            return 0
        
        recent_value = recent_sales['FinalValue'].sum()
        older_value = older_sales['FinalValue'].sum()
        
        if older_value == 0:
            return 0
        
        return ((recent_value - older_value) / older_value) * 100
    
    def _calculate_revenue_consistency(self, customer_sales: pd.DataFrame) -> float:
        """Calculate revenue consistency score."""
        if 'Date' not in customer_sales.columns or 'FinalValue' not in customer_sales.columns:
            return 0
        
        # Group by month and calculate monthly revenues
        monthly_revenues = customer_sales.groupby(customer_sales['Date'].dt.to_period('M'))['FinalValue'].sum()
        
        if len(monthly_revenues) < 2:
            return 0
        
        # Calculate coefficient of variation (lower = more consistent)
        cv = monthly_revenues.std() / monthly_revenues.mean() if monthly_revenues.mean() > 0 else float('inf')
        return max(0, 100 - cv * 20)  # Convert to consistency score
    
    def _calculate_monthly_active_ratio(self, customer_sales: pd.DataFrame) -> float:
        """Calculate ratio of months customer was active."""
        if 'Date' not in customer_sales.columns or len(customer_sales) == 0:
            return 0
        
        # Get unique months with purchases
        active_months = customer_sales['Date'].dt.to_period('M').nunique()
        
        # Calculate total months since first purchase
        first_purchase = customer_sales['Date'].min()
        total_months = max(1, (datetime.now() - first_purchase).days // 30)
        
        return (active_months / total_months) * 100
    
    def _calculate_purchase_momentum(self, customer_sales: pd.DataFrame) -> float:
        """Calculate purchase momentum (recent activity vs historical)."""
        if 'Date' not in customer_sales.columns or len(customer_sales) < 4:
            return 0
        
        # Split data into recent (last 25%) and historical (first 75%)
        sorted_sales = customer_sales.sort_values('Date')
        split_point = int(len(sorted_sales) * 0.75)
        
        historical = sorted_sales.iloc[:split_point]
        recent = sorted_sales.iloc[split_point:]
        
        if len(historical) == 0 or len(recent) == 0:
            return 0
        
        # Calculate average monthly frequency for each period
        hist_freq = len(historical) / max(1, (historical['Date'].max() - historical['Date'].min()).days / 30)
        recent_freq = len(recent) / max(1, (recent['Date'].max() - recent['Date'].min()).days / 30)
        
        if hist_freq == 0:
            return 100 if recent_freq > 0 else 0
        
        return ((recent_freq - hist_freq) / hist_freq) * 100
    
    def _calculate_avg_fulfillment_time(self, customer_sales: pd.DataFrame) -> float:
        """Calculate average fulfillment time."""
        if 'CreationDate' not in customer_sales.columns or 'SubmittedDate' not in customer_sales.columns:
            return 0
        
        # Convert to datetime
        creation_dates = pd.to_datetime(customer_sales['CreationDate'])
        submitted_dates = pd.to_datetime(customer_sales['SubmittedDate'])
        
        # Calculate fulfillment times
        fulfillment_times = (submitted_dates - creation_dates).dt.total_seconds() / 3600  # Hours
        fulfillment_times = fulfillment_times[fulfillment_times >= 0]  # Remove negative values
        
        return fulfillment_times.mean() if len(fulfillment_times) > 0 else 0
    
    def _calculate_completion_rate(self, customer_sales: pd.DataFrame) -> float:
        """Calculate order completion rate."""
        if 'ERPOrderNumber' not in customer_sales.columns:
            return 1.0  # Assume 100% if no ERP data
        
        # Count orders with ERP numbers (completed orders)
        completed_orders = customer_sales['ERPOrderNumber'].notna().sum()
        total_orders = len(customer_sales)
        
        return (completed_orders / total_orders) * 100 if total_orders > 0 else 0
    
    def _find_nearby_gps_activity(self, coords: Tuple[float, float], 
                                 gps_df: pd.DataFrame, radius_km: float = 1) -> pd.DataFrame:
        """Find GPS activity within radius of customer location."""
        if gps_df.empty or 'Latitude' not in gps_df.columns or 'Longitude' not in gps_df.columns:
            return pd.DataFrame()
        
        nearby_records = []
        for _, record in gps_df.iterrows():
            if pd.notna(record['Latitude']) and pd.notna(record['Longitude']):
                gps_coords = (record['Latitude'], record['Longitude'])
                distance = geodesic(coords, gps_coords).kilometers
                if distance <= radius_km:
                    nearby_records.append(record)
        
        return pd.DataFrame(nearby_records) if nearby_records else pd.DataFrame()
    
    def _calculate_visit_frequency(self, nearby_gps: pd.DataFrame) -> float:
        """Calculate average visit frequency from GPS data."""
        if nearby_gps.empty or 'RecievedDate' not in nearby_gps.columns:
            return 0
        
        # Convert to datetime
        nearby_gps['RecievedDate'] = pd.to_datetime(nearby_gps['RecievedDate'])
        
        # Group by user and calculate visit frequency
        user_visits = nearby_gps.groupby('UserCode')['RecievedDate'].dt.date.nunique()
        
        return user_visits.mean() if len(user_visits) > 0 else 0
    
    def _identify_peak_visit_hours(self, nearby_gps: pd.DataFrame) -> int:
        """Identify peak visit hours from GPS data."""
        if nearby_gps.empty or 'RecievedDate' not in nearby_gps.columns:
            return 12
        
        # Convert to datetime and extract hour
        nearby_gps['RecievedDate'] = pd.to_datetime(nearby_gps['RecievedDate'])
        hours = nearby_gps['RecievedDate'].dt.hour
        
        return hours.mode().iloc[0] if len(hours) > 0 else 12
    
    def _calculate_avg_visit_duration(self, nearby_gps: pd.DataFrame) -> float:
        """Calculate average visit duration."""
        if nearby_gps.empty or 'RecievedDate' not in nearby_gps.columns:
            return 0
        
        # Group by user and tour code to identify visit sessions
        nearby_gps['RecievedDate'] = pd.to_datetime(nearby_gps['RecievedDate'])
        
        visit_durations = []
        for (user, tour), group in nearby_gps.groupby(['UserCode', 'TourCode']):
            if len(group) > 1:
                duration = (group['RecievedDate'].max() - group['RecievedDate'].min()).total_seconds() / 3600
                visit_durations.append(duration)
        
        return np.mean(visit_durations) if visit_durations else 0
    
    def _calculate_route_accessibility(self, coords: Tuple[float, float], 
                                     gps_df: pd.DataFrame) -> float:
        """Calculate route accessibility score."""
        # Find unique routes that pass near this location
        nearby_gps = self._find_nearby_gps_activity(coords, gps_df, radius_km=2)
        
        if nearby_gps.empty:
            return 0
        
        # Count unique dealers and routes
        unique_dealers = nearby_gps['UserCode'].nunique() if 'UserCode' in nearby_gps.columns else 0
        unique_routes = nearby_gps['TourCode'].nunique() if 'TourCode' in nearby_gps.columns else 0
        
        return min(100, (unique_dealers * 10) + (unique_routes * 5))
    
    def _calculate_traffic_pattern(self, coords: Tuple[float, float], 
                                 gps_df: pd.DataFrame) -> float:
        """Calculate traffic pattern score."""
        nearby_gps = self._find_nearby_gps_activity(coords, gps_df, radius_km=3)
        
        if nearby_gps.empty:
            return 0
        
        # Calculate traffic density
        traffic_density = len(nearby_gps) / max(1, nearby_gps['RecievedDate'].dt.date.nunique())
        
        return min(100, traffic_density * 2)
    
    def _calculate_route_efficiency(self, coords: Tuple[float, float], 
                                  gps_df: pd.DataFrame) -> float:
        """Calculate route efficiency score."""
        # This is a simplified calculation
        # In practice, would analyze actual route optimization
        nearby_gps = self._find_nearby_gps_activity(coords, gps_df, radius_km=5)
        
        if nearby_gps.empty:
            return 50  # Neutral score
        
        # Calculate based on visit frequency and coverage
        visit_frequency = len(nearby_gps) / max(1, nearby_gps['UserCode'].nunique())
        
        return min(100, visit_frequency * 10)
    
    def _calculate_dealer_network_density(self, coords: Tuple[float, float], 
                                        gps_df: pd.DataFrame) -> float:
        """Calculate dealer network density around customer."""
        nearby_gps = self._find_nearby_gps_activity(coords, gps_df, radius_km=10)
        
        if nearby_gps.empty:
            return 0
        
        unique_dealers = nearby_gps['UserCode'].nunique() if 'UserCode' in nearby_gps.columns else 0
        
        return min(100, unique_dealers * 20)
    
    def _calculate_service_coverage(self, coords: Tuple[float, float], 
                                  gps_df: pd.DataFrame) -> float:
        """Calculate service coverage score."""
        # Calculate coverage at different radii
        coverage_1km = len(self._find_nearby_gps_activity(coords, gps_df, 1))
        coverage_5km = len(self._find_nearby_gps_activity(coords, gps_df, 5))
        coverage_10km = len(self._find_nearby_gps_activity(coords, gps_df, 10))
        
        # Weighted coverage score
        coverage_score = (coverage_1km * 5) + (coverage_5km * 2) + coverage_10km
        
        return min(100, coverage_score / 10)
    
    def _calculate_rolling_trend(self, series: pd.Series, window: int) -> float:
        """Calculate rolling trend."""
        if len(series) < window:
            return 0
        
        # Get rolling average
        rolling_avg = series.rolling(window, min_periods=1).mean()
        
        if len(rolling_avg) < 2:
            return 0
        
        # Calculate trend (slope of last few points)
        x = np.arange(len(rolling_avg))
        y = rolling_avg.values
        
        slope = np.polyfit(x[-min(len(x), window):], y[-min(len(y), window):], 1)[0]
        
        return np.tanh(slope / rolling_avg.mean()) * 100 if rolling_avg.mean() > 0 else 0
    
    def _calculate_monsoon_sales_ratio(self, customer_sales: pd.DataFrame) -> float:
        """Calculate monsoon season sales ratio."""
        if 'Date' not in customer_sales.columns or 'FinalValue' not in customer_sales.columns:
            return 0
        
        # Define monsoon months (May-September for Southwest, October-February for Northeast)
        monsoon_months = [5, 6, 7, 8, 9, 10, 11, 12, 1, 2]
        
        total_sales = customer_sales['FinalValue'].sum()
        monsoon_sales = customer_sales[customer_sales['Date'].dt.month.isin(monsoon_months)]['FinalValue'].sum()
        
        return (monsoon_sales / total_sales) * 100 if total_sales > 0 else 0
    
    def _calculate_festive_season_boost(self, customer_sales: pd.DataFrame) -> float:
        """Calculate festive season sales boost."""
        if 'Date' not in customer_sales.columns or 'FinalValue' not in customer_sales.columns:
            return 0
        
        # Define festive months (April, May, October, December)
        festive_months = [4, 5, 10, 12]
        
        avg_monthly_sales = customer_sales.groupby(customer_sales['Date'].dt.month)['FinalValue'].sum().mean()
        festive_sales = customer_sales[customer_sales['Date'].dt.month.isin(festive_months)]['FinalValue'].sum()
        festive_months_count = customer_sales['Date'].dt.month.isin(festive_months).sum()
        
        if festive_months_count == 0 or avg_monthly_sales == 0:
            return 0
        
        avg_festive_sales = festive_sales / festive_months_count
        
        return ((avg_festive_sales - avg_monthly_sales) / avg_monthly_sales) * 100
    
    def _calculate_quarter_seasonality(self, customer_sales: pd.DataFrame) -> float:
        """Calculate quarterly seasonality score."""
        if 'Date' not in customer_sales.columns or 'FinalValue' not in customer_sales.columns:
            return 0
        
        quarterly_sales = customer_sales.groupby(customer_sales['Date'].dt.quarter)['FinalValue'].sum()
        
        if len(quarterly_sales) < 2:
            return 0
        
        # Calculate coefficient of variation for quarterly sales
        cv = quarterly_sales.std() / quarterly_sales.mean() if quarterly_sales.mean() > 0 else 0
        
        return min(100, cv * 50)
    
    def _calculate_weekly_pattern_strength(self, customer_sales: pd.DataFrame) -> float:
        """Calculate weekly pattern strength."""
        if 'Date' not in customer_sales.columns or len(customer_sales) < 7:
            return 0
        
        daily_sales = customer_sales.groupby(customer_sales['Date'].dt.dayofweek)['FinalValue'].sum()
        
        if len(daily_sales) < 2:
            return 0
        
        cv = daily_sales.std() / daily_sales.mean() if daily_sales.mean() > 0 else 0
        
        return min(100, cv * 30)
    
    def _calculate_monthly_pattern_strength(self, customer_sales: pd.DataFrame) -> float:
        """Calculate monthly pattern strength."""
        if 'Date' not in customer_sales.columns or len(customer_sales) < 30:
            return 0
        
        monthly_sales = customer_sales.groupby(customer_sales['Date'].dt.month)['FinalValue'].sum()
        
        if len(monthly_sales) < 2:
            return 0
        
        cv = monthly_sales.std() / monthly_sales.mean() if monthly_sales.mean() > 0 else 0
        
        return min(100, cv * 25)
    
    def _estimate_weather_sensitivity(self, customer_sales: pd.DataFrame) -> float:
        """Estimate weather sensitivity (simplified)."""
        if 'Date' not in customer_sales.columns:
            return 0
        
        # Simple proxy: higher sales variation during monsoon months indicates weather sensitivity
        monsoon_months = [5, 6, 7, 8, 9, 10, 11, 12, 1, 2]
        
        monsoon_sales = customer_sales[customer_sales['Date'].dt.month.isin(monsoon_months)]['FinalValue']
        non_monsoon_sales = customer_sales[~customer_sales['Date'].dt.month.isin(monsoon_months)]['FinalValue']
        
        if len(monsoon_sales) == 0 or len(non_monsoon_sales) == 0:
            return 0
        
        monsoon_cv = monsoon_sales.std() / monsoon_sales.mean() if monsoon_sales.mean() > 0 else 0
        non_monsoon_cv = non_monsoon_sales.std() / non_monsoon_sales.mean() if non_monsoon_sales.mean() > 0 else 0
        
        sensitivity = abs(monsoon_cv - non_monsoon_cv) * 50
        
        return min(100, sensitivity)
    
    def _calculate_po_location_consistency(self, customer_pos: pd.DataFrame) -> float:
        """Calculate PO location consistency."""
        if 'Latitude' not in customer_pos.columns or 'Longitude' not in customer_pos.columns:
            return 1.0  # Assume consistent if no location data
        
        # Calculate distances between PO locations
        locations = customer_pos[['Latitude', 'Longitude']].dropna()
        
        if len(locations) < 2:
            return 1.0
        
        distances = []
        for i, (_, loc1) in enumerate(locations.iterrows()):
            for _, loc2 in locations.iloc[i+1:].iterrows():
                coord1 = (loc1['Latitude'], loc1['Longitude'])
                coord2 = (loc2['Latitude'], loc2['Longitude'])
                distances.append(geodesic(coord1, coord2).kilometers)
        
        if not distances:
            return 1.0
        
        # Lower average distance = higher consistency
        avg_distance = np.mean(distances)
        consistency = max(0, 1 - (avg_distance / 100))  # Normalize by 100km
        
        return consistency
    
    def _calculate_avg_delivery_distance(self, customer_pos: pd.DataFrame) -> float:
        """Calculate average delivery distance."""
        if 'Latitude' not in customer_pos.columns or 'Longitude' not in customer_pos.columns:
            return 0
        
        # Calculate distances from a central depot (using Colombo as reference)
        colombo = (6.9271, 79.8612)
        
        distances = []
        for _, pos in customer_pos.iterrows():
            if pd.notna(pos['Latitude']) and pd.notna(pos['Longitude']):
                po_coords = (pos['Latitude'], pos['Longitude'])
                distance = geodesic(colombo, po_coords).kilometers
                distances.append(distance)
        
        return np.mean(distances) if distances else 0
    
    def normalize_features(self, features_df: pd.DataFrame, 
                          feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Normalize numerical features."""
        if feature_columns is None:
            # Auto-detect numerical columns
            feature_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove ID columns
            feature_columns = [col for col in feature_columns if 'ID' not in col.upper()]
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = features_df.copy()
        
        for col in feature_columns:
            if col in normalized_features.columns:
                normalized_features[f'{col}_normalized'] = scaler.fit_transform(
                    normalized_features[col].values.reshape(-1, 1)
                ).flatten()
                
                # Store scaler for later use
                self.scalers[col] = scaler
        
        return normalized_features
    
    def create_feature_importance_analysis(self, features_df: pd.DataFrame, 
                                         target_column: str) -> Dict[str, float]:
        """Create feature importance analysis."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import mutual_info_regression
        
        # Prepare features
        feature_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in feature_columns if col != target_column and 'ID' not in col.upper()]
        
        X = features_df[feature_columns].fillna(0)
        y = features_df[target_column].fillna(0)
        
        # Random Forest feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        rf_importance = dict(zip(feature_columns, rf.feature_importances_))
        
        # Mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_importance = dict(zip(feature_columns, mi_scores))
        
        # Combine scores
        combined_importance = {}
        for feature in feature_columns:
            combined_importance[feature] = (rf_importance[feature] + mi_importance[feature]) / 2
        
        # Sort by importance
        sorted_importance = dict(sorted(combined_importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def save_feature_metadata(self, features_df: pd.DataFrame, filepath: str):
        """Save feature metadata for documentation."""
        metadata = {
            'total_features': len(features_df.columns),
            'feature_types': {},
            'missing_value_summary': {},
            'feature_categories': {
                'temporal': [col for col in features_df.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'month', 'year', 'season'])],
                'spatial': [col for col in features_df.columns if any(keyword in col.lower() for keyword in ['distance', 'location', 'coord', 'density', 'territory'])],
                'behavioral': [col for col in features_df.columns if any(keyword in col.lower() for keyword in ['behavior', 'pattern', 'loyalty', 'switch'])],
                'performance': [col for col in features_df.columns if any(keyword in col.lower() for keyword in ['revenue', 'growth', 'performance', 'rank'])],
                'interaction': [col for col in features_df.columns if 'interaction' in col.lower()],
            },
            'creation_timestamp': datetime.now().isoformat()
        }
        
        # Feature types
        for col in features_df.columns:
            metadata['feature_types'][col] = str(features_df[col].dtype)
        
        # Missing values
        for col in features_df.columns:
            missing_pct = (features_df[col].isnull().sum() / len(features_df)) * 100
            metadata['missing_value_summary'][col] = f"{missing_pct:.2f}%"
        
        # Save as JSON
        import json
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Feature metadata saved to {filepath}")

# Usage Example:
def main():
    """
    Example usage of the FeatureEngineer class
    """
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Load your data (replace with actual data loading)
    # customer_df = pd.read_csv('customer.csv')
    # sales_df = pd.read_excel('SFA_Orders.xlsx', sheet_name='Jan')
    # gps_df = pd.read_csv('SFA_GPSData.csv')
    # po_df = pd.read_csv('SFA_PO.csv')  # Optional
    
    # Create comprehensive features
    # features_df = feature_engineer.create_comprehensive_features(
    #     customer_df=customer_df,
    #     sales_df=sales_df,
    #     gps_df=gps_df,
    #     po_df=po_df
    # )
    
    # Normalize features
    # normalized_features = feature_engineer.normalize_features(features_df)
    
    # Analyze feature importance (if you have a target variable)
    # importance_scores = feature_engineer.create_feature_importance_analysis(
    #     features_df, target_column='total_revenue'
    # )
    
    # Save feature metadata
    # feature_engineer.save_feature_metadata(features_df, 'feature_metadata.json')
    
    print("Feature engineering pipeline completed!")

if __name__ == "__main__":
    main()