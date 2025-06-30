import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
import xgboost as xgb
from datetime import datetime, timedelta
import joblib
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DealerPerformanceClassifier:
    """
    Advanced dealer performance classification system
    Classifies dealers into performance tiers: Top, High, Medium, Low, Underperforming
    Uses sales data, GPS tracking, and behavioral patterns
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.performance_categories = ['Underperforming', 'Low', 'Medium', 'High', 'Top']
        self.kmeans_model = None
        
    def prepare_dealer_features(self, sales_df: pd.DataFrame, gps_df: pd.DataFrame, 
                              customer_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create comprehensive dealer performance features
        """
        logger.info("Preparing dealer performance features...")
        
        # Convert date columns
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        sales_df['CreationDate'] = pd.to_datetime(sales_df['CreationDate'])
        sales_df['SubmittedDate'] = pd.to_datetime(sales_df['SubmittedDate'])
        
        current_date = sales_df['Date'].max()
        
        # Sales Performance Features
        sales_features = sales_df.groupby('UserCode').agg({
            'FinalValue': ['sum', 'mean', 'std', 'count', 'min', 'max'],
            'Date': ['min', 'max'],
            'DistributorCode': 'nunique',  # Number of unique customers served
            'Code': 'nunique'  # Number of unique orders
        }).reset_index()
        
        # Flatten column names
        sales_features.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                for col in sales_features.columns.values]
        sales_features.rename(columns={'UserCode_': 'UserCode'}, inplace=True)
        
        # Calculate derived sales metrics
        sales_features['total_revenue'] = sales_features['FinalValue_sum']
        sales_features['avg_order_value'] = sales_features['FinalValue_mean']
        sales_features['revenue_consistency'] = 1 - (sales_features['FinalValue_std'] / (sales_features['FinalValue_mean'] + 1))
        sales_features['customer_base_size'] = sales_features['DistributorCode_nunique']
        sales_features['total_orders'] = sales_features['Code_nunique']
        sales_features['orders_per_customer'] = sales_features['total_orders'] / (sales_features['customer_base_size'] + 1)
        
        # Time-based features
        sales_features['days_active'] = (sales_features['Date_max'] - sales_features['Date_min']).dt.days + 1
        sales_features['daily_avg_revenue'] = sales_features['total_revenue'] / sales_features['days_active']
        sales_features['days_since_last_sale'] = (current_date - sales_features['Date_max']).dt.days
        
        # Monthly performance trends
        monthly_performance = self._calculate_monthly_trends(sales_df)
        sales_features = sales_features.merge(monthly_performance, on='UserCode', how='left')
        
        # GPS-based Activity Features
        gps_df['RecievedDate'] = pd.to_datetime(gps_df['RecievedDate'])
        
        gps_features = gps_df.groupby('UserCode').agg({
            'Latitude': ['std', 'nunique', 'mean'],
            'Longitude': ['std', 'nunique', 'mean'],
            'RecievedDate': ['count', 'min', 'max'],
            'TourCode': 'nunique',
            'DivisionCode': 'nunique'
        }).reset_index()
        
        gps_features.columns = ['_'.join(col).strip() if col[1] else col[0] 
                              for col in gps_features.columns.values]
        gps_features.rename(columns={'UserCode_': 'UserCode'}, inplace=True)
        
        # Calculate mobility and activity metrics
        gps_features['location_coverage'] = gps_features['Latitude_std'] + gps_features['Longitude_std']
        gps_features['unique_locations'] = gps_features['Latitude_nunique'] * gps_features['Longitude_nunique']
        gps_features['tracking_days'] = (gps_features['RecievedDate_max'] - gps_features['RecievedDate_min']).dt.days + 1
        gps_features['daily_activity_points'] = gps_features['RecievedDate_count'] / gps_features['tracking_days']
        gps_features['tour_diversity'] = gps_features['TourCode_nunique']
        gps_features['division_coverage'] = gps_features['DivisionCode_nunique']
        
        # Calculate work pattern features
        gps_work_patterns = self._calculate_work_patterns(gps_df)
        gps_features = gps_features.merge(gps_work_patterns, on='UserCode', how='left')
        
        # Merge sales and GPS features
        dealer_features = sales_features.merge(gps_features, on='UserCode', how='outer')
        
        # Customer relationship features
        if customer_df is not None:
            customer_relationship_features = self._calculate_customer_relationships(sales_df, customer_df)
            dealer_features = dealer_features.merge(customer_relationship_features, on='UserCode', how='left')
        
        # Performance efficiency metrics
        dealer_features['revenue_per_location'] = dealer_features['total_revenue'] / (dealer_features['unique_locations'] + 1)
        dealer_features['revenue_per_activity_point'] = dealer_features['total_revenue'] / (dealer_features['RecievedDate_count'] + 1)
        dealer_features['customer_acquisition_rate'] = dealer_features['customer_base_size'] / dealer_features['days_active']
        
        # Territory efficiency
        dealer_features['territory_efficiency'] = (
            dealer_features['total_revenue'] / 
            (dealer_features['location_coverage'] + 1)
        )
        
        # Fill missing values
        numeric_columns = dealer_features.select_dtypes(include=[np.number]).columns
        dealer_features[numeric_columns] = dealer_features[numeric_columns].fillna(0)
        
        logger.info(f"Created {len(dealer_features.columns)} features for {len(dealer_features)} dealers")
        return dealer_features
    
    def _calculate_monthly_trends(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly performance trends for each dealer"""
        sales_monthly = sales_df.copy()
        sales_monthly['year_month'] = sales_monthly['Date'].dt.to_period('M')
        
        monthly_sales = sales_monthly.groupby(['UserCode', 'year_month']).agg({
            'FinalValue': 'sum',
            'DistributorCode': 'nunique'
        }).reset_index()
        
        trend_features = []
        for user_code in monthly_sales['UserCode'].unique():
            user_monthly = monthly_sales[monthly_sales['UserCode'] == user_code].copy()
            user_monthly = user_monthly.sort_values('year_month')
            
            if len(user_monthly) >= 2:
                # Revenue trend
                x = np.arange(len(user_monthly))
                revenue_trend = np.polyfit(x, user_monthly['FinalValue'], 1)[0] if len(x) > 1 else 0
                
                # Customer growth trend
                customer_trend = np.polyfit(x, user_monthly['DistributorCode'], 1)[0] if len(x) > 1 else 0
                
                # Recent vs historical performance
                recent_months = user_monthly.tail(3)
                historical_months = user_monthly.head(-3) if len(user_monthly) > 3 else user_monthly
                
                recent_avg_revenue = recent_months['FinalValue'].mean()
                historical_avg_revenue = historical_months['FinalValue'].mean()
                performance_trend = recent_avg_revenue / (historical_avg_revenue + 1)
                
                # Consistency metrics
                revenue_cv = user_monthly['FinalValue'].std() / (user_monthly['FinalValue'].mean() + 1)
                
                trend_features.append({
                    'UserCode': user_code,
                    'revenue_trend_slope': revenue_trend,
                    'customer_growth_trend': customer_trend,
                    'performance_momentum': performance_trend,
                    'revenue_consistency_cv': revenue_cv,
                    'active_months': len(user_monthly)
                })
        
        return pd.DataFrame(trend_features)
    
    def _calculate_work_patterns(self, gps_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate work pattern features from GPS data"""
        gps_df['hour'] = gps_df['RecievedDate'].dt.hour
        gps_df['day_of_week'] = gps_df['RecievedDate'].dt.dayofweek
        gps_df['date'] = gps_df['RecievedDate'].dt.date
        
        work_patterns = []
        for user_code in gps_df['UserCode'].unique():
            user_gps = gps_df[gps_df['UserCode'] == user_code]
            
            # Working hours analysis
            hourly_activity = user_gps['hour'].value_counts()
            peak_hours = hourly_activity.nlargest(3).index.tolist()
            work_hours_span = hourly_activity.index.max() - hourly_activity.index.min()
            
            # Weekly pattern analysis
            weekly_activity = user_gps['day_of_week'].value_counts()
            active_days_per_week = len(weekly_activity)
            
            # Daily consistency
            daily_activity = user_gps.groupby('date').size()
            daily_consistency = 1 - (daily_activity.std() / (daily_activity.mean() + 1))
            
            # Distance covered (simplified using lat/lng variance)
            total_distance_proxy = (
                user_gps['Latitude'].std() + user_gps['Longitude'].std()
            ) * len(user_gps)
            
            work_patterns.append({
                'UserCode': user_code,
                'work_hours_span': work_hours_span,
                'active_days_per_week': active_days_per_week,
                'daily_consistency': daily_consistency,
                'total_distance_proxy': total_distance_proxy,
                'avg_daily_activity': len(user_gps) / len(daily_activity)
            })
        
        return pd.DataFrame(work_patterns)
    
    def _calculate_customer_relationships(self, sales_df: pd.DataFrame, 
                                        customer_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate customer relationship quality metrics"""
        # Merge sales with customer data
        sales_customer = sales_df.merge(
            customer_df.rename(columns={'No.': 'DistributorCode'}),
            on='DistributorCode', how='left'
        )
        
        relationship_features = []
        for user_code in sales_customer['UserCode'].unique():
            user_sales = sales_customer[sales_customer['UserCode'] == user_code]
            
            # Customer loyalty metrics
            customer_repeat_rate = (
                user_sales.groupby('DistributorCode').size() > 1
            ).mean()
            
            # Average order frequency per customer
            customer_order_freq = user_sales.groupby('DistributorCode').size().mean()
            
            # Customer value distribution
            customer_values = user_sales.groupby('DistributorCode')['FinalValue'].sum()
            customer_value_cv = customer_values.std() / (customer_values.mean() + 1)
            
            # Geographic customer spread
            unique_cities = user_sales['City'].nunique() if 'City' in user_sales.columns else 1
            
            # Customer portfolio quality
            top_customers_revenue = customer_values.nlargest(5).sum()
            total_revenue = customer_values.sum()
            top_customer_dependency = top_customers_revenue / (total_revenue + 1)
            
            relationship_features.append({
                'UserCode': user_code,
                'customer_repeat_rate': customer_repeat_rate,
                'avg_customer_order_freq': customer_order_freq,
                'customer_value_diversity': 1 - customer_value_cv,
                'geographic_customer_spread': unique_cities,
                'top_customer_dependency': top_customer_dependency
            })
        
        return pd.DataFrame(relationship_features)
    
    def create_performance_labels(self, dealer_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create performance labels using clustering and business rules
        """
        logger.info("Creating performance labels...")
        
        # Key performance indicators for labeling
        performance_metrics = [
            'total_revenue', 'customer_base_size', 'revenue_consistency',
            'daily_avg_revenue', 'territory_efficiency', 'customer_repeat_rate',
            'performance_momentum', 'daily_activity_points'
        ]
        
        # Prepare data for clustering
        cluster_data = dealer_features[performance_metrics].fillna(0)
        
        # Normalize data for clustering
        scaler = MinMaxScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Use K-means clustering to identify performance groups
        self.kmeans_model = KMeans(n_clusters=5, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(cluster_data_scaled)
        
        # Create performance scores
        dealer_features['cluster'] = cluster_labels
        
        # Calculate composite performance score
        weights = {
            'total_revenue': 0.25,
            'customer_base_size': 0.15,
            'revenue_consistency': 0.15,
            'daily_avg_revenue': 0.20,
            'territory_efficiency': 0.10,
            'customer_repeat_rate': 0.10,
            'performance_momentum': 0.05
        }
        
        # Normalize metrics to 0-1 scale
        for metric in weights.keys():
            if metric in dealer_features.columns:
                max_val = dealer_features[metric].max()
                min_val = dealer_features[metric].min()
                if max_val > min_val:
                    dealer_features[f'{metric}_normalized'] = (
                        (dealer_features[metric] - min_val) / (max_val - min_val)
                    )
                else:
                    dealer_features[f'{metric}_normalized'] = 0
        
        # Calculate weighted performance score
        dealer_features['performance_score'] = 0
        for metric, weight in weights.items():
            if f'{metric}_normalized' in dealer_features.columns:
                dealer_features['performance_score'] += (
                    dealer_features[f'{metric}_normalized'] * weight
                )
        
        # Create performance labels based on percentiles
        performance_percentiles = dealer_features['performance_score'].quantile([0.2, 0.4, 0.6, 0.8])
        
        def assign_performance_category(score):
            if score <= performance_percentiles[0.2]:
                return 'Underperforming'
            elif score <= performance_percentiles[0.4]:
                return 'Low'
            elif score <= performance_percentiles[0.6]:
                return 'Medium'
            elif score <= performance_percentiles[0.8]:
                return 'High'
            else:
                return 'Top'
        
        dealer_features['performance_category'] = dealer_features['performance_score'].apply(
            assign_performance_category
        )
        
        logger.info(f"Performance category distribution:")
        logger.info(dealer_features['performance_category'].value_counts())
        
        return dealer_features
    
    def train_classifier(self, dealer_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the dealer performance classifier
        """
        logger.info("Training dealer performance classifier...")
        
        # Prepare features and target
        feature_columns = [col for col in dealer_features.columns 
                          if col not in ['UserCode', 'performance_category', 'cluster', 'performance_score']
                          and not col.endswith('_normalized')]
        
        X = dealer_features[feature_columns].fillna(0)
        y = dealer_features['performance_category']
        
        # Encode target labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoders['performance_category'] = le
        
        # Store feature names
        self.feature_names = feature_columns
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100, random_state=42, eval_metric='mlogloss'
            )
        }
        
        model_results = {}
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            if name == 'xgboost':
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train_scaled, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            # Test predictions
            y_pred = model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            model_results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_accuracy,
                'classification_report': classification_report(
                    y_test, y_pred, target_names=le.classes_
                )
            }
            
            logger.info(f"{name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            logger.info(f"{name} - Test Accuracy: {test_accuracy:.4f}")
            
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                best_model_name = name
        
        # Select best model
        self.model = model_results[best_model_name]['model']
        logger.info(f"Best model: {best_model_name}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 most important features:")
            logger.info(feature_importance.head(10))
        
        return model_results
    
    def predict_performance(self, dealer_features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict dealer performance categories
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_classifier first.")
        
        # Prepare features
        X = dealer_features[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        prediction_proba = self.model.predict_proba(X_scaled)
        
        # Decode predictions
        le = self.label_encoders['performance_category']
        predicted_categories = le.inverse_transform(predictions)
        
        # Create results dataframe
        results = dealer_features[['UserCode']].copy()
        results['predicted_performance'] = predicted_categories
        results['prediction_confidence'] = prediction_proba.max(axis=1)
        
        # Add probability for each class
        for i, class_name in enumerate(le.classes_):
            results[f'prob_{class_name}'] = prediction_proba[:, i]
        
        return results
    
    def get_performance_insights(self, dealer_features: pd.DataFrame, 
                               predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate insights about dealer performance
        """
        insights = {}
        
        # Performance distribution
        performance_dist = predictions['predicted_performance'].value_counts()
        insights['performance_distribution'] = performance_dist.to_dict()
        
        # Top and bottom performers
        high_performers = predictions[
            predictions['predicted_performance'].isin(['Top', 'High'])
        ]['UserCode'].tolist()
        
        low_performers = predictions[
            predictions['predicted_performance'].isin(['Underperforming', 'Low'])
        ]['UserCode'].tolist()
        
        insights['high_performers'] = high_performers
        insights['low_performers'] = low_performers
        
        # Average metrics by performance category
        merged_data = dealer_features.merge(predictions, on='UserCode')
        
        key_metrics = ['total_revenue', 'customer_base_size', 'daily_avg_revenue',
                      'territory_efficiency', 'customer_repeat_rate']
        
        performance_metrics = {}
        for category in self.performance_categories:
            category_data = merged_data[merged_data['predicted_performance'] == category]
            if len(category_data) > 0:
                performance_metrics[category] = {
                    metric: category_data[metric].mean() 
                    for metric in key_metrics if metric in category_data.columns
                }
        
        insights['performance_metrics_by_category'] = performance_metrics
        
        return insights
    
    def save_model(self, filepath: str):
        """Save the trained model and preprocessing components"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'kmeans_model': self.kmeans_model,
            'performance_categories': self.performance_categories
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and preprocessing components"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.kmeans_model = model_data['kmeans_model']
        self.performance_categories = model_data['performance_categories']
        logger.info(f"Model loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize classifier
    classifier = DealerPerformanceClassifier()
    
    # Example of how to use with your data
    # Note: Replace with actual data loading
    """
    # Load your datasets
    sales_df = pd.read_excel('SFA_Orders.xlsx', sheet_name='Jan')  # Combine all months
    gps_df = pd.read_csv('SFA_GPSData_202_2023January.csv')
    customer_df = pd.read_excel('Customer.xlsx')
    
    # Prepare features
    dealer_features = classifier.prepare_dealer_features(sales_df, gps_df, customer_df)
    
    # Create performance labels
    dealer_features_labeled = classifier.create_performance_labels(dealer_features)
    
    # Train classifier
    model_results = classifier.train_classifier(dealer_features_labeled)
    
    # Make predictions
    predictions = classifier.predict_performance(dealer_features_labeled)
    
    # Get insights
    insights = classifier.get_performance_insights(dealer_features_labeled, predictions)
    
    # Save model
    classifier.save_model('dealer_performance_model.pkl')
    
    print("Dealer Performance Classification completed!")
    print(f"Performance Distribution: {insights['performance_distribution']}")
    """