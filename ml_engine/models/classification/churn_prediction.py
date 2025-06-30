'''
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
import joblib
import json
from pathlib import Path
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class ChurnConfig:
    """Configuration for churn prediction"""
    test_size: float = 0.2
    random_state: int = 42
    model_save_path: str = "models/churn/"
    churn_threshold_days: int = 90  # Days without purchase to consider churned
    min_historical_days: int = 180  # Minimum historical data required
    cross_val_folds: int = 5
    enable_feature_selection: bool = True
    balance_dataset: bool = True
    use_ensemble: bool = True

class ChurnPredictor:
    """
    Advanced customer churn prediction using multiple ML algorithms
    """
    
    def __init__(self, config: ChurnConfig = None):
        self.config = config or ChurnConfig()
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.feature_names = []
        self.feature_importance = {}
        self.is_fitted = False
        self.best_model_name = None
        self.performance_metrics = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create model directory
        Path(self.config.model_save_path).mkdir(parents=True, exist_ok=True)
    
    def prepare_churn_features(self, sales_data: pd.DataFrame, 
                             customer_data: pd.DataFrame = None,
                             gps_data: pd.DataFrame = None,
                             reference_date: datetime = None) -> pd.DataFrame:
        """
        Prepare comprehensive features for churn prediction
        """
        self.logger.info("Preparing churn prediction features...")
        
        if reference_date is None:
            reference_date = pd.to_datetime(sales_data['Date'].max())
        
        # Ensure proper datetime format
        sales_data['Date'] = pd.to_datetime(sales_data['Date'])
        
        # Filter data to have minimum historical period
        min_date = reference_date - timedelta(days=self.config.min_historical_days)
        historical_data = sales_data[sales_data['Date'] >= min_date].copy()
        
        # Calculate churn labels
        churn_labels = self._calculate_churn_labels(historical_data, reference_date)
        
        # Calculate behavioral features
        behavioral_features = self._calculate_behavioral_features(historical_data, reference_date)
        
        # Calculate RFM features
        rfm_features = self._calculate_rfm_features(historical_data, reference_date)
        
        # Calculate trend features
        trend_features = self._calculate_trend_features(historical_data, reference_date)
        
        # Calculate engagement features
        engagement_features = self._calculate_engagement_features(historical_data, reference_date)
        
        # Calculate product affinity features
        product_features = self._calculate_product_affinity_features(historical_data)
        
        # Combine all features
        feature_dfs = [behavioral_features, rfm_features, trend_features, 
                      engagement_features, product_features]
        
        churn_features = churn_labels
        for df in feature_dfs:
            churn_features = churn_features.merge(df, left_index=True, right_index=True, how='left')
        
        # Add geographical features if available
        if customer_data is not None:
            geo_features = self._calculate_geographical_features(customer_data)
            churn_features = churn_features.merge(geo_features, left_index=True, right_index=True, how='left')
        
        # Add dealer interaction features if GPS data available
        if gps_data is not None:
            interaction_features = self._calculate_interaction_features(gps_data, historical_data)
            churn_features = churn_features.merge(interaction_features, left_index=True, right_index=True, how='left')
        
        # Fill missing values
        numeric_columns = churn_features.select_dtypes(include=[np.number]).columns
        churn_features[numeric_columns] = churn_features[numeric_columns].fillna(churn_features[numeric_columns].median())
        
        # Fill categorical missing values
        categorical_columns = churn_features.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'churned':
                churn_features[col] = churn_features[col].fillna('Unknown')
        
        self.feature_names = [col for col in churn_features.columns if col != 'churned']
        self.logger.info(f"Created {len(self.feature_names)} features for {len(churn_features)} customers")
        
        return churn_features
    
    def _calculate_churn_labels(self, sales_data: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
        """Calculate churn labels based on recency"""
        
        customer_col = 'DistributorCode' if 'DistributorCode' in sales_data.columns else 'CustomerID'
        
        # Calculate days since last purchase
        last_purchase = sales_data.groupby(customer_col)['Date'].max().reset_index()
        last_purchase['days_since_last_purchase'] = (reference_date - last_purchase['Date']).dt.days
        
        # Define churn based on threshold
        last_purchase['churned'] = (last_purchase['days_since_last_purchase'] > self.config.churn_threshold_days).astype(int)
        
        # Set customer as index
        churn_labels = last_purchase.set_index(customer_col)[['churned', 'days_since_last_purchase']]
        
        self.logger.info(f"Churn rate: {churn_labels['churned'].mean():.2%}")
        
        return churn_labels
    
    def _calculate_behavioral_features(self, sales_data: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
        """Calculate customer behavioral features"""
        
        customer_col = 'DistributorCode' if 'DistributorCode' in sales_data.columns else 'CustomerID'
        value_col = 'FinalValue' if 'FinalValue' in sales_data.columns else 'Total'
        
        # Basic transaction statistics
        behavioral = sales_data.groupby(customer_col).agg({
            value_col: ['count', 'sum', 'mean', 'std', 'min', 'max'],
            'Date': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        behavioral.columns = ['_'.join(col) for col in behavioral.columns]
        behavioral.columns = [
            'transaction_count', 'total_spent', 'avg_transaction_value', 'transaction_std',
            'min_transaction', 'max_transaction', 'first_purchase_date', 'last_purchase_date'
        ]
        
        # Calculate derived metrics
        behavioral['customer_age_days'] = (reference_date - behavioral['first_purchase_date']).dt.days
        behavioral['days_since_last_purchase'] = (reference_date - behavioral['last_purchase_date']).dt.days
        behavioral['purchase_frequency'] = behavioral['transaction_count'] / (behavioral['customer_age_days'] + 1)
        behavioral['avg_days_between_purchases'] = behavioral['customer_age_days'] / (behavioral['transaction_count'] + 1)
        
        # Transaction consistency metrics
        behavioral['transaction_cv'] = behavioral['transaction_std'] / (behavioral['avg_transaction_value'] + 1)
        behavioral['spending_velocity'] = behavioral['total_spent'] / (behavioral['customer_age_days'] + 1)
        
        # Drop date columns
        behavioral = behavioral.drop(['first_purchase_date', 'last_purchase_date'], axis=1)
        
        return behavioral
    
    def _calculate_rfm_features(self, sales_data: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
        """Calculate RFM-based features"""
        
        customer_col = 'DistributorCode' if 'DistributorCode' in sales_data.columns else 'CustomerID'
        value_col = 'FinalValue' if 'FinalValue' in sales_data.columns else 'Total'
        
        rfm = sales_data.groupby(customer_col).agg({
            'Date': lambda x: (reference_date - x.max()).days,
            value_col: ['count', 'sum']
        })
        
        rfm.columns = ['recency_days', 'frequency', 'monetary_total']
        
        # Calculate RFM scores (1-5 scale)
        rfm['recency_score'] = pd.qcut(rfm['recency_days'], 5, labels=[5,4,3,2,1], duplicates='drop')
        rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
        rfm['monetary_score'] = pd.qcut(rfm['monetary_total'], 5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Convert to numeric
        rfm['recency_score'] = pd.to_numeric(rfm['recency_score'])
        rfm['frequency_score'] = pd.to_numeric(rfm['frequency_score'])
        rfm['monetary_score'] = pd.to_numeric(rfm['monetary_score'])
        
        # Combined RFM score
        rfm['rfm_score'] = rfm['recency_score'] + rfm['frequency_score'] + rfm['monetary_score']
        
        # RFM ratios
        rfm['frequency_monetary_ratio'] = rfm['frequency'] / (rfm['monetary_total'] + 1)
        rfm['recency_frequency_ratio'] = rfm['recency_days'] / (rfm['frequency'] + 1)
        
        return rfm
    
    def _calculate_trend_features(self, sales_data: pd.DataFrame, reference_date: datetime, 
                                window_days: int = 30) -> pd.DataFrame:
        """Calculate trend-based features"""
        
        customer_col = 'DistributorCode' if 'DistributorCode' in sales_data.columns else 'CustomerID'
        value_col = 'FinalValue' if 'FinalValue' in sales_data.columns else 'Total'
        
        # Create time windows
        current_window_start = reference_date - timedelta(days=window_days)
        previous_window_start = reference_date - timedelta(days=2*window_days)
        
        # Current period data
        current_data = sales_data[sales_data['Date'] >= current_window_start]
        current_stats = current_data.groupby(customer_col).agg({
            value_col: ['count', 'sum'],
            'Date': 'nunique'
        })
        current_stats.columns = ['current_transactions', 'current_spending', 'current_active_days']
        
        # Previous period data
        previous_data = sales_data[(sales_data['Date'] >= previous_window_start) & 
                                 (sales_data['Date'] < current_window_start)]
        previous_stats = previous_data.groupby(customer_col).agg({
            value_col: ['count', 'sum'],
            'Date': 'nunique'
        })
        previous_stats.columns = ['previous_transactions', 'previous_spending', 'previous_active_days']
        
        # Combine and calculate trends
        trend_features = current_stats.merge(previous_stats, left_index=True, right_index=True, how='outer')
        trend_features = trend_features.fillna(0)
        
        # Calculate trend ratios
        trend_features['transaction_trend'] = (
            trend_features['current_transactions'] / (trend_features['previous_transactions'] + 1)
        )
        trend_features['spending_trend'] = (
            trend_features['current_spending'] / (trend_features['previous_spending'] + 1)
        )
        trend_features['activity_trend'] = (
            trend_features['current_active_days'] / (trend_features['previous_active_days'] + 1)
        )
        
        # Calculate decline indicators
        trend_features['is_declining_transactions'] = (trend_features['transaction_trend'] < 0.5).astype(int)
        trend_features['is_declining_spending'] = (trend_features['spending_trend'] < 0.5).astype(int)
        trend_features['is_declining_activity'] = (trend_features['activity_trend'] < 0.5).astype(int)
        
        return trend_features
    
    def _calculate_engagement_features(self, sales_data: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
        """Calculate customer engagement features"""
        
        customer_col = 'DistributorCode' if 'DistributorCode' in sales_data
'''





import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
from datetime import datetime, timedelta
import joblib
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnPredictor:
    """
    Advanced customer churn prediction using Random Forest and XGBoost
    Analyzes sales patterns, GPS data, and customer behavior to predict churn risk
    """
    
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.churn_threshold_days = 60  # Days without order to consider potential churn
        
    def prepare_features(self, sales_df: pd.DataFrame, customer_df: pd.DataFrame, 
                        gps_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create comprehensive features for churn prediction
        """
        logger.info("Preparing features for churn prediction...")
        
        # Convert date columns
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        sales_df['CreationDate'] = pd.to_datetime(sales_df['CreationDate'])
        
        # Calculate current date for recency calculations
        current_date = sales_df['Date'].max()
        
        # Aggregate sales features by customer
        customer_features = sales_df.groupby('DistributorCode').agg({
            'FinalValue': ['sum', 'mean', 'std', 'count', 'min', 'max'],
            'Date': ['min', 'max'],
            'Code': 'nunique'  # Number of unique orders
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                   for col in customer_features.columns.values]
        customer_features.rename(columns={'DistributorCode_': 'DistributorCode'}, inplace=True)
        
        # Calculate recency features
        customer_features['days_since_last_order'] = (current_date - customer_features['Date_max']).dt.days
        customer_features['days_since_first_order'] = (current_date - customer_features['Date_min']).dt.days
        customer_features['customer_lifetime_days'] = (customer_features['Date_max'] - customer_features['Date_min']).dt.days
        
        # Calculate frequency and monetary features
        customer_features['avg_order_value'] = customer_features['FinalValue_sum'] / customer_features['FinalValue_count']
        customer_features['order_frequency'] = customer_features['FinalValue_count'] / (customer_features['customer_lifetime_days'] + 1)
        customer_features['revenue_consistency'] = customer_features['FinalValue_std'] / (customer_features['FinalValue_mean'] + 1)
        
        # Time-based features
        sales_monthly = sales_df.copy()
        sales_monthly['year_month'] = sales_monthly['Date'].dt.to_period('M')
        monthly_sales = sales_monthly.groupby(['DistributorCode', 'year_month'])['FinalValue'].sum().reset_index()
        
        # Calculate trend features
        trend_features = []
        for customer_id in monthly_sales['DistributorCode'].unique():
            customer_monthly = monthly_sales[monthly_sales['DistributorCode'] == customer_id].copy()
            customer_monthly = customer_monthly.sort_values('year_month')
            
            if len(customer_monthly) >= 3:
                # Calculate trend using linear regression slope
                x = np.arange(len(customer_monthly))
                y = customer_monthly['FinalValue'].values
                trend_slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
                
                # Calculate seasonality (coefficient of variation)
                seasonality = customer_monthly['FinalValue'].std() / (customer_monthly['FinalValue'].mean() + 1)
                
                # Recent vs historical performance
                recent_avg = customer_monthly.tail(3)['FinalValue'].mean()
                historical_avg = customer_monthly.head(-3)['FinalValue'].mean() if len(customer_monthly) > 3 else recent_avg
                performance_ratio = recent_avg / (historical_avg + 1)
                
                trend_features.append({
                    'DistributorCode': customer_id,
                    'sales_trend_slope': trend_slope,
                    'sales_seasonality': seasonality,
                    'recent_vs_historical': performance_ratio,
                    'active_months': len(customer_monthly)
                })
        
        trend_df = pd.DataFrame(trend_features)
        
        # Merge all features
        final_features = customer_features.merge(trend_df, on='DistributorCode', how='left')
        
        # Add GPS-based features if available
        if gps_df is not None:
            gps_features = self._create_gps_features(gps_df)
            final_features = final_features.merge(gps_features, on='DistributorCode', how='left')
        
        # Add customer demographic features
        if 'City' in customer_df.columns:
            city_encoder = LabelEncoder()
            customer_df['city_encoded'] = city_encoder.fit_transform(customer_df['City'].fillna('Unknown'))
            self.label_encoders['city'] = city_encoder
            
            final_features = final_features.merge(
                customer_df[['No.', 'city_encoded']].rename(columns={'No.': 'DistributorCode'}),
                on='DistributorCode', how='left'
            )
        
        # Create churn target variable
        final_features['is_churned'] = (final_features['days_since_last_order'] > self.churn_threshold_days).astype(int)
        
        # Fill missing values
        numeric_columns = final_features.select_dtypes(include=[np.number]).columns
        final_features[numeric_columns] = final_features[numeric_columns].fillna(0)
        
        # Create risk categories
        final_features['churn_risk_category'] = pd.cut(
            final_features['days_since_last_order'],
            bins=[0, 15, 30, 45, 60, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        logger.info(f"Created {len(final_features.columns)} features for {len(final_features)} customers")
        return final_features
    
    def _create_gps_features(self, gps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create GPS-based behavioral features
        """
        gps_df['RecievedDate'] = pd.to_datetime(gps_df['RecievedDate'])
        
        gps_features = gps_df.groupby('UserCode').agg({
            'Latitude': ['std', 'nunique'],
            'Longitude': ['std', 'nunique'],
            'RecievedDate': ['count', 'min', 'max'],
            'TourCode': 'nunique'
        }).reset_index()
        
        gps_features.columns = ['_'.join(col).strip() if col[1] else col[0] 
                               for col in gps_features.columns.values]
        gps_features.rename(columns={'UserCode_': 'DistributorCode'}, inplace=True)
        
        # Calculate mobility patterns
        gps_features['location_variance'] = gps_features['Latitude_std'] + gps_features['Longitude_std']
        gps_features['unique_locations'] = gps_features['Latitude_nunique'] * gps_features['Longitude_nunique']
        gps_features['tracking_days'] = (gps_features['RecievedDate_max'] - gps_features['RecievedDate_min']).dt.days
        gps_features['avg_daily_pings'] = gps_features['RecievedDate_count'] / (gps_features['tracking_days'] + 1)
        
        return gps_features
    
    def train_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train both Random Forest and XGBoost models
        """
        logger.info("Training churn prediction models...")
        
        # Prepare features and target
        feature_columns = [col for col in features_df.columns 
                          if col not in ['DistributorCode', 'is_churned', 'churn_risk_category', 
                                       'Date_min', 'Date_max']]
        
        X = features_df[feature_columns].copy()
        y = features_df['is_churned'].copy()
        
        # Handle any remaining non-numeric columns
        for col in X.columns:
            if X[col].dtype == 'object':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        self.feature_names = feature_columns
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # Train Random Forest
        logger.info("Training Random Forest model...")
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 4, 6],
            'class_weight': ['balanced']
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params, cv=5, scoring='roc_auc', n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        self.rf_model = rf_grid.best_estimator_
        
        # Train XGBoost
        logger.info("Training XGBoost model...")
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            xgb_params, cv=5, scoring='roc_auc', n_jobs=-1
        )
        xgb_grid.fit(X_train_scaled, y_train)
        self.xgb_model = xgb_grid.best_estimator_
        
        # Evaluate models
        rf_pred = self.rf_model.predict(X_test)
        rf_pred_proba = self.rf_model.predict_proba(X_test)[:, 1]
        
        xgb_pred = self.xgb_model.predict(X_test_scaled)
        xgb_pred_proba = self.xgb_model.predict_proba(X_test_scaled)[:, 1]
        
        results = {
            'rf_accuracy': (rf_pred == y_test).mean(),
            'rf_auc': roc_auc_score(y_test, rf_pred_proba),
            'xgb_accuracy': (xgb_pred == y_test).mean(),
            'xgb_auc': roc_auc_score(y_test, xgb_pred_proba),
            'rf_classification_report': classification_report(y_test, rf_pred),
            'xgb_classification_report': classification_report(y_test, xgb_pred),
            'feature_importance_rf': dict(zip(feature_columns, self.rf_model.feature_importances_)),
            'feature_importance_xgb': dict(zip(feature_columns, self.xgb_model.feature_importances_))
        }
        
        logger.info(f"Random Forest AUC: {results['rf_auc']:.4f}")
        logger.info(f"XGBoost AUC: {results['xgb_auc']:.4f}")
        
        return results
    
    def predict_churn_probability(self, features_df: pd.DataFrame, model_type: str = 'ensemble') -> pd.DataFrame:
        """
        Predict churn probability for customers
        """
        feature_columns = [col for col in features_df.columns 
                          if col in self.feature_names]
        
        X = features_df[feature_columns].copy()
        
        # Handle categorical columns
        for col in X.columns:
            if X[col].dtype == 'object' and col in self.label_encoders:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        X_scaled = self.scaler.transform(X)
        
        if model_type == 'rf':
            churn_prob = self.rf_model.predict_proba(X)[:, 1]
        elif model_type == 'xgb':
            churn_prob = self.xgb_model.predict_proba(X_scaled)[:, 1]
        else:  # ensemble
            rf_prob = self.rf_model.predict_proba(X)[:, 1]
            xgb_prob = self.xgb_model.predict_proba(X_scaled)[:, 1]
            churn_prob = (rf_prob + xgb_prob) / 2
        
        results = features_df[['DistributorCode']].copy()
        results['churn_probability'] = churn_prob
        results['churn_risk'] = pd.cut(
            churn_prob,
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        return results.sort_values('churn_probability', ascending=False)
    
    def get_churn_insights(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate actionable insights about customer churn patterns
        """
        insights = {}
        
        # High-risk customers
        predictions = self.predict_churn_probability(features_df)
        high_risk = predictions[predictions['churn_probability'] > 0.7]
        
        insights['high_risk_customers'] = len(high_risk)
        insights['high_risk_customer_list'] = high_risk['DistributorCode'].tolist()
        
        # Feature importance analysis
        if self.rf_model:
            feature_importance = sorted(
                zip(self.feature_names, self.rf_model.feature_importances_),
                key=lambda x: x[1], reverse=True
            )
            insights['top_churn_indicators'] = feature_importance[:10]
        
        # Risk distribution
        risk_distribution = predictions['churn_risk'].value_counts()
        insights['risk_distribution'] = risk_distribution.to_dict()
        
        return insights
    
    def save_model(self, filepath: str):
        """Save trained models and preprocessing components"""
        model_data = {
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'churn_threshold_days': self.churn_threshold_days
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained models and preprocessing components"""
        model_data = joblib.load(filepath)
        self.rf_model = model_data['rf_model']
        self.xgb_model = model_data['xgb_model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.churn_threshold_days = model_data['churn_threshold_days']
        logger.info(f"Model loaded from {filepath}")

# Usage example
if __name__ == "__main__":
    # Initialize predictor
    churn_predictor = ChurnPredictor()
    
    # Load your data
    # sales_df = pd.read_excel('SFA_Orders.xlsx', sheet_name='Jan')  # Load all sheets
    # customer_df = pd.read_excel('Customer.xlsx')
    # gps_df = pd.read_csv('SFA_GPSData.csv')
    
    # Prepare features
    # features_df = churn_predictor.prepare_features(sales_df, customer_df, gps_df)
    
    # Train models
    # results = churn_predictor.train_models(features_df)
    
    # Make predictions
    # predictions = churn_predictor.predict_churn_probability(features_df)
    
    # Get insights
    # insights = churn_predictor.get_churn_insights(features_df)
    
    # Save model
    # churn_predictor.save_model('churn_prediction_model.pkl')
    
    print("Churn prediction model implementation complete!")