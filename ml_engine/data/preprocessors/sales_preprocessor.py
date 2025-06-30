import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesPreprocessor:
    """
    Advanced sales data preprocessing for geosales intelligence platform.
    Handles normalization, missing values, outliers, and feature engineering.
    """
    
    def __init__(self, 
                 outlier_method: str = 'iqr',
                 missing_strategy: str = 'knn',
                 normalization_method: str = 'robust',
                 seasonality_detection: bool = True):
        
        self.outlier_method = outlier_method
        self.missing_strategy = missing_strategy
        self.normalization_method = normalization_method
        self.seasonality_detection = seasonality_detection
        
        # Initialize scalers and encoders
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
        # Business rules and thresholds
        self.business_rules = {
            'min_order_value': 0,
            'max_order_value': 10000000,  # 10M max order value
            'valid_months': list(range(1, 13)),
            'working_hours': (6, 22),  # 6 AM to 10 PM
        }
    
    def preprocess_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing pipeline for sales data.
        
        Args:
            df: Raw sales dataframe with columns: Code, Date, DistributorCode, 
                UserCode, UserName, FinalValue, CreationDate, SubmittedDate, ERPOrderNumber
        
        Returns:
            Preprocessed sales dataframe with additional features
        """
        logger.info(f"Starting sales preprocessing for {len(df)} records")
        
        # Make a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Step 1: Basic data cleaning
        processed_df = self._clean_basic_data(processed_df)
        
        # Step 2: Date and time processing
        processed_df = self._process_datetime_features(processed_df)
        
        # Step 3: Sales value validation and cleaning
        processed_df = self._clean_sales_values(processed_df)
        
        # Step 4: Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Step 5: Detect and handle outliers
        processed_df = self._handle_outliers(processed_df)
        
        # Step 6: Feature engineering
        processed_df = self._engineer_features(processed_df)
        
        # Step 7: Encode categorical variables
        processed_df = self._encode_categorical_variables(processed_df)
        
        # Step 8: Normalize numerical features
        processed_df = self._normalize_features(processed_df)
        
        # Step 9: Create business intelligence features
        processed_df = self._create_business_features(processed_df)
        
        # Step 10: Quality assessment
        processed_df = self._assess_data_quality(processed_df)
        
        logger.info(f"Sales preprocessing completed. {len(processed_df)} records remaining")
        
        return processed_df
    
    def _clean_basic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning and standardization."""
        
        # Convert date columns to datetime
        date_columns = ['Date', 'CreationDate', 'SubmittedDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=[col for col in date_columns if col in df.columns])
        
        # Convert FinalValue to numeric
        df['FinalValue'] = pd.to_numeric(df['FinalValue'], errors='coerce')
        
        # Remove rows with missing or invalid sales values
        df = df.dropna(subset=['FinalValue'])
        df = df[df['FinalValue'] >= 0]  # Sales values should be non-negative
        
        # Standardize text fields
        text_columns = ['Code', 'DistributorCode', 'UserCode', 'UserName', 'ERPOrderNumber']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        
        # Remove duplicate orders (same code and date)
        df = df.drop_duplicates(subset=['Code', 'Date'])
        
        # Sort by date and user
        df = df.sort_values(['UserCode', 'Date']).reset_index(drop=True)
        
        return df
    
    def _process_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and process datetime features."""
        
        # Extract date components
        for col in ['Date', 'CreationDate', 'SubmittedDate']:
            if col in df.columns:
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6])
        
        # Calculate processing time (difference between creation and submission)
        if 'CreationDate' in df.columns and 'SubmittedDate' in df.columns:
            df['processing_time_minutes'] = (
                df['SubmittedDate'] - df['CreationDate']
            ).dt.total_seconds() / 60
            
            # Flag rush orders (processed very quickly)
            df['is_rush_order'] = df['processing_time_minutes'] < 5
        
        # Create time-based business features
        df['is_business_hours'] = (
            (df['Date_hour'] >= self.business_rules['working_hours'][0]) &
            (df['Date_hour'] <= self.business_rules['working_hours'][1])
        )
        
        # Season detection
        df['season'] = df['Date_month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        # Sri Lankan festival seasons (approximate)
        df['is_festival_season'] = df['Date_month'].isin([4, 12, 1])  # Avurudu, Christmas/New Year
        
        return df
    
    def _clean_sales_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate sales values."""
        
        initial_count = len(df)
        
        # Remove unrealistic sales values
        df = df[
            (df['FinalValue'] >= self.business_rules['min_order_value']) &
            (df['FinalValue'] <= self.business_rules['max_order_value'])
        ]
        
        # Log transform for highly skewed sales data
        df['FinalValue_log'] = np.log1p(df['FinalValue'])
        
        # Create sales value categories
        df['sales_category'] = pd.cut(
            df['FinalValue'],
            bins=[0, 1000, 5000, 20000, 50000, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Calculate rolling statistics for trend analysis
        df['sales_7d_avg'] = df.groupby('UserCode')['FinalValue'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        df['sales_30d_avg'] = df.groupby('UserCode')['FinalValue'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        
        logger.info(f"Sales value cleaning: {initial_count - len(df)} records removed")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using various strategies."""
        
        # Identify columns with missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if not missing_cols:
            return df
        
        logger.info(f"Handling missing values in columns: {missing_cols}")
        
        # Strategy 1: Simple imputation for categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_value)
        
        # Strategy 2: KNN imputation for numerical variables
        if self.missing_strategy == 'knn':
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            missing_numerical = [col for col in numerical_cols if df[col].isnull().sum() > 0]
            
            if missing_numerical:
                imputer = KNNImputer(n_neighbors=5, weights='distance')
                df[missing_numerical] = imputer.fit_transform(df[missing_numerical])
                self.imputers['knn_numerical'] = imputer
        
        # Strategy 3: Forward fill for time series data
        time_series_cols = ['FinalValue', 'processing_time_minutes']
        for col in time_series_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col] = df.groupby('UserCode')[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using multiple methods."""
        
        initial_count = len(df)
        
        # Columns to check for outliers
        outlier_cols = ['FinalValue', 'processing_time_minutes']
        outlier_flags = pd.DataFrame(False, index=df.index, columns=outlier_cols)
        
        for col in outlier_cols:
            if col not in df.columns:
                continue
            
            if self.outlier_method == 'iqr':
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_flags[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif self.outlier_method == 'zscore':
                # Z-score method
                z_scores = np.abs(stats.zscore(df[col]))
                outlier_flags[col] = z_scores > 3
            
            elif self.outlier_method == 'isolation':
                # Isolation Forest (simplified version)
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_pred = iso_forest.fit_predict(df[[col]])
                outlier_flags[col] = outlier_pred == -1
        
        # Create composite outlier flag
        df['is_outlier'] = outlier_flags.any(axis=1)
        
        # Option 1: Remove outliers
        # df_clean = df[~df['is_outlier']].copy()
        
        # Option 2: Cap outliers (more conservative approach)
        df_clean = df.copy()
        for col in outlier_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.05)
                Q99 = df[col].quantile(0.95)
                df_clean[col] = df_clean[col].clip(lower=Q1, upper=Q99)
        
        logger.info(f"Outlier handling: {df['is_outlier'].sum()} outliers detected")
        
        return df_clean
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features for better model performance."""
        
        # Sales performance metrics
        df['sales_per_hour'] = df['FinalValue'] / (df['Date_hour'] + 1)
        
        # Customer/Dealer relationship features
        df['customer_dealer_combo'] = df['DistributorCode'] + '_' + df['UserCode']
        
        # Recency, Frequency, Monetary (RFM) features
        reference_date = df['Date'].max()
        
        rfm_features = df.groupby('DistributorCode').agg({
            'Date': lambda x: (reference_date - x.max()).days,  # Recency
            'Code': 'count',  # Frequency
            'FinalValue': 'sum'  # Monetary
        }).rename(columns={
            'Date': 'recency_days',
            'Code': 'frequency_orders',
            'FinalValue': 'monetary_total'
        })
        
        # Merge RFM features back
        df = df.merge(rfm_features, left_on='DistributorCode', right_index=True, how='left')
        
        # Sales trend features
        df['sales_trend_7d'] = df.groupby('UserCode')['FinalValue'].transform(
            lambda x: x.rolling(window=7, min_periods=2).apply(
                lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
            )
        )
        
        # Volatility features
        df['sales_volatility_30d'] = df.groupby('UserCode')['FinalValue'].transform(
            lambda x: x.rolling(window=30, min_periods=5).std()
        )
        
        # Market penetration features
        df['market_penetration'] = df.groupby(['Date', 'UserCode'])['DistributorCode'].transform('nunique')
        
        # Sales efficiency features
        df['orders_per_day'] = df.groupby(['UserCode', 'Date'])['Code'].transform('count')
        df['avg_order_value'] = df['FinalValue'] / df['orders_per_day']
        
        # Cyclical features for seasonality
        df['month_sin'] = np.sin(2 * np.pi * df['Date_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Date_month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['Date_day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['Date_day'] / 31)
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for machine learning."""
        
        # Identify categorical columns
        categorical_cols = [
            'DistributorCode', 'UserCode', 'season', 'sales_category',
            'customer_dealer_combo'
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                # Use label encoding for high cardinality features
                if df[col].nunique() > 50:
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
                else:
                    # Use one-hot encoding for low cardinality features
                    dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                    df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features."""
        
        # Features to normalize
        normalize_cols = [
            'FinalValue', 'processing_time_minutes', 'sales_7d_avg', 'sales_30d_avg',
            'recency_days', 'frequency_orders', 'monetary_total', 'sales_volatility_30d'
        ]
        
        normalize_cols = [col for col in normalize_cols if col in df.columns]
        
        if self.normalization_method == 'standard':
            scaler = StandardScaler()
        elif self.normalization_method == 'robust':
            scaler = RobustScaler()
        else:
            return df
        
        if normalize_cols:
            df[normalize_cols] = scaler.fit_transform(df[normalize_cols])
            self.scalers['main'] = scaler
        
        return df
    
    def _create_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create business intelligence features."""
        
        # Sales performance tiers
        df['performance_tier'] = pd.cut(
            df['FinalValue'],
            bins=df['FinalValue'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            labels=['Bottom 20%', 'Low', 'Medium', 'High', 'Top 20%'],
            include_lowest=True
        )
        
        # Customer lifetime value proxy
        df['customer_lifetime_value'] = df.groupby('DistributorCode')['FinalValue'].transform('sum')
        
        # Sales concentration (Gini coefficient approximation)
        def gini_coefficient(x):
            """Calculate Gini coefficient for sales concentration."""
            x = np.array(x)
            x = x[x > 0]  # Remove zeros
            if len(x) == 0:
                return 0
            x = np.sort(x)
            n = len(x)
            index = np.arange(1, n + 1)
            return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n
        
        df['sales_concentration'] = df.groupby('UserCode')['FinalValue'].transform(
            lambda x: gini_coefficient(x)
        )
        
        # Market share features
        total_sales_by_date = df.groupby('Date')['FinalValue'].sum()
        df['daily_market_share'] = df.apply(
            lambda row: row['FinalValue'] / total_sales_by_date[row['Date']]
            if total_sales_by_date[row['Date']] > 0 else 0,
            axis=1
        )
        
        # Growth metrics
        df['sales_growth_mom'] = df.groupby('UserCode')['FinalValue'].pct_change(periods=30)
        df['sales_growth_yoy'] = df.groupby('UserCode')['FinalValue'].pct_change(periods=365)
        
        return df
    
    def _assess_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assess and score data quality."""
        
        df['data_quality_score'] = 100.0
        
        # Penalize missing values
        missing_penalty = df.isnull().sum(axis=1) * 2
        df['data_quality_score'] -= missing_penalty
        
        # Penalize outliers
        df.loc[df['is_outlier'], 'data_quality_score'] -= 20
        
        # Penalize unusual processing times
        if 'processing_time_minutes' in df.columns:
            df.loc[df['processing_time_minutes'] > 60, 'data_quality_score'] -= 10
            df.loc[df['processing_time_minutes'] < 0, 'data_quality_score'] -= 30
        
        # Penalize weekend/non-business hour transactions (unusual)
        df.loc[df['Date_is_weekend'], 'data_quality_score'] -= 5
        df.loc[~df['is_business_hours'], 'data_quality_score'] -= 5
        
        # Ensure score doesn't go below 0
        df['data_quality_score'] = df['data_quality_score'].clip(lower=0)
        
        return df
    
    def create_features_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create final feature set optimized for machine learning."""
        
        # Select features for ML models
        ml_features = [
            # Basic sales features
            'FinalValue', 'FinalValue_log', 'processing_time_minutes',
            
            # Time features
            'Date_month', 'Date_day', 'Date_dayofweek', 'Date_hour',
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            
            # Business features
            'recency_days', 'frequency_orders', 'monetary_total',
            'sales_7d_avg', 'sales_30d_avg', 'sales_trend_7d',
            'sales_volatility_30d', 'market_penetration',
            
            # Performance features
            'orders_per_day', 'avg_order_value', 'sales_per_hour',
            'customer_lifetime_value', 'sales_concentration',
            'daily_market_share', 'sales_growth_mom',
            
            # Boolean features
            'is_weekend', 'is_business_hours', 'is_festival_season',
            'is_rush_order', 'is_outlier'
        ]
        
        # Filter features that exist in the dataframe
        available_features = [col for col in ml_features if col in df.columns]
        
        # Create final ML dataset
        ml_df = df[available_features + ['DistributorCode', 'UserCode', 'Date']].copy()
        
        # Handle any remaining missing values
        ml_df = ml_df.fillna(ml_df.mean(numeric_only=True))
        
        return ml_df
    
    def generate_preprocessing_report(self, original_df: pd.DataFrame, 
                                    processed_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive preprocessing report."""
        
        report = {
            'summary': {
                'original_records': len(original_df),
                'processed_records': len(processed_df),
                'records_removed': len(original_df) - len(processed_df),
                'removal_rate': (len(original_df) - len(processed_df)) / len(original_df) * 100
            },
            'sales_analysis': {
                'total_sales_value': processed_df['FinalValue'].sum(),
                'avg_order_value': processed_df['FinalValue'].mean(),
                'median_order_value': processed_df['FinalValue'].median(),
                'sales_std': processed_df['FinalValue'].std(),
                'min_order_value': processed_df['FinalValue'].min(),
                'max_order_value': processed_df['FinalValue'].max()
            },
            'data_quality': {
                'avg_quality_score': processed_df['data_quality_score'].mean(),
                'high_quality_records': len(processed_df[processed_df['data_quality_score'] >= 80]),
                'low_quality_records': len(processed_df[processed_df['data_quality_score'] < 50]),
                'outliers_detected': processed_df['is_outlier'].sum(),
                'missing_values_handled': original_df.isnull().sum().sum()
            },
            'business_insights': {
                'unique_distributors': processed_df['DistributorCode'].nunique(),
                'unique_users': processed_df['UserCode'].nunique(),
                'date_range': {
                    'start': processed_df['Date'].min(),
                    'end': processed_df['Date'].max()
                },
                'peak_sales_month': processed_df.groupby('Date_month')['FinalValue'].sum().idxmax(),
                'peak_sales_day': processed_df.groupby('Date_dayofweek')['FinalValue'].sum().idxmax(),
                'weekend_sales_ratio': processed_df[processed_df['Date_is_weekend']]['FinalValue'].sum() / 
                                     processed_df['FinalValue'].sum()
            },
            'feature_engineering': {
                'total_features_created': len(processed_df.columns) - len(original_df.columns),
                'categorical_features_encoded': len(self.encoders),
                'numerical_features_scaled': len(self.scalers),
                'rfm_features_created': 'recency_days' in processed_df.columns
            }
        }
        
        return report

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = SalesPreprocessor(
        outlier_method='iqr',
        missing_strategy='knn',
        normalization_method='robust'
    )
    
    # Load sample data
    sample_data = {
        'Code': ['PO202U248230031', 'PO202U260230015', 'PO202U248230032'],
        'Date': pd.to_datetime(['2023-01-02 10:15:41', '2023-01-02 10:26:03', '2023-01-02 11:42:49']),
        'DistributorCode': ['116836', '126671', '120456'],
        'UserCode': ['KNS-002645', 'KHC-002653', 'KNS-002645'],
        'UserName': ['K Nuwan Sameera', 'K H Chathura Jayasanka', 'K Nuwan Sameera'],
        'FinalValue': [206700.01, 259920.14, 64800.03],
        'CreationDate': pd.to_datetime(['2023-01-02 10:16:44', '2023-01-02 11:44:47', '2023-01-02 12:02:59']),
        'SubmittedDate': pd.to_datetime(['2023-01-02 10:16:44', '2023-01-02 11:44:47', '2023-01-02 12:02:59']),
        'ERPOrderNumber': ['SO202-B1860022', 'SO202-B1860027', 'SO202-B1860023']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Preprocess the data
    processed_df = preprocessor.preprocess_sales_data(df)
    
    # Create ML features
    ml_df = preprocessor.create_features_for_ml(processed_df)
    
    # Generate report
    report = preprocessor.generate_preprocessing_report(df, processed_df)
    
    print("Sales Preprocessing Report:")
    print(f"Original records: {report['summary']['original_records']}")
    print(f"Processed records: {report['summary']['processed_records']}")
    print(f"Average order value: {report['sales_analysis']['avg_order_value']:.2f}")
    print(f"Features created: {report['feature_engineering']['total_features_created']}")