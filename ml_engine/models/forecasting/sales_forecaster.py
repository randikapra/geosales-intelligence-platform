# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import xgboost as xgb
# from typing import Dict, List, Tuple, Any
# import joblib
# import warnings
# warnings.filterwarnings('ignore')

# class SalesForecaster:
#     def __init__(self):
#         self.model = None
#         self.scaler = StandardScaler()
#         self.feature_names = []
#         self.is_trained = False
        
#     def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Create comprehensive features for sales forecasting"""
        
#         # Ensure date column is datetime
#         df['date'] = pd.to_datetime(df['date'])
#         df = df.sort_values('date')
        
#         # Time-based features
#         df['year'] = df['date'].dt.year
#         df['month'] = df['date'].dt.month
#         df['day'] = df['date'].dt.day
#         df['day_of_week'] = df['date'].dt.dayofweek
#         df['day_of_year'] = df['date'].dt.dayofyear
#         df['week_of_year'] = df['date'].dt.isocalendar().week
#         df['quarter'] = df['date'].dt.quarter
#         df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
#         df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
#         df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
#         # Lag features
#         for lag in [1, 2, 3, 7, 14, 30]:
#             df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
        
#         # Rolling statistics
#         for window in [3, 7, 14, 30]:
#             df[f'sales_rolling_mean_{window}'] = df['sales'].rolling(window=window).mean()
#             df[f'sales_rolling_std_{window}'] = df['sales'].rolling(window=window).std()
#             df[f'sales_rolling_min_{window}'] = df['sales'].rolling(window=window).min()
#             df[f'sales_rolling_max_{window}'] = df['sales'].rolling(window=window).max()
        
#         # Exponential moving averages
#         for alpha in [0.3, 0.5, 0.7]:
#             df[f'sales_ema_{alpha}'] = df['sales'].ewm(alpha=alpha).mean()
        
#         # Seasonal decomposition features
#         df['sales_trend'] = df['sales'].rolling(window=30, center=True).mean()
#         df['sales_detrended'] = df['sales'] - df['sales_trend']
        
#         # Cyclical features
#         df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
#         df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
#         df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
#         df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
#         df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
#         df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        
#         # Customer and dealer features
#         if 'customer_code' in df.columns:
#             df['customer_visits_30d'] = df.groupby('customer_code')['sales'].rolling(30).count().reset_index(0, drop=True)
#             df['customer_avg_order_30d'] = df.groupby('customer_code')['sales'].rolling(30).mean().reset_index(0, drop=True)
        
#         if 'dealer_code' in df.columns:
#             df['dealer_performance_30d'] = df.groupby('dealer_code')['sales'].rolling(30).sum().reset_index(0, drop=True)
#             df['dealer_visit_frequency'] = df.groupby('dealer_code')['sales'].rolling(30).count().reset_index(0, drop=True)
        
#         # Territory features
#         if 'territory_code' in df.columns:
#             df['territory_total_30d'] = df.groupby('territory_code')['sales'].rolling(30).sum().reset_index(0, drop=True)
#             df['territory_avg_30d'] = df.groupby('territory_code')['sales'].rolling(30).mean().reset_index(0, drop=True)
        
#         # GPS-based features (if location data available)
#         if 'latitude' in df.columns and 'longitude' in df.columns:
#             df['distance_from_center'] = np.sqrt((df['latitude'] - df['latitude'].mean())**2 + 
#                                                 (df['longitude'] - df['longitude'].mean())**2)
#             df['location_cluster'] = self._cluster_locations(df[['latitude', 'longitude']])
        
#         # External factors (weather, holidays, etc.)
#         df['is_holiday'] = self._mark_holidays(df['date'])
#         df['season'] = df['month'].map({12: 'winter', 1: 'winter', 2: 'winter',
#                                       3: 'spring', 4: 'spring', 5: 'spring',
#                                       6: 'summer', 7: 'summer', 8: 'summer',
#                                       9: 'autumn', 10: 'autumn', 11: 'autumn'})
        
#         # Drop rows with NaN values created by lag features
#         df = df.dropna()
        
#         return df
    
#     def train(self, data: pd.DataFrame, target_column: str = 'sales') -> Dict[str, Any]:
#         """Train the sales forecasting model"""
        
#         # Create features
#         df = self.create_features(data.copy())
        
#         # Separate features and target
#         feature_columns = [col for col in df.columns if col not in ['date', target_column]]
#         X = df[feature_columns]
#         y = df[target_column]
        
#         # Store feature names
#         self.feature_names = feature_columns
        
#         # Time series split for validation
#         tscv = TimeSeriesSplit(n_splits=5)
        
#         # Scale features
#         X_scaled = self.scaler.fit_transform(X)
        
#         # Train multiple models and ensemble
#         models = {
#             'rf': RandomForestRegressor(n_estimators=100, random_state=42),
#             'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
#             'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42)
#         }
        
#         best_model = None
#         best_score = float('inf')
#         model_scores = {}
        
#         for name, model in models.items():
#             scores = []
#             for train_idx, val_idx in tscv.split(X_scaled):
#                 X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
#                 y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
#                 model.fit(X_train, y_train)
#                 predictions = model.predict(X_val)
#                 score = mean_absolute_error(y_val, predictions)
#                 scores.append(score)
            
#             avg_score = np.mean(scores)
#             model_scores[name] = avg_score
            
#             if avg_score < best_score:
#                 best_score = avg_score
#                 best_model = model
        
#         # Train final model on all data
#         self.model = best_model
#         self.model.fit(X_scaled, y)
#         self.is_trained = True
        
#         # Calculate final metrics
#         y_pred = self.model.predict(X_scaled)
#         metrics = {
#             'mae': mean_absolute_error(y, y_pred),
#             'mse': mean_squared_error(y, y_pred),
#             'rmse': np.sqrt(mean_squared_error(y, y_pred)),
#             'r2': r2_score(y, y_pred),
#             'model_scores': model_scores,
#             'best_model': type(best_model).__name__
#         }
        
#         return metrics
    
#     def predict(self, data: pd.DataFrame, periods: int = 30) -> Dict[str, Any]:
#         """Generate sales forecasts"""
        
#         if not self.is_trained:
#             raise ValueError("Model must be trained before making predictions")
        
#         # Create features for prediction data
#         df = self.create_features(data.copy())
        
#         # Make predictions for existing data
#         X = df[self.feature_names]
#         X_scaled = self.scaler.transform(X)
        
#         predictions = []
#         confidence_intervals = []
        
#         # Generate future predictions
#         last_date = df['date'].max()
#         future_data = df.tail(50).copy()  # Use last 50 records for context
        
#         for i in range(periods):
#             pred_date = last_date + timedelta(days=i+1)
            
#             # Create future features
#             future_row = self._create_future_features(future_data, pred_date)
#             future_features = future_row[self.feature_names].values.reshape(1, -1)
#             future_scaled = self.scaler.transform(future_features)
            
#             # Make prediction
#             pred = self.model.predict(future_scaled)[0]
#             predictions.append(pred)
            
#             # Calculate confidence interval (using prediction uncertainty)
#             ci = self._calculate_confidence_interval(pred, future_scaled)
#             confidence_intervals.append(ci)
            
#             # Add prediction to future_data for next iteration
#             new_row = future_row.copy()
#             new_row['sales'] = pred
#             future_data = pd.concat([future_data, new_row.to_frame().T], ignore_index=True)
        
#         return {
#             'predictions': predictions,
#             'confidence_intervals': confidence_intervals,
#             'dates': [last_date + timedelta(days=i+1) for i in range(periods)]
#         }
    
#     def _cluster_locations(self, coordinates: pd.DataFrame) -> np.ndarray:
#         """Cluster geographic locations"""
#         from sklearn.cluster import KMeans
        
#         if len(coordinates) < 3:
#             return np.zeros(len(coordinates))
        
#         kmeans = KMeans(n_clusters=min(5, len(coordinates)//10 + 1), random_state=42)
#         return kmeans.fit_predict(coordinates)
    
#     def _mark_holidays(self, dates: pd.Series) -> pd.Series:
#         """Mark holiday dates (customize for your region)"""
#         holidays = [
#             '2023-01-01', '2023-04-14', '2023-04-15', '2023-05-01',
#             '2023-05-22', '2023-08-15', '2023-12-25', '2023-12-26'
#         ]
        
#         holiday_dates = pd.to_datetime(holidays).date
#         return dates.dt.date.isin(holiday_dates).astype(int)
    
#     def _create_future_features(self, historical_data: pd.DataFrame, pred_date: datetime) -> pd.Series:
#         """Create features for future prediction"""
        
#         # Create base row with date features
#         row = pd.Series(index=historical_data.columns)
#         row['date'] = pred_date
#         row['year'] = pred_date.year
#         row['month'] = pred_date.month
#         row['day'] = pred_date.day
#         row['day_of_week'] = pred_date.weekday()
#         row['day_of_year'] = pred_date.timetuple().tm_yday
#         row['quarter'] = (pred_date.month - 1) // 3 + 1
#         row['is_weekend'] = int(pred_date.weekday() >= 5)
        
#         # Use recent historical values for lag features
#         recent_sales = historical_data['sales'].tail(30).values
        
#         # Fill lag features
#         for lag in [1, 2, 3, 7, 14, 30]:
#             if lag <= len(recent_sales):
#                 row[f'sales_lag_{lag}'] = recent_sales[-lag]
#             else:
#                 row[f'sales_lag_{lag}'] = recent_sales[-1]
        
#         # Fill rolling features with recent averages
#         for window in [3, 7, 14, 30]:
#             if len(recent_sales) >= window:
#                 row[f'sales_rolling_mean_{window}'] = np.mean(recent_sales[-window:])
#                 row[f'sales_rolling_std_{window}'] = np.std(recent_sales[-window:])
#                 row[f'sales_rolling_min_{window}'] = np.min(recent_sales[-window:])
#                 row[f'sales_rolling_max_{window}'] = np.max(recent_sales[-window:])
#             else:
#                 row[f'sales_rolling_mean_{window}'] = np.mean(recent_sales)
#                 row[f'sales_rolling_std_{window}'] = np.std(recent_sales)
#                 row[f'sales_rolling_min_{window}'] = np.min(recent_sales)
#                 row[f'sales_rolling_max_{window}'] = np.max(recent_sales)
        
#         # Fill other features with default values
#         row = row.fillna(0)
        
#         return row
    
#     def _calculate_confidence_interval(self, prediction: float, features: np.ndarray) -> Tuple[float, float]:
#         """Calculate prediction confidence interval"""
        
#         # Simple approach using prediction variance
#         # In practice, you might want to use more sophisticated methods
#         uncertainty = 0.1 * abs(prediction)  # 10% uncertainty
        
#         return (prediction - uncertainty, prediction + uncertainty)
    
#     def save_model(self, filepath: str):
#         """Save trained model"""
#         if not self.is_trained:
#             raise ValueError("Model must be trained before saving")
        
#         model_data = {
#             'model': self.model,
#             'scaler': self.scaler,
#             'feature_names': self.feature_names,
#             'is_trained': self.is_trained
#         }
        
#         joblib.dump(model_data, filepath)
    
#     def load_model(self, filepath: str):
#         """Load trained model"""
#         model_data = joblib.load(filepath)
        
#         self.model = model_data['model']
#         self.scaler = model_data['scaler']
#         self.feature_names = model_data['feature_names']
#         self.is_trained = model_data['is_trained']

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Scientific Libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# Utilities
import joblib
import json
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ForecastConfig:
    """Configuration for forecasting models"""
    forecast_horizon: int = 30  # days
    confidence_level: float = 0.95
    seasonal_periods: int = 7  # weekly seasonality
    validation_split: float = 0.2
    model_save_path: str = "models/forecasting/"
    
class SalesForecaster:
    """
    Advanced Sales Forecasting System with multiple models
    Supports ARIMA, SARIMA, Prophet, LSTM, and Ensemble methods
    """
    
    def __init__(self, config: ForecastConfig = None):
        self.config = config or ForecastConfig()
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'FinalValue'
        self.date_column = 'Date'
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create model directory
        Path(self.config.model_save_path).mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, df: pd.DataFrame, dealer_code: str = None) -> pd.DataFrame:
        """
        Prepare sales data for forecasting
        """
        # Convert date column
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        # Filter by dealer if specified
        if dealer_code:
            df = df[df['UserCode'] == dealer_code].copy()
        
        # Create daily aggregations
        daily_sales = df.groupby(self.date_column).agg({
            'FinalValue': 'sum',
            'Code': 'count',  # number of orders
            'DistributorCode': 'nunique'  # unique customers
        }).reset_index()
        
        daily_sales.columns = ['Date', 'TotalSales', 'OrderCount', 'CustomerCount']
        
        # Create complete date range
        date_range = pd.date_range(
            start=daily_sales['Date'].min(),
            end=daily_sales['Date'].max(),
            freq='D'
        )
        
        # Reindex to fill missing dates
        daily_sales = daily_sales.set_index('Date').reindex(date_range, fill_value=0)
        daily_sales.index.name = 'Date'
        daily_sales = daily_sales.reset_index()
        
        # Create features
        daily_sales = self._create_features(daily_sales)
        
        return daily_sales
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based and lag features
        """
        df = df.copy()
        
        # Time-based features
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfMonth'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        
        # Cyclical encoding
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Lag features
        for lag in [1, 3, 7, 14, 30]:
            df[f'TotalSales_lag_{lag}'] = df['TotalSales'].shift(lag)
            df[f'OrderCount_lag_{lag}'] = df['OrderCount'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'TotalSales_rolling_mean_{window}'] = df['TotalSales'].rolling(window=window).mean()
            df[f'TotalSales_rolling_std_{window}'] = df['TotalSales'].rolling(window=window).std()
            df[f'OrderCount_rolling_mean_{window}'] = df['OrderCount'].rolling(window=window).mean()
        
        # Exponential moving averages
        df['TotalSales_ema_7'] = df['TotalSales'].ewm(span=7).mean()
        df['TotalSales_ema_30'] = df['TotalSales'].ewm(span=30).mean()
        
        # Trend features
        df['TotalSales_pct_change'] = df['TotalSales'].pct_change()
        df['TotalSales_diff'] = df['TotalSales'].diff()
        
        return df
    
    def train_arima_model(self, df: pd.DataFrame) -> Dict:
        """
        Train ARIMA/SARIMA model with automatic parameter selection
        """
        self.logger.info("Training ARIMA model...")
        
        # Prepare data
        sales_series = df.set_index('Date')['TotalSales']
        
        # Auto ARIMA parameter selection
        from pmdarima import auto_arima
        
        model = auto_arima(
            sales_series,
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            seasonal=True,
            m=7,  # weekly seasonality
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        
        self.models['arima'] = model
        
        # Generate forecasts
        forecast, conf_int = model.predict(
            n_periods=self.config.forecast_horizon,
            return_conf_int=True,
            alpha=1-self.config.confidence_level
        )
        
        # Create forecast dates
        last_date = df['Date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=self.config.forecast_horizon,
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast,
            'Lower_CI': conf_int[:, 0],
            'Upper_CI': conf_int[:, 1],
            'Model': 'ARIMA'
        })
        
        return {
            'model': model,
            'forecast': forecast_df,
            'aic': model.aic(),
            'params': model.get_params()
        }
    
    def train_prophet_model(self, df: pd.DataFrame) -> Dict:
        """
        Train Facebook Prophet model with additional regressors
        """
        self.logger.info("Training Prophet model...")
        
        # Prepare data for Prophet
        prophet_df = df[['Date', 'TotalSales']].rename(columns={
            'Date': 'ds',
            'TotalSales': 'y'
        })
        
        # Initialize Prophet with seasonality
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            holidays_prior_scale=10,
            interval_width=self.config.confidence_level
        )
        
        # Add additional regressors
        if 'OrderCount' in df.columns:
            model.add_regressor('OrderCount')
            prophet_df['OrderCount'] = df['OrderCount']
        
        if 'CustomerCount' in df.columns:
            model.add_regressor('CustomerCount')
            prophet_df['CustomerCount'] = df['CustomerCount']
        
        # Fit model
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(
            periods=self.config.forecast_horizon,
            freq='D'
        )
        
        # Add regressor values for future dates (using last known values)
        for col in ['OrderCount', 'CustomerCount']:
            if col in prophet_df.columns:
                last_value = prophet_df[col].iloc[-1]
                future[col] = future[col].fillna(last_value)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract forecast for future dates only
        forecast_df = forecast.tail(self.config.forecast_horizon)[
            ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        ].rename(columns={
            'ds': 'Date',
            'yhat': 'Forecast',
            'yhat_lower': 'Lower_CI',
            'yhat_upper': 'Upper_CI'
        })
        forecast_df['Model'] = 'Prophet'
        
        self.models['prophet'] = model
        
        return {
            'model': model,
            'forecast': forecast_df,
            'components': model.predict(future)[['ds', 'trend', 'weekly', 'yearly']].tail(self.config.forecast_horizon)
        }
    
    def train_lstm_model(self, df: pd.DataFrame) -> Dict:
        """
        Train LSTM model with attention mechanism
        """
        self.logger.info("Training LSTM model...")
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['Date', 'TotalSales']]
        self.feature_columns = feature_cols
        
        # Scale data
        scaler_X = StandardScaler()
        scaler_y = MinMaxScaler()
        
        X = scaler_X.fit_transform(df[feature_cols].fillna(0))
        y = scaler_y.fit_transform(df[['TotalSales']])
        
        self.scalers['lstm_X'] = scaler_X
        self.scalers['lstm_y'] = scaler_y
        
        # Create sequences
        def create_sequences(X, y, seq_length=30):
            X_seq, y_seq = [], []
            for i in range(seq_length, len(X)):
                X_seq.append(X[i-seq_length:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        
        seq_length = 30
        X_seq, y_seq = create_sequences(X, y, seq_length)
        
        if len(X_seq) == 0:
            self.logger.warning("Not enough data for LSTM training")
            return None
        
        # Train-validation split
        split_idx = int(len(X_seq) * (1 - self.config.validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Build LSTM model
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=(seq_length, len(feature_cols))),
            Dropout(0.2),
            Bidirectional(LSTM(50, return_sequences=True)),
            Dropout(0.2),
            LSTM(25),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        self.models['lstm'] = model
        
        # Generate forecasts
        last_sequence = X_seq[-1:]
        forecasts = []
        
        for _ in range(self.config.forecast_horizon):
            pred = model.predict(last_sequence, verbose=0)
            forecasts.append(pred[0, 0])
            
            # Update sequence (simplified - in practice, you'd need future features)
            new_row = last_sequence[0, -1:].copy()
            new_row[0, 0] = pred[0, 0]  # Update target feature
            last_sequence = np.concatenate([last_sequence[0, 1:], new_row]).reshape(1, seq_length, -1)
        
        # Inverse transform forecasts
        forecasts = scaler_y.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
        
        # Create forecast dataframe
        last_date = df['Date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=self.config.forecast_horizon,
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecasts,
            'Model': 'LSTM'
        })
        
        return {
            'model': model,
            'forecast': forecast_df,
            'history': history.history,
            'val_loss': min(history.history['val_loss'])
        }
    
    def train_ensemble_model(self, df: pd.DataFrame) -> Dict:
        """
        Train ensemble of XGBoost, LightGBM, and Random Forest
        """
        self.logger.info("Training ensemble model...")
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['Date', 'TotalSales']]
        X = df[feature_cols].fillna(0)
        y = df['TotalSales']
        
        # Train-test split
        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Initialize models
        models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        }
        
        # Train models
        trained_models = {}
        predictions = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
            predictions[name] = model.predict(X_val)
        
        # Ensemble weights based on validation performance
        weights = {}
        total_weight = 0
        
        for name, pred in predictions.items():
            mse = mean_squared_error(y_val, pred)
            weight = 1 / (mse + 1e-8)  # Inverse MSE weighting
            weights[name] = weight
            total_weight += weight
        
        # Normalize weights
        for name in weights:
            weights[name] /= total_weight
        
        self.models['ensemble'] = {
            'models': trained_models,
            'weights': weights
        }
        
        # Generate ensemble forecasts
        # For simplicity, using last known values for future features
        last_features = X.iloc[-1:].copy()
        forecasts = []
        
        for _ in range(self.config.forecast_horizon):
            ensemble_pred = 0
            for name, model in trained_models.items():
                pred = model.predict(last_features)[0]
                ensemble_pred += pred * weights[name]
            
            forecasts.append(ensemble_pred)
            
            # Update features (simplified)
            last_features.iloc[0, 0] = ensemble_pred  # Update lag feature
        
        # Create forecast dataframe
        last_date = df['Date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=self.config.forecast_horizon,
            freq='D'
        )
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecasts,
            'Model': 'Ensemble'
        })
        
        return {
            'models': trained_models,
            'weights': weights,
            'forecast': forecast_df,
            'feature_importance': self._get_feature_importance(trained_models, feature_cols)
        }
    
    def _get_feature_importance(self, models: Dict, feature_cols: List[str]) -> Dict:
        """
        Get feature importance from ensemble models
        """
        importance_dict = {}
        
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = dict(zip(feature_cols, model.feature_importances_))
        
        return importance_dict
    
    def train_all_models(self, df: pd.DataFrame, dealer_code: str = None) -> Dict:
        """
        Train all forecasting models and return combined results
        """
        self.logger.info(f"Training all models for dealer: {dealer_code or 'All'}")
        
        # Prepare data
        prepared_data = self.prepare_data(df, dealer_code)
        
        if len(prepared_data) < 60:  # Need minimum data
            self.logger.warning("Insufficient data for training")
            return None
        
        results = {}
        
        # Train each model
        try:
            results['arima'] = self.train_arima_model(prepared_data)
        except Exception as e:
            self.logger.error(f"ARIMA training failed: {e}")
        
        try:
            results['prophet'] = self.train_prophet_model(prepared_data)
        except Exception as e:
            self.logger.error(f"Prophet training failed: {e}")
        
        try:
            results['lstm'] = self.train_lstm_model(prepared_data)
        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
        
        try:
            results['ensemble'] = self.train_ensemble_model(prepared_data)
        except Exception as e:
            self.logger.error(f"Ensemble training failed: {e}")
        
        # Create combined forecast
        if results:
            combined_forecast = self._create_combined_forecast(results)
            results['combined'] = combined_forecast
        
        return results
    
    def _create_combined_forecast(self, results: Dict) -> Dict:
        """
        Create combined forecast from all models
        """
        forecasts = []
        
        for model_name, result in results.items():
            if 'forecast' in result:
                forecast_df = result['forecast'].copy()
                forecasts.append(forecast_df)
        
        if not forecasts:
            return None
        
        # Combine forecasts (simple average)
        combined_df = forecasts[0][['Date']].copy()
        combined_df['Combined_Forecast'] = np.mean([df['Forecast'].values for df in forecasts], axis=0)
        combined_df['Model'] = 'Combined'
        
        # Calculate prediction intervals (using ensemble if available)
        if 'ensemble' in results:
            ensemble_forecast = results['ensemble']['forecast']
            combined_df['Lower_CI'] = ensemble_forecast['Forecast'] * 0.8
            combined_df['Upper_CI'] = ensemble_forecast['Forecast'] * 1.2
        
        return {
            'forecast': combined_df,
            'individual_forecasts': forecasts
        }
    
    def save_models(self, dealer_code: str = None):
        """
        Save trained models to disk
        """
        suffix = f"_{dealer_code}" if dealer_code else "_global"
        
        for model_name, model in self.models.items():
            filename = f"{self.config.model_save_path}/{model_name}_model{suffix}.pkl"
            joblib.dump(model, filename)
            self.logger.info(f"Saved {model_name} model to {filename}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            filename = f"{self.config.model_save_path}/{scaler_name}_scaler{suffix}.pkl"
            joblib.dump(scaler, filename)
    
    def load_models(self, dealer_code: str = None):
        """
        Load trained models from disk
        """
        suffix = f"_{dealer_code}" if dealer_code else "_global"
        
        for model_name in ['arima', 'prophet', 'lstm', 'ensemble']:
            try:
                filename = f"{self.config.model_save_path}/{model_name}_model{suffix}.pkl"
                self.models[model_name] = joblib.load(filename)
                self.logger.info(f"Loaded {model_name} model from {filename}")
            except FileNotFoundError:
                self.logger.warning(f"Model file not found: {filename}")
    
    def predict(self, model_name: str = 'combined', horizon: int = None) -> pd.DataFrame:
        """
        Generate predictions using specified model
        """
        horizon = horizon or self.config.forecast_horizon
        
        if model_name not in self.models:
            self.logger.error(f"Model {model_name} not found")
            return None
        
        # Implementation would depend on the specific model
        # This is a placeholder for the prediction logic
        pass
    
    def evaluate_model(self, df: pd.DataFrame, model_name: str) -> Dict:
        """
        Evaluate model performance on historical data
        """
        # Split data for evaluation
        split_idx = int(len(df) * 0.8)
        train_data = df.iloc[:split_idx]
        test_data = df.iloc[split_idx:]
        
        # Generate predictions for test period
        # (Implementation would vary by model)
        
        # Calculate metrics
        metrics = {
            'mae': 0,  # mean_absolute_error(actual, predicted)
            'mse': 0,  # mean_squared_error(actual, predicted)
            'rmse': 0,  # np.sqrt(mse)
            'mape': 0,  # Mean Absolute Percentage Error
            'r2': 0    # r2_score(actual, predicted)
        }
        
        return metrics

# Example usage
if __name__ == "__main__":
    # Initialize forecaster
    config = ForecastConfig(
        forecast_horizon=30,
        confidence_level=0.95,
        model_save_path="models/forecasting/"
    )
    
    forecaster = SalesForecaster(config)
    
    # Load sample data (you would load your actual data here)
    # df = pd.read_excel("SFA_Orders.xlsx", sheet_name="Jan")
    
    # Train all models
    # results = forecaster.train_all_models(df, dealer_code="KNS-002645")
    
    # Save models
    # forecaster.save_models(dealer_code="KNS-002645")
    
    print("Sales Forecaster initialized successfully!")