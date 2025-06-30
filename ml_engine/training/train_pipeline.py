"""
End-to-End Training Pipeline Orchestration
Comprehensive ML pipeline for SFA system with all models
"""

import os
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import joblib
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import mlflow
import mlflow.sklearn
import mlflow.pytorch

# Custom imports (assuming these exist in your project)
from data.preprocessors.gps_preprocessor import GPSPreprocessor
from data.preprocessors.sales_preprocessor import SalesPreprocessor
from data.preprocessors.customer_preprocessor import CustomerPreprocessor
from data.preprocessors.feature_engineer import FeatureEngineer

from models.forecasting.sales_forecaster import SalesForecaster
from models.forecasting.demand_forecaster import DemandForecaster
from models.classification.customer_segmentation import CustomerSegmentation
from models.classification.churn_prediction import ChurnPredictor
from models.classification.dealer_performance import DealerPerformanceClassifier
from models.optimization.route_optimizer import RouteOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the training pipeline"""
    data_path: str
    model_output_path: str
    experiment_name: str
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    enable_mlflow: bool = True
    parallel_training: bool = True
    max_workers: int = 4
    
    # Model specific configs
    sales_forecast_horizon: int = 30  # days
    demand_forecast_horizon: int = 7   # days
    customer_segments: int = 5
    enable_hyperparameter_tuning: bool = True
    cross_validation_folds: int = 5

class TrainingPipeline:
    """Main training pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.models = {}
        self.metrics = {}
        self.preprocessors = {}
        self.feature_engineers = {}
        
        # Setup MLflow
        if config.enable_mlflow:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.experiment_name)
    
    async def load_and_preprocess_data(self) -> Dict[str, pd.DataFrame]:
        """Load and preprocess all datasets"""
        logger.info("Loading and preprocessing data...")
        
        # Load raw data
        raw_data = await self._load_raw_data()
        
        # Initialize preprocessors
        self.preprocessors = {
            'gps': GPSPreprocessor(),
            'sales': SalesPreprocessor(),
            'customer': CustomerPreprocessor(),
        }
        
        # Preprocess data in parallel
        tasks = []
        for data_type, preprocessor in self.preprocessors.items():
            if data_type in raw_data:
                task = asyncio.create_task(
                    self._preprocess_data_async(preprocessor, raw_data[data_type], data_type)
                )
                tasks.append(task)
        
        preprocessed_results = await asyncio.gather(*tasks)
        
        # Combine results
        preprocessed_data = {}
        for result in preprocessed_results:
            preprocessed_data.update(result)
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        self.feature_engineers['main'] = feature_engineer
        
        enhanced_data = await self._engineer_features(feature_engineer, preprocessed_data)
        
        logger.info("Data preprocessing completed")
        return enhanced_data
    
    async def _load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load raw datasets"""
        data = {}
        
        # Load Customer data
        customer_excel_path = os.path.join(self.config.data_path, "Customer.xlsx")
        customer_csv_path = os.path.join(self.config.data_path, "customer.csv")
        
        if os.path.exists(customer_excel_path):
            data['customer_excel'] = pd.read_excel(customer_excel_path)
        if os.path.exists(customer_csv_path):
            data['customer_csv'] = pd.read_csv(customer_csv_path)
        
        # Load GPS data
        gps_files = [f for f in os.listdir(self.config.data_path) 
                    if f.startswith('SFA_GPSData') and f.endswith('.csv')]
        
        gps_data_list = []
        for gps_file in gps_files:
            gps_df = pd.read_csv(os.path.join(self.config.data_path, gps_file))
            gps_data_list.append(gps_df)
        
        if gps_data_list:
            data['gps'] = pd.concat(gps_data_list, ignore_index=True)
        
        # Load Sales data (multiple sheets)
        sales_excel_path = os.path.join(self.config.data_path, "SFA_Orders.xlsx")
        if os.path.exists(sales_excel_path):
            xl_file = pd.ExcelFile(sales_excel_path)
            sales_sheets = []
            for sheet_name in xl_file.sheet_names:
                sheet_df = pd.read_excel(xl_file, sheet_name=sheet_name)
                sheet_df['month'] = sheet_name
                sales_sheets.append(sheet_df)
            data['sales'] = pd.concat(sales_sheets, ignore_index=True)
        
        # Load PO data
        po_path = os.path.join(self.config.data_path, "SFA_PO.csv")
        if os.path.exists(po_path):
            data['po'] = pd.read_csv(po_path)
        
        return data
    
    async def _preprocess_data_async(self, preprocessor, data: pd.DataFrame, data_type: str) -> Dict[str, pd.DataFrame]:
        """Preprocess data asynchronously"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            processed_data = await loop.run_in_executor(
                executor, preprocessor.preprocess, data
            )
        return {data_type: processed_data}
    
    async def _engineer_features(self, feature_engineer: FeatureEngineer, 
                               data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Engineer features from preprocessed data"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            enhanced_data = await loop.run_in_executor(
                executor, feature_engineer.create_features, data
            )
        return enhanced_data
    
    async def train_all_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train all models in parallel"""
        logger.info("Starting model training...")
        
        training_tasks = []
        
        # Sales Forecasting
        if 'sales' in data:
            task = asyncio.create_task(
                self._train_sales_forecaster(data['sales'])
            )
            training_tasks.append(('sales_forecaster', task))
        
        # Demand Forecasting
        if 'sales' in data and 'customer' in data:
            task = asyncio.create_task(
                self._train_demand_forecaster(data['sales'], data.get('customer'))
            )
            training_tasks.append(('demand_forecaster', task))
        
        # Customer Segmentation
        if 'customer' in data:
            task = asyncio.create_task(
                self._train_customer_segmentation(data['customer'])
            )
            training_tasks.append(('customer_segmentation', task))
        
        # Churn Prediction
        if 'sales' in data and 'customer' in data:
            task = asyncio.create_task(
                self._train_churn_predictor(data['sales'], data['customer'])
            )
            training_tasks.append(('churn_predictor', task))
        
        # Dealer Performance Classification
        if 'sales' in data and 'gps' in data:
            task = asyncio.create_task(
                self._train_dealer_performance(data['sales'], data.get('gps'))
            )
            training_tasks.append(('dealer_performance', task))
        
        # Route Optimization
        if 'gps' in data and 'customer' in data:
            task = asyncio.create_task(
                self._train_route_optimizer(data['gps'], data['customer'])
            )
            training_tasks.append(('route_optimizer', task))
        
        # Execute all training tasks
        results = {}
        for model_name, task in training_tasks:
            try:
                model, metrics = await task
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'trained_at': datetime.now()
                }
                logger.info(f"✅ {model_name} training completed")
            except Exception as e:
                logger.error(f"❌ {model_name} training failed: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    async def _train_sales_forecaster(self, sales_data: pd.DataFrame) -> Tuple[Any, Dict]:
        """Train sales forecasting model"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            forecaster = SalesForecaster(
                forecast_horizon=self.config.sales_forecast_horizon
            )
            
            # Prepare time series data
            ts_data = self._prepare_sales_timeseries(sales_data)
            
            # Train model
            model = await loop.run_in_executor(
                executor, forecaster.fit, ts_data
            )
            
            # Evaluate
            metrics = await loop.run_in_executor(
                executor, self._evaluate_forecaster, forecaster, ts_data
            )
            
            return model, metrics
    
    async def _train_demand_forecaster(self, sales_data: pd.DataFrame, 
                                     customer_data: pd.DataFrame) -> Tuple[Any, Dict]:
        """Train demand forecasting model"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            forecaster = DemandForecaster(
                forecast_horizon=self.config.demand_forecast_horizon
            )
            
            # Prepare demand data
            demand_data = self._prepare_demand_data(sales_data, customer_data)
            
            # Train model
            model = await loop.run_in_executor(
                executor, forecaster.fit, demand_data
            )
            
            # Evaluate
            metrics = await loop.run_in_executor(
                executor, self._evaluate_forecaster, forecaster, demand_data
            )
            
            return model, metrics
    
    async def _train_customer_segmentation(self, customer_data: pd.DataFrame) -> Tuple[Any, Dict]:
        """Train customer segmentation model"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            segmenter = CustomerSegmentation(
                n_clusters=self.config.customer_segments
            )
            
            # Prepare features
            features = self._prepare_customer_features(customer_data)
            
            # Train model
            model = await loop.run_in_executor(
                executor, segmenter.fit, features
            )
            
            # Evaluate
            metrics = await loop.run_in_executor(
                executor, self._evaluate_clustering, segmenter, features
            )
            
            return model, metrics
    
    async def _train_churn_predictor(self, sales_data: pd.DataFrame, 
                                   customer_data: pd.DataFrame) -> Tuple[Any, Dict]:
        """Train churn prediction model"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            predictor = ChurnPredictor()
            
            # Prepare churn features
            X, y = self._prepare_churn_data(sales_data, customer_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, 
                random_state=self.config.random_state, stratify=y
            )
            
            # Train model
            model = await loop.run_in_executor(
                executor, predictor.fit, X_train, y_train
            )
            
            # Evaluate
            metrics = await loop.run_in_executor(
                executor, self._evaluate_classifier, predictor, X_test, y_test
            )
            
            return model, metrics
    
    async def _train_dealer_performance(self, sales_data: pd.DataFrame, 
                                      gps_data: pd.DataFrame) -> Tuple[Any, Dict]:
        """Train dealer performance classification model"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            classifier = DealerPerformanceClassifier()
            
            # Prepare dealer features
            X, y = self._prepare_dealer_features(sales_data, gps_data)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, 
                random_state=self.config.random_state, stratify=y
            )
            
            # Train model
            model = await loop.run_in_executor(
                executor, classifier.fit, X_train, y_train
            )
            
            # Evaluate
            metrics = await loop.run_in_executor(
                executor, self._evaluate_classifier, classifier, X_test, y_test
            )
            
            return model, metrics
    
    async def _train_route_optimizer(self, gps_data: pd.DataFrame, 
                                   customer_data: pd.DataFrame) -> Tuple[Any, Dict]:
        """Train route optimization model"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            optimizer = RouteOptimizer()
            
            # Prepare route data
            route_data = self._prepare_route_data(gps_data, customer_data)
            
            # Train/optimize
            model = await loop.run_in_executor(
                executor, optimizer.fit, route_data
            )
            
            # Evaluate
            metrics = await loop.run_in_executor(
                executor, self._evaluate_optimizer, optimizer, route_data
            )
            
            return model, metrics
    
    def _prepare_sales_timeseries(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare sales data for time series forecasting"""
        # Convert date columns
        sales_data['Date'] = pd.to_datetime(sales_data['Date'])
        
        # Aggregate by date
        ts_data = sales_data.groupby('Date').agg({
            'FinalValue': 'sum',
            'Code': 'count'
        }).reset_index()
        
        ts_data.columns = ['date', 'total_sales', 'order_count']
        ts_data = ts_data.set_index('date').sort_index()
        
        return ts_data
    
    def _prepare_demand_data(self, sales_data: pd.DataFrame, 
                           customer_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare demand forecasting data"""
        # Merge sales with customer location data
        merged_data = sales_data.merge(
            customer_data, 
            left_on='DistributorCode', 
            right_on='No', 
            how='left'
        )
        
        # Create demand features by location and time
        demand_data = merged_data.groupby(['Date', 'City']).agg({
            'FinalValue': 'sum',
            'Code': 'count'
        }).reset_index()
        
        return demand_data
    
    def _prepare_customer_features(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare customer features for segmentation"""
        # Select relevant features for clustering
        feature_cols = ['Contact', 'City']  # Add more features as available
        features = customer_data[feature_cols].copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        for col in features.select_dtypes(include=['object']).columns:
            features[col] = le.fit_transform(features[col].astype(str))
        
        return features
    
    def _prepare_churn_data(self, sales_data: pd.DataFrame, 
                          customer_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare churn prediction data"""
        # Calculate recency, frequency, monetary features
        current_date = sales_data['Date'].max()
        customer_features = sales_data.groupby('DistributorCode').agg({
            'Date': lambda x: (current_date - x.max()).days,  # Recency
            'Code': 'count',  # Frequency
            'FinalValue': ['sum', 'mean']  # Monetary
        }).reset_index()
        
        customer_features.columns = [
            'DistributorCode', 'recency', 'frequency', 'monetary_sum', 'monetary_avg'
        ]
        
        # Define churn (no purchase in last 90 days)
        customer_features['churned'] = (customer_features['recency'] > 90).astype(int)
        
        X = customer_features[['recency', 'frequency', 'monetary_sum', 'monetary_avg']]
        y = customer_features['churned']
        
        return X, y
    
    def _prepare_dealer_features(self, sales_data: pd.DataFrame, 
                               gps_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare dealer performance features"""
        # Sales performance metrics
        sales_metrics = sales_data.groupby('UserCode').agg({
            'FinalValue': ['sum', 'mean', 'count'],
            'Date': lambda x: x.nunique()  # Active days
        }).reset_index()
        
        sales_metrics.columns = [
            'UserCode', 'total_sales', 'avg_order_value', 'order_count', 'active_days'
        ]
        
        # GPS activity metrics
        if gps_data is not None:
            gps_metrics = gps_data.groupby('UserCode').agg({
                'Latitude': 'count',  # GPS points
                'RecievedDate': lambda x: pd.to_datetime(x).dt.date.nunique()  # Active GPS days
            }).reset_index()
            
            gps_metrics.columns = ['UserCode', 'gps_points', 'gps_active_days']
            
            # Merge features
            features = sales_metrics.merge(gps_metrics, on='UserCode', how='left')
        else:
            features = sales_metrics
        
        # Create performance labels (top 25% = high, bottom 25% = low, middle = medium)
        features['sales_rank'] = features['total_sales'].rank(pct=True)
        conditions = [
            features['sales_rank'] <= 0.25,
            features['sales_rank'] >= 0.75
        ]
        choices = [0, 2]  # 0=Low, 1=Medium, 2=High
        features['performance_class'] = np.select(conditions, choices, default=1)
        
        X = features.drop(['UserCode', 'sales_rank', 'performance_class'], axis=1)
        y = features['performance_class']
        
        return X, y
    
    def _prepare_route_data(self, gps_data: pd.DataFrame, 
                          customer_data: pd.DataFrame) -> Dict:
        """Prepare route optimization data"""
        # Extract unique locations from GPS data
        locations = gps_data[['Latitude', 'Longitude']].drop_duplicates()
        
        # Add customer locations
        if 'Latitude' in customer_data.columns and 'Longitude' in customer_data.columns:
            customer_locations = customer_data[['Latitude', 'Longitude']].drop_duplicates()
            locations = pd.concat([locations, customer_locations]).drop_duplicates()
        
        return {
            'locations': locations.values,
            'gps_tracks': gps_data
        }
    
    def _evaluate_forecaster(self, forecaster, data: pd.DataFrame) -> Dict:
        """Evaluate forecasting model"""
        # Split data for evaluation
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # Make predictions
        predictions = forecaster.predict(len(test_data))
        
        # Calculate metrics
        mae = mean_absolute_error(test_data.iloc[:, 0], predictions)
        mse = mean_squared_error(test_data.iloc[:, 0], predictions)
        rmse = np.sqrt(mse)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'train_size': len(train_data),
            'test_size': len(test_data)
        }
    
    def _evaluate_clustering(self, clusterer, data: pd.DataFrame) -> Dict:
        """Evaluate clustering model"""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        labels = clusterer.predict(data)
        
        # Calculate clustering metrics
        silhouette = silhouette_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)
        
        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'n_clusters': len(np.unique(labels)),
            'n_samples': len(data)
        }
    
    def _evaluate_classifier(self, classifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate classification model"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'test_size': len(X_test)
        }
        
        # Add AUC for binary classification
        if len(np.unique(y_test)) == 2:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        return metrics
    
    def _evaluate_optimizer(self, optimizer, data: Dict) -> Dict:
        """Evaluate route optimizer"""
        # Calculate basic route metrics
        locations = data['locations']
        n_locations = len(locations)
        
        # Simple evaluation metrics
        return {
            'n_locations': n_locations,
            'optimization_method': type(optimizer).__name__,
            'data_points': len(data.get('gps_tracks', [])),
        }
    
    async def save_models(self, trained_models: Dict[str, Any]):
        """Save all trained models"""
        logger.info("Saving trained models...")
        
        # Create output directory
        os.makedirs(self.config.model_output_path, exist_ok=True)
        
        for model_name, model_data in trained_models.items():
            if 'error' in model_data:
                continue
                
            try:
                # Save model
                model_path = os.path.join(self.config.model_output_path, f"{model_name}.joblib")
                joblib.dump(model_data['model'], model_path)
                
                # Save metrics
                metrics_path = os.path.join(self.config.model_output_path, f"{model_name}_metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(model_data['metrics'], f, indent=2, default=str)
                
                # Log to MLflow
                if self.config.enable_mlflow:
                    with mlflow.start_run(run_name=f"{model_name}_training"):
                        mlflow.log_params({
                            'model_type': model_name,
                            'training_time': model_data['trained_at']
                        })
                        mlflow.log_metrics(model_data['metrics'])
                        mlflow.log_artifact(model_path)
                
                logger.info(f"✅ Saved {model_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to save {model_name}: {str(e)}")
    
    async def generate_training_report(self, trained_models: Dict[str, Any]) -> str:
        """Generate comprehensive training report"""
        report = {
            'training_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_models': len(trained_models),
                'successful_models': len([m for m in trained_models.values() if 'error' not in m]),
                'failed_models': len([m for m in trained_models.values() if 'error' in m])
            },
            'model_details': {}
        }
        
        for model_name, model_data in trained_models.items():
            if 'error' in model_data:
                report['model_details'][model_name] = {'status': 'failed', 'error': model_data['error']}
            else:
                report['model_details'][model_name] = {
                    'status': 'success',
                    'metrics': model_data['metrics'],
                    'trained_at': model_data['trained_at'].isoformat()
                }
        
        # Save report
        report_path = os.path.join(self.config.model_output_path, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_path
    
    async def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        try:
            logger.info("🚀 Starting ML Training Pipeline")
            
            # Load and preprocess data
            data = await self.load_and_preprocess_data()
            
            # Train all models
            trained_models = await self.train_all_models(data)
            
            # Save models
            await self.save_models(trained_models)
            
            # Generate report
            report_path = await self.generate_training_report(trained_models)
            
            logger.info(f"🎉 Pipeline completed! Report saved to: {report_path}")
            
            return {
                'status': 'completed',
                'models': trained_models,
                'report_path': report_path
            }
            
        except Exception as e:
            logger.error(f"💥 Pipeline failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

# Usage example
async def main():
    config = PipelineConfig(
        data_path="./data",
        model_output_path="./models/trained",
        experiment_name="SFA_ML_Pipeline",
        enable_mlflow=True,
        parallel_training=True
    )
    
    pipeline = TrainingPipeline(config)
    results = await pipeline.run_pipeline()
    
    print("Pipeline Results:", results)

if __name__ == "__main__":
    asyncio.run(main())