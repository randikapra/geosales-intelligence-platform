"""
Advanced Model Loading and Prediction Interface
Unified interface for all SFA ML models with caching and optimization
"""
import os
import logging
import pickle
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.base import BaseEstimator
import tensorflow as tf
from tensorflow import keras

# Custom imports (assume they exist in your project)
from data.preprocessors.gps_preprocessor import GPSPreprocessor
from data.preprocessors.sales_preprocessor import SalesPreprocessor
from data.preprocessors.customer_preprocessor import CustomerPreprocessor
from data.preprocessors.feature_engineer import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model loading and prediction"""
    model_path: str
    model_type: str  # 'sklearn', 'tensorflow', 'pytorch', 'prophet', 'custom'
    preprocessing_config: Dict
    feature_columns: List[str]
    target_column: str
    model_version: str = "1.0"
    created_date: str = None
    performance_metrics: Dict = None
    
    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now().isoformat()

class ModelRegistry:
    """Registry for managing multiple models"""
    
    def __init__(self, registry_path: str = "./models/registry.json"):
        self.registry_path = registry_path
        self.models = {}
        self.load_registry()
    
    def load_registry(self):
        """Load model registry from file"""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, 'r') as f:
                    registry_data = json.load(f)
                    for name, config_dict in registry_data.items():
                        self.models[name] = ModelConfig(**config_dict)
                logger.info(f"Loaded {len(self.models)} models from registry")
            else:
                logger.info("No existing registry found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
    
    def save_registry(self):
        """Save model registry to file"""
        try:
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            registry_data = {}
            for name, config in self.models.items():
                registry_data[name] = {
                    'model_path': config.model_path,
                    'model_type': config.model_type,
                    'preprocessing_config': config.preprocessing_config,
                    'feature_columns': config.feature_columns,
                    'target_column': config.target_column,
                    'model_version': config.model_version,
                    'created_date': config.created_date,
                    'performance_metrics': config.performance_metrics
                }
            
            with open(self.registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
            logger.info(f"Registry saved with {len(self.models)} models")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register_model(self, name: str, config: ModelConfig):
        """Register a new model"""
        self.models[name] = config
        self.save_registry()
        logger.info(f"Model '{name}' registered successfully")
    
    def get_model_config(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration"""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.models.keys())
    
    def remove_model(self, name: str):
        """Remove model from registry"""
        if name in self.models:
            del self.models[name]
            self.save_registry()
            logger.info(f"Model '{name}' removed from registry")
        else:
            logger.warning(f"Model '{name}' not found in registry")

class UniversalPredictor:
    """Universal predictor that can handle all types of SFA models"""
    
    def __init__(self, cache_size: int = 10):
        self.model_cache = {}
        self.preprocessor_cache = {}
        self.cache_size = cache_size
        self.registry = ModelRegistry()
        
        # Initialize preprocessors
        self.gps_preprocessor = GPSPreprocessor()
        self.sales_preprocessor = SalesPreprocessor()
        self.customer_preprocessor = CustomerPreprocessor()
        self.feature_engineer = FeatureEngineer()
    
    def load_model(self, model_name: str, force_reload: bool = False) -> Any:
        """Load model with caching"""
        if model_name in self.model_cache and not force_reload:
            logger.info(f"Loading model '{model_name}' from cache")
            return self.model_cache[model_name]
        
        config = self.registry.get_model_config(model_name)
        if not config:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model file not found: {config.model_path}")
        
        try:
            if config.model_type == 'sklearn':
                model = joblib.load(config.model_path)
            elif config.model_type == 'tensorflow':
                model = keras.models.load_model(config.model_path)
            elif config.model_type == 'pickle':
                with open(config.model_path, 'rb') as f:
                    model = pickle.load(f)
            elif config.model_type == 'prophet':
                # Handle Prophet models
                with open(config.model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            
            # Cache management
            if len(self.model_cache) >= self.cache_size:
                # Remove oldest model
                oldest_model = next(iter(self.model_cache))
                del self.model_cache[oldest_model]
                logger.info(f"Removed '{oldest_model}' from cache")
            
            self.model_cache[model_name] = model
            logger.info(f"Model '{model_name}' loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame, model_name: str, 
                       data_type: str = 'mixed') -> pd.DataFrame:
        """Preprocess data based on model configuration"""
        config = self.registry.get_model_config(model_name)
        if not config:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        preprocessing_config = config.preprocessing_config
        
        try:
            processed_data = data.copy()
            
            # Apply preprocessing based on data type
            if data_type == 'gps' or 'gps' in data_type.lower():
                processed_data = self.gps_preprocessor.preprocess(processed_data)
            elif data_type == 'sales' or 'sales' in data_type.lower():
                processed_data = self.sales_preprocessor.preprocess(processed_data)
            elif data_type == 'customer' or 'customer' in data_type.lower():
                processed_data = self.customer_preprocessor.preprocess(processed_data)
            
            # Apply feature engineering
            if preprocessing_config.get('feature_engineering', True):
                processed_data = self.feature_engineer.create_features(processed_data)
            
            # Apply scaling if specified
            if preprocessing_config.get('scaling'):
                scaler_type = preprocessing_config['scaling']['type']
                if scaler_type == 'standard':
                    scaler = StandardScaler()
                elif scaler_type == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = None
                
                if scaler:
                    numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
                    processed_data[numeric_columns] = scaler.fit_transform(processed_data[numeric_columns])
            
            # Select required features
            if config.feature_columns:
                available_features = [col for col in config.feature_columns if col in processed_data.columns]
                if len(available_features) != len(config.feature_columns):
                    missing_features = set(config.feature_columns) - set(available_features)
                    logger.warning(f"Missing features: {missing_features}")
                processed_data = processed_data[available_features]
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to preprocess data for model '{model_name}': {e}")
            raise
    
    def predict(self, model_name: str, data: pd.DataFrame, 
                data_type: str = 'mixed', return_probabilities: bool = False) -> Dict[str, Any]:
        """Make predictions using specified model"""
        try:
            # Load model
            model = self.load_model(model_name)
            config = self.registry.get_model_config(model_name)
            
            # Preprocess data
            processed_data = self.preprocess_data(data, model_name, data_type)
            
            # Make predictions
            if config.model_type == 'tensorflow':
                predictions = model.predict(processed_data.values)
                if predictions.shape[1] == 1:
                    predictions = predictions.flatten()
            else:
                predictions = model.predict(processed_data)
            
            # Get prediction probabilities if available and requested
            probabilities = None
            if return_probabilities and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_data)
            
            # Prepare results
            results = {
                'model_name': model_name,
                'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                'prediction_time': datetime.now().isoformat(),
                'data_shape': processed_data.shape,
                'model_version': config.model_version
            }
            
            if probabilities is not None:
                results['probabilities'] = probabilities.tolist()
            
            # Add confidence scores if available
            if hasattr(model, 'decision_function'):
                confidence_scores = model.decision_function(processed_data)
                results['confidence_scores'] = confidence_scores.tolist()
            
            logger.info(f"Prediction completed for model '{model_name}' on {len(data)} samples")
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed for model '{model_name}': {e}")
            raise
    
    def predict_batch(self, model_name: str, data_list: List[pd.DataFrame], 
                     data_type: str = 'mixed') -> List[Dict[str, Any]]:
        """Make batch predictions"""
        results = []
        for i, data in enumerate(data_list):
            try:
                result = self.predict(model_name, data, data_type)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction failed for index {i}: {e}")
                results.append({
                    'batch_index': i,
                    'error': str(e),
                    'prediction_time': datetime.now().isoformat()
                })
        
        return results
    
    def predict_with_explanation(self, model_name: str, data: pd.DataFrame, 
                               data_type: str = 'mixed') -> Dict[str, Any]:
        """Make predictions with feature importance explanation"""
        try:
            # Get basic prediction
            result = self.predict(model_name, data, data_type)
            
            # Add feature importance if available
            model = self.load_model(model_name)
            if hasattr(model, 'feature_importances_'):
                config = self.registry.get_model_config(model_name)
                feature_importance = dict(zip(config.feature_columns, model.feature_importances_))
                result['feature_importance'] = feature_importance
            
            # Add SHAP values if SHAP is available
            try:
                import shap
                processed_data = self.preprocess_data(data, model_name, data_type)
                
                if hasattr(model, 'predict_proba'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(processed_data)
                    result['shap_values'] = shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values
                
            except ImportError:
                logger.info("SHAP not available for explanations")
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction with explanation failed: {e}")
            raise
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        config = self.registry.get_model_config(model_name)
        if not config:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        info = {
            'name': model_name,
            'type': config.model_type,
            'version': config.model_version,
            'created_date': config.created_date,
            'feature_columns': config.feature_columns,
            'target_column': config.target_column,
            'preprocessing_config': config.preprocessing_config,
            'performance_metrics': config.performance_metrics
        }
        
        # Add model file info
        if os.path.exists(config.model_path):
            stat = os.stat(config.model_path)
            info['file_size'] = stat.st_size
            info['last_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        # Add model-specific info if model is loaded
        if model_name in self.model_cache:
            model = self.model_cache[model_name]
            if hasattr(model, 'get_params'):
                info['model_parameters'] = model.get_params()
        
        return info
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the predictor system"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'cached_models': len(self.model_cache),
            'registered_models': len(self.registry.list_models()),
            'issues': []
        }
        
        # Check model files
        for model_name in self.registry.list_models():
            config = self.registry.get_model_config(model_name)
            if not os.path.exists(config.model_path):
                health_status['issues'].append(f"Model file missing: {model_name}")
                health_status['status'] = 'degraded'
        
        # Check preprocessors
        try:
            test_data = pd.DataFrame({'test': [1, 2, 3]})
            self.gps_preprocessor.preprocess(test_data)
            self.sales_preprocessor.preprocess(test_data)
            self.customer_preprocessor.preprocess(test_data)
        except Exception as e:
            health_status['issues'].append(f"Preprocessor issue: {str(e)}")
            health_status['status'] = 'unhealthy'
        
        return health_status
    
    def clear_cache(self):
        """Clear model cache"""
        self.model_cache.clear()
        self.preprocessor_cache.clear()
        logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return {
            'cached_models': list(self.model_cache.keys()),
            'cache_size': len(self.model_cache),
            'max_cache_size': self.cache_size,
            'memory_usage': {
                'model_cache': len(self.model_cache),
                'preprocessor_cache': len(self.preprocessor_cache)
            }
        }

# Factory function for easy instantiation
def create_predictor(cache_size: int = 10, registry_path: str = "./models/registry.json") -> UniversalPredictor:
    """Factory function to create a UniversalPredictor instance"""
    predictor = UniversalPredictor(cache_size=cache_size)
    predictor.registry.registry_path = registry_path
    predictor.registry.load_registry()
    return predictor

# Example usage and testing
if __name__ == "__main__":
    # Create predictor instance
    predictor = create_predictor()
    
    # Example model registration
    sample_config = ModelConfig(
        model_path="./models/sample_model.pkl",
        model_type="sklearn",
        preprocessing_config={
            "scaling": {"type": "standard"},
            "feature_engineering": True
        },
        feature_columns=["feature1", "feature2", "feature3"],
        target_column="target",
        model_version="1.0",
        performance_metrics={"accuracy": 0.95, "f1_score": 0.92}
    )
    
    predictor.registry.register_model("sample_model", sample_config)
    
    # Health check
    health = predictor.health_check()
    print("Health Status:", health)
    
    # Cache info
    cache_info = predictor.get_cache_info()
    print("Cache Info:", cache_info)