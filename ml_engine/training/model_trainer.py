"""
Advanced Model Training Logic with Hyperparameter Tuning
Comprehensive training system for all SFA ML models
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, 
    TimeSeriesSplit, StratifiedKFold, KFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import optuna
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    # General settings
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.1
    cv_folds: int = 5
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = True
    tuning_method: str = 'optuna'  # 'grid', 'random', 'optuna'
    n_trials: int = 100
    timeout: int = 3600  # 1 hour
    
    # Model specific settings
    enable_ensemble: bool = True
    early_stopping_rounds: int = 50
    max_iter: int = 1000
    
    # Evaluation settings
    scoring_metric: str = 'auto'  # auto, rmse, accuracy, f1, etc.
    save_plots: bool = True
    save_feature_importance: bool = True

class BaseModelTrainer(ABC):
    """Abstract base class for model trainers"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.best_params = None
        self.training_history = {}
        self.feature_importance = None
        
    @abstractmethod
    def get_model_space(self) -> Dict:
        """Get hyperparameter search space"""
        pass
    
    @abstractmethod
    def create_model(self, params: Dict = None) -> Any:
        """Create model instance with given parameters"""
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict:
        """Get default model parameters"""
        pass
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series = None, 
                       fit_preprocessor: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess training data"""
        # Handle missing values
        X_processed = X.fillna(X.mean(numeric_only=True))
        X_processed = X_processed.fillna(0)  # For non-numeric
        
        # Scale features if needed
        if not hasattr(self, 'scaler'):
            self.scaler = StandardScaler()
        
        if fit_preprocessor:
            X_scaled = self.scaler.fit_transform(X_processed)
        else:
            X_scaled = self.scaler.transform(X_processed)
        
        y_processed = y.values if y is not None else None
        
        return X_scaled, y_processed
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize hyperparameters using specified method"""
        if not self.config.enable_hyperparameter_tuning:
            return self.get_default_params()
        
        if self.config.tuning_method == 'optuna':
            return self._optuna_optimization(X, y)
        elif self.config.tuning_method == 'grid':
            return self._grid_search_optimization(X, y)
        elif self.config.tuning_method == 'random':
            return self._random_search_optimization(X, y)
        else:
            logger.warning(f"Unknown tuning method: {self.config.tuning_method}")
            return self.get_default_params()
    
    def _optuna_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize using Optuna"""
        def objective(trial):
            # Get parameter suggestions from search space
            params = {}
            search_space = self.get_model_space()
            
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            # Create and evaluate model
            model = self.create_model(params)
            scores = self._cross_validate(model, X, y)
            return np.mean(scores)
        
        # Create study
        study = optuna.create_study(
            direction='maximize' if self._is_classification() else 'minimize'
        )
        
        # Optimize
        study.optimize(
            objective, 
            n_trials=self.config.n_trials,
            timeout=self.config.timeout
        )
        
        logger.info(f"Best score: {study.best_value}")
        return study.best_params
    
    def _grid_search_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize using GridSearchCV"""
        param_grid = self._convert_space_to_grid(self.get_model_space())
        
        model = self.create_model()
        cv = self._get_cv_splitter()
        scoring = self._get_scoring_metric()
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        return grid_search.best_params_
    
    def _random_search_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize using RandomizedSearchCV"""
        param_distributions = self._convert_space_to_random(self.get_model_space())
        
        model = self.create_model()
        cv = self._get_cv_splitter()
        scoring = self._get_scoring_metric()
        
        random_search = RandomizedSearchCV(
            model, param_distributions, cv=cv, scoring=scoring,
            n_iter=self.config.n_trials, n_jobs=-1, 
            random_state=self.config.random_state, verbose=1
        )
        
        random_search.fit(X, y)
        return random_search.best_params_
    
    def _cross_validate(self, model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perform cross-validation"""
        cv = self._get_cv_splitter()
        scoring = self._get_scoring_metric()
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return scores
    
    def _get_cv_splitter(self):
        """Get appropriate CV splitter"""
        if self._is_time_series():
            return TimeSeriesSplit(n_splits=self.config.cv_folds)
        elif self._is_classification():
            return StratifiedKFold(
                n_splits=self.config.cv_folds, 
                shuffle=True, 
                random_state=self.config.random_state
            )
        else:
            return KFold(
                n_splits=self.config.cv_folds, 
                shuffle=True, 
                random_state=self.config.random_state
            )
    
    def _get_scoring_metric(self) -> str:
        """Get appropriate scoring metric"""
        if self.config.scoring_metric != 'auto':
            return self.config.scoring_metric
        
        if self._is_classification():
            return 'f1_weighted'
        else:
            return 'neg_root_mean_squared_error'
    
    def _is_classification(self) -> bool:
        """Check if this is a classification task"""
        return 'Classifier' in self.__class__.__name__
    
    def _is_time_series(self) -> bool:
        """Check if this is a time series task"""
        return 'Forecaster' in self.__class__.__name__
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train the model"""
        logger.info(f"Training {self.__class__.__name__}...")
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y, fit_preprocessor=True)
        
        # Optimize hyperparameters
        self.best_params = self.optimize_hyperparameters(X_processed, y_processed)
        logger.info(f"Best parameters: {self.best_params}")
        
        # Train final model
        self.model = self.create_model(self.best_params)
        
        # Fit model
        start_time = datetime.now()
        self.model.fit(X_processed, y_processed)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_).flatten()
        
        # Store training history
        self.training_history = {
            'training_time_seconds': training_time,
            'best_params': self.best_params,
            'feature_names': X.columns.tolist(),
            'training_samples': len(X)
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'model': self.model,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance
        }
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Preprocess test data
        X_processed, y_processed = self.preprocess_data(X_test, y_test, fit_preprocessor=False)
        
        # Make predictions
        y_pred = self.model.predict(X_processed)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_processed, y_pred)
        
        # Add prediction probabilities for classification
        if self._is_classification() and hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_processed)
            metrics['predictions_proba'] = y_pred_proba
        
        metrics['predictions'] = y_pred
        metrics['actual'] = y_processed
        
        return metrics
    
    @abstractmethod
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate model-specific metrics"""
        pass
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'best_params': self.best_params,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        instance = cls(model_data['config'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.best_params = model_data['best_params']
        instance.training_history = model_data['training_history']
        instance.feature_importance = model_data['feature_importance']
        
        return instance

class RegressionTrainer(BaseModelTrainer):
    """Trainer for regression models"""
    
    def get_model_space(self) -> Dict:
        """Get regression model hyperparameter space"""
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]}
        }
    
    def create_model(self, params: Dict = None) -> Any:
        """Create regression model"""
        if params is None:
            params = self.get_default_params()
        
        return RandomForestRegressor(
            random_state=self.config.random_state,
            **params
        )
    
    def get_default_params(self) -> Dict:
        """Get default regression parameters"""
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt'
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

class ClassificationTrainer(BaseModelTrainer):
    """Trainer for classification models"""
    
    def get_model_space(self) -> Dict:
        """Get classification model hyperparameter space"""
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]}
        }
    
    def create_model(self, params: Dict = None) -> Any:
        """Create classification model"""
        if params is None:
            params = self.get_default_params()
        
        return RandomForestClassifier(
            random_state=self.config.random_state,
            **params
        )
    
    def get_default_params(self) -> Dict:
        """Get default classification parameters"""
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt'
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Add AUC for binary classification
        if len(np.unique(y_true)) == 2:
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(self.scaler.transform(
                    pd.DataFrame(y_true).fillna(0)
                ))[:, 1]
                metrics['auc'] = roc_auc_score(y_true, y_prob)
        
        return metrics

class XGBoostTrainer(BaseModelTrainer):
    """Trainer for XGBoost models"""
    
    def __init__(self, config: TrainingConfig, task_type: str = 'regression'):
        super().__init__(config)
        self.task_type = task_type
    
    def get_model_space(self) -> Dict:
        """Get XGBoost hyperparameter space"""
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 1000},
            'max_depth': {'type': 'int', 'low': 3, 'high': 12},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'reg_alpha': {'type': 'float', 'low': 0, 'high': 10},
            'reg_lambda': {'type': 'float', 'low': 0, 'high': 10}
        }
    
    def create_model(self, params: Dict = None) -> Any:
        """Create XGBoost model"""
        if params is None:
            params = self.get_default_params()
        
        if self.task_type == 'classification':
            return xgb.XGBClassifier(
                random_state=self.config.random_state,
                **params
            )
        else:
            return xgb.XGBRegressor(
                random_state=self.config.random_state,
                **params
            )
    
    def get_default_params(self) -> Dict:
        """Get default XGBoost parameters"""
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    
    def _is_classification(self) -> bool:
        """Check if this is a classification task"""
        return self.task_type == 'classification'
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate XGBoost metrics"""
        if self.task_type == 'classification':
            return ClassificationTrainer._calculate_metrics(self, y_true, y_pred)
        else:
            return RegressionTrainer._calculate_metrics(self, y_true, y_pred)

class EnsembleTrainer(BaseModelTrainer):
    """Trainer for ensemble models"""
    
    def __init__(self, config: TrainingConfig, task_type: str = 'regression'):
        super().__init__(config)
        self.task_type = task_type
        self.models = {}
        self.model_weights = {}
    
    def get_model_space(self) -> Dict:
        """Ensemble doesn't need hyperparameter tuning"""
        return {}
    
    def create_model(self, params: Dict = None) -> Any:
        """Create ensemble of models"""
        models = {}
        
        if self.task_type == 'classification':
            models['rf'] = RandomForestClassifier(
                n_estimators=100, random_state=self.config.random_state
            )
            models['xgb'] = xgb.XGBClassifier(
                n_estimators=100, random_state=self.config.random_state
            )
            models['lgb'] = lgb.LGBMClassifier(
                n_estimators=100, random_state=self.config.random_state, verbose=-1
            )
        else:
            models['rf'] = RandomForestRegressor(
                n_estimators=100, random_state=self.config.random_state
            )
            models['xgb'] = xgb.XGBRegressor(
                n_estimators=100, random_state=self.config.random_state
            )
            models['lgb'] = lgb.LGBMRegressor(
                n_estimators=100, random_state=self.config.random_state, verbose=-1
            )
        
        return models
    
    def get_default_params(self) -> Dict:
        """No default params for ensemble"""
        return {}
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train ensemble of models"""
        logger.info(f"Training {self.__class__.__name__}...")
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y, fit_preprocessor=True)
        
        # Create models
        self.models = self.create_model()
        
        # Train each model
        start_time = datetime.now()
        model_scores = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_processed, y_processed)
            
            # Cross-validate to get weights
            scores = self._cross_validate(model, X_processed, y_processed)
            model_scores[name] = np.mean(scores)
        
        # Calculate model weights based on performance
        total_score = sum(model_scores.values())
        self.model_weights = {
            name: score / total_score 
            for name, score in model_scores.items()
        }
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store training history
        self.training_history = {
            'training_time_seconds': training_time,
            'model_scores': model_scores,
            'model_weights': self.model_weights,
            'feature_names': X.columns.tolist(),
            'training_samples': len(X)
        }
        
        logger.info(f"Ensemble training completed in {training_time:.2f} seconds")
        logger.info(f"Model weights: {self.model_weights}")
        
        return {
            'models': self.models,
            'training_history': self.training_history,
            'model_weights': self.model_weights
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        X_processed, _ = self.preprocess_data(X, fit_preprocessor=False)
        
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X_processed)
            weighted_pred = pred * self.model_weights[name]
            predictions.append(weighted_pred)
        
        return np.sum(predictions, axis=0)
    
    def _is_classification(self) -> bool:
        """Check if this is a classification task"""
        return self.task_type == 'classification'
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate ensemble metrics"""
        if self.task_type == 'classification':
            return ClassificationTrainer._calculate_metrics(self, y_true, y_pred)
        else:
            return RegressionTrainer._calculate_metrics(self, y_true, y_pred)

class ModelTrainerFactory:
    """Factory for creating model trainers"""
    
    @staticmethod
    def create_trainer(model_type: str, config: TrainingConfig, **kwargs) -> BaseModelTrainer:
        """Create appropriate trainer based on model type"""
        
        trainers = {
            'regression': RegressionTrainer,
            'classification': ClassificationTrainer,
            'xgboost_regression': lambda config: XGBoostTrainer(config, 'regression'),
            'xgboost_classification': lambda config: XGBoostTrainer(config, 'classification'),
            'ensemble_regression': lambda config: EnsembleTrainer(config, 'regression'),
            'ensemble_classification': lambda config: EnsembleTrainer(config, 'classification')
        }
        
        if model_type not in trainers:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return trainers[model_type](config)

# Utility functions
def _convert_space_to_grid(search_space: Dict) -> Dict:
    """Convert Optuna search space to GridSearchCV format"""
    param_grid = {}
    
    for param_name, param_config in search_space.items():
        if param_config['type'] == 'int':
            param_grid[param_name] = list(range(
                param_config['low'], param_config['high'] + 1, 
                max(1, (param_config['high'] - param_config['low']) // 10)
            ))
        elif param_config['type'] == 'float':
            if param_config.get('log', False):
                param_grid[param_name] = np.logspace(
                    np.log10(param_config['low']), 
                    np.log10(param_config['high']), 
                    10
                )
            else:
                param_grid[param_name] = np.linspace(
                    param_config['low'], param_config['high'], 10
                )
        elif param_config['type'] == 'categorical':
            param_grid[param_name] = param_config['choices']
    
    return param_grid

def _convert_space_to_random(search_space: Dict) -> Dict:
    """Convert Optuna search space to RandomizedSearchCV format"""
    param_distributions = {}
    
    for param_name, param_config in search_space.items():
        if param_config['type'] == 'int':
            param_distributions[param_name] = stats.randint(
                param_config['low'], param_config['high'] + 1
            )
        elif param_config['type'] == 'float':
            if param_config.get('log', False):
                param_distributions[param_name] = stats.loguniform(
                    param_config['low'], param_config['high']
                )
            else:
                param_distributions[param_name] = stats.uniform(
                    param_config['low'], 
                    param_config['high'] - param_config['low']
                )
        elif param_config['type'] == 'categorical':
            param_distributions[param_name] = param_config['choices']
    
    return param_distributions

# Usage example
def main():
    # Configuration
    config = TrainingConfig(
        enable_hyperparameter_tuning=True,
        tuning_method='optuna',
        n_trials=50,
        cv_folds=5
    )
    
    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 10), 
                     columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.random.randn(1000))
    
    # Train regression model
    trainer = ModelTrainerFactory.create_trainer('regression', config)
    results = trainer.train(X, y)
    
    print("Training completed!")
    print(f"Best parameters: {results['training_history']['best_params']}")

if __name__ == "__main__":
    main()