"""
Comprehensive Model Validation and Performance Metrics
Advanced evaluation system for all SFA ML models
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Evaluation Libraries
from sklearn.metrics import (
    # Regression metrics
    mean_absolute_error, mean_squared_error, r2_score, 
    mean_absolute_percentage_error, median_absolute_error,
    
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, 
    cohen_kappa_score, matthews_corrcoef,
    
    # Clustering metrics
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.model_selection import (
    cross_val_score, cross_validate, learning_curve,
    validation_curve, TimeSeriesSplit, StratifiedKFold
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
import shap
from scipy import stats
import joblib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    # General settings
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    
    # Evaluation options
    enable_cross_validation: bool = True
    enable_learning_curves: bool = True
    enable_feature_importance: bool = True
    enable_shap_analysis: bool = True
    enable_calibration_plots: bool = True
    
    # Plotting options
    save_plots: bool = True
    plot_dir: str = "./evaluation_plots"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    
    # Advanced analysis
    enable_statistical_tests: bool = True
    confidence_level: float = 0.95
    enable_residual_analysis: bool = True
    enable_bias_detection: bool = True

class ModelEvaluator:
    """Comprehensive model evaluation system"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluation_results = {}
        self.plots = {}
        
        # Create plot directory
        if self.config.save_plots:
            os.makedirs(self.config.plot_dir, exist_ok=True)
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      X_train: pd.DataFrame = None, y_train: pd.Series = None,
                      model_name: str = "model") -> Dict:
        """Comprehensive model evaluation"""
        logger.info(f"Evaluating {model_name}...")
        
        evaluation_results = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'test_samples': len(X_test),
                'train_samples': len(X_train) if X_train is not None else None,
                'features': X_test.shape[1],
                'feature_names': list(X_test.columns)
            }
        }
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Check if model supports probability prediction
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                pass
        
        # Determine model type
        model_type = self._determine_model_type(y_test, y_pred)
        evaluation_results['model_type'] = model_type
        
        # Evaluate based on model type
        if model_type == 'regression':
            evaluation_results.update(self._evaluate_regression(y_test, y_pred))
        elif model_type == 'classification':
            evaluation_results.update(self._evaluate_classification(y_test, y_pred, y_pred_proba))
        elif model_type == 'clustering':
            evaluation_results.update(self._evaluate_clustering(X_test, y_pred))
        
        # Cross-validation
        if self.config.enable_cross_validation and X_train is not None and y_train is not None:
            cv_results = self._perform_cross_validation(model, X_train, y_train, model_type)
            evaluation_results['cross_validation'] = cv_results
        
        # Feature importance analysis
        if self.config.enable_feature_importance:
            feature_importance = self._analyze_feature_importance(model, X_test, y_test)
            evaluation_results['feature_importance'] = feature_importance
        
        # SHAP analysis
        if self.config.enable_shap_analysis:
            try:
                shap_values = self._perform_shap_analysis(model, X_test, model_name)
                evaluation_results['shap_analysis'] = shap_values
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
        
        # Learning curves
        if self.config.enable_learning_curves and X_train is not None and y_train is not None:
            learning_curve_data = self._plot_learning_curves(model, X_train, y_train, model_name)
            evaluation_results['learning_curves'] = learning_curve_data
        
        # Model-specific advanced analysis
        if model_type == 'regression' and self.config.enable_residual_analysis:
            residual_analysis = self._analyze_residuals(y_test, y_pred, model_name)
            evaluation_results['residual_analysis'] = residual_analysis
        
        if model_type == 'classification' and self.config.enable_calibration_plots and y_pred_proba is not None:
            calibration_analysis = self._analyze_calibration(y_test, y_pred_proba, model_name)
            evaluation_results['calibration_analysis'] = calibration_analysis
        
        # Statistical tests
        if self.config.enable_statistical_tests:
            statistical_tests = self._perform_statistical_tests(y_test, y_pred, model_type)
            evaluation_results['statistical_tests'] = statistical_tests
        
        # Bias detection
        if self.config.enable_bias_detection:
            bias_analysis = self._detect_bias(y_test, y_pred, X_test)
            evaluation_results['bias_analysis'] = bias_analysis
        
        # Store results
        self.evaluation_results[model_name] = evaluation_results
        
        return evaluation_results
    
    def _determine_model_type(self, y_true, y_pred) -> str:
        """Determine if model is regression, classification, or clustering"""
        if len(np.unique(y_true)) < 20 and np.all(np.equal(np.mod(y_true, 1), 0)):
            return 'classification'
        elif hasattr(y_pred, 'dtype') and y_pred.dtype == 'int':
            return 'clustering'
        else:
            return 'regression'
    
    def _evaluate_regression(self, y_true, y_pred) -> Dict:
        """Comprehensive regression evaluation"""
        metrics = {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred)),
            'mape': float(mean_absolute_percentage_error(y_true, y_pred)) * 100,
            'median_ae': float(median_absolute_error(y_true, y_pred)),
            'explained_variance': float(1 - np.var(y_true - y_pred) / np.var(y_true))
        }
        
        # Additional regression metrics
        residuals = y_true - y_pred
        metrics.update({
            'mean_residual': float(np.mean(residuals)),
            'std_residual': float(np.std(residuals)),
            'max_residual': float(np.max(np.abs(residuals))),
            'residual_skewness': float(stats.skew(residuals)),
            'residual_kurtosis': float(stats.kurtosis(residuals))
        })
        
        return {'regression_metrics': metrics}
    
    def _evaluate_classification(self, y_true, y_pred, y_pred_proba=None) -> Dict:
        """Comprehensive classification evaluation"""
        # Basic metrics
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
            'matthews_corrcoef': float(matthews_corrcoef(y_true, y_pred))
        }
        
        # ROC AUC for binary/multiclass
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
                else:
                    metrics['roc_auc_ovr'] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr'))
                    metrics['roc_auc_ovo'] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovo'))
            except:
                pass
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        return {'classification_metrics': metrics}
    
    def _evaluate_clustering(self, X, labels) -> Dict:
        """Comprehensive clustering evaluation"""
        metrics = {}
        
        try:
            metrics['silhouette_score'] = float(silhouette_score(X, labels))
            metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(X, labels))
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(X, labels))
            metrics['n_clusters'] = int(len(np.unique(labels)))
            
            # Cluster statistics
            unique_labels = np.unique(labels)
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            metrics['cluster_sizes'] = cluster_sizes
            metrics['avg_cluster_size'] = float(np.mean(cluster_sizes))
            metrics['cluster_size_std'] = float(np.std(cluster_sizes))
            
        except Exception as e:
            logger.warning(f"Clustering evaluation failed: {e}")
            metrics['error'] = str(e)
        
        return {'clustering_metrics': metrics}
    
    def _perform_cross_validation(self, model, X, y, model_type) -> Dict:
        """Perform cross-validation analysis"""
        cv_results = {}
        
        try:
            if model_type == 'regression':
                scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
            else:
                scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            
            # Choose appropriate CV strategy
            if model_type == 'regression' and 'time' in str(X.columns).lower():
                cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            else:
                cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                                   random_state=self.config.random_state)
            
            cv_scores = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                                     return_train_score=True, n_jobs=-1)
            
            for metric in scoring:
                test_scores = cv_scores[f'test_{metric}']
                train_scores = cv_scores[f'train_{metric}']
                
                cv_results[metric] = {
                    'test_mean': float(np.mean(test_scores)),
                    'test_std': float(np.std(test_scores)),
                    'train_mean': float(np.mean(train_scores)),
                    'train_std': float(np.std(train_scores)),
                    'overfitting_score': float(np.mean(train_scores) - np.mean(test_scores))
                }
        
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            cv_results['error'] = str(e)
        
        return cv_results
    
    def _analyze_feature_importance(self, model, X, y) -> Dict:
        """Analyze feature importance using multiple methods"""
        importance_results = {}
        
        try:
            # Built-in feature importance
            if hasattr(model, 'feature_importances_'):
                importance_results['builtin_importance'] = {
                    'features': list(X.columns),
                    'importance': model.feature_importances_.tolist()
                }
            
            # Permutation importance
            perm_importance = permutation_importance(model, X, y, n_repeats=10, 
                                                   random_state=self.config.random_state)
            importance_results['permutation_importance'] = {
                'features': list(X.columns),
                'importance_mean': perm_importance.importances_mean.tolist(),
                'importance_std': perm_importance.importances_std.tolist()
            }
        
        except Exception as e:
            logger.warning(f"Feature importance analysis failed: {e}")
            importance_results['error'] = str(e)
        
        return importance_results
    
    def _perform_shap_analysis(self, model, X, model_name) -> Dict:
        """Perform SHAP analysis for model interpretability"""
        shap_results = {}
        
        try:
            # Sample data if too large
            if len(X) > 1000:
                sample_idx = np.random.choice(len(X), 1000, replace=False)
                X_sample = X.iloc[sample_idx]
            else:
                X_sample = X
            
            # Create explainer
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            
            # Global feature importance
            shap_results['global_importance'] = {
                'features': list(X.columns),
                'mean_abs_shap': np.mean(np.abs(shap_values.values), axis=0).tolist()
            }
            
            # Save SHAP plots
            if self.config.save_plots:
                # Summary plot
                plt.figure(figsize=self.config.figure_size)
                shap.summary_plot(shap_values, X_sample, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.plot_dir, f'{model_name}_shap_summary.png'), 
                           dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
                
                # Feature importance plot
                plt.figure(figsize=self.config.figure_size)
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.plot_dir, f'{model_name}_shap_importance.png'), 
                           dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
        
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            shap_results['error'] = str(e)
        
        return shap_results
    
    def _plot_learning_curves(self, model, X, y, model_name) -> Dict:
        """Generate learning curves"""
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=self.config.cv_folds,
                train_sizes=np.linspace(0.1, 1.0, 10),
                random_state=self.config.random_state, n_jobs=-1
            )
            
            if self.config.save_plots:
                plt.figure(figsize=self.config.figure_size)
                
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
                plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                               alpha=0.1, color='blue')
                
                plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
                plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                               alpha=0.1, color='red')
                
                plt.xlabel('Training Set Size')
                plt.ylabel('Score')
                plt.title(f'Learning Curves - {model_name}')
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plt.savefig(os.path.join(self.config.plot_dir, f'{model_name}_learning_curves.png'), 
                           dpi=self.config.dpi, bbox_inches='tight')
                plt.close()
            
            return {
                'train_sizes': train_sizes.tolist(),
                'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                'train_scores_std': np.std(train_scores, axis=1).tolist(),
                'val_scores_mean': np.mean(val_scores, axis=1).tolist(),
                'val_scores_std': np.std(val_scores, axis=1).tolist()
            }
        
        except Exception as e:
            logger.warning(f"Learning curves failed: {e}")
            return {'error': str(e)}
    
    def _analyze_residuals(self, y_true, y_pred, model_name) -> Dict:
        """Analyze residuals for regression models"""
        residuals = y_true - y_pred
        
        analysis = {
            'residual_stats': {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'skewness': float(stats.skew(residuals)),
                'kurtosis': float(stats.kurtosis(residuals)),
                'jarque_bera_stat': float(stats.jarque_bera(residuals)[0]),
                'jarque_bera_pvalue': float(stats.jarque_bera(residuals)[1])
            }
        }
        
        if self.config.save_plots:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Residuals vs Fitted
            axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_xlabel('Fitted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Fitted')
            
            # Q-Q plot
            stats.probplot(residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot')
            
            # Histogram of residuals
            axes[1, 0].hist(residuals, bins=30, alpha=0.7, density=True)
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Distribution of Residuals')
            
            # Scale-Location plot
            standardized_residuals = np.abs(residuals) / np.std(residuals)
            axes[1, 1].scatter(y_pred, standardized_residuals, alpha=0.6)
            axes[1, 1].set_xlabel('Fitted Values')
            axes[1, 1].set_ylabel('|Standardized Residuals|')
            axes[1, 1].set_title('Scale-Location Plot')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.plot_dir, f'{model_name}_residual_analysis.png'), 
                       dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
        
        return analysis
    
    def _analyze_calibration(self, y_true, y_pred_proba, model_name) -> Dict:
        """Analyze model calibration for classification"""
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_pred_proba[:, 1], n_bins=10
                )
                
                if self.config.save_plots:
                    plt.figure(figsize=self.config.figure_size)
                    plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                            label=f"{model_name}")
                    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                    plt.xlabel("Mean Predicted Probability")
                    plt.ylabel("Fraction of Positives")
                    plt.title("Calibration Plot")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    plt.savefig(os.path.join(self.config.plot_dir, f'{model_name}_calibration.png'), 
                               dpi=self.config.dpi, bbox_inches='tight')
                    plt.close()
                
                return {
                    'calibration_error': float(np.mean(np.abs(fraction_of_positives - mean_predicted_value))),
                    'brier_score': float(np.mean((y_pred_proba[:, 1] - y_true) ** 2))
                }
            
        except Exception as e:
            logger.warning(f"Calibration analysis failed: {e}")
            return {'error': str(e)}
    
    def _perform_statistical_tests(self, y_true, y_pred, model_type) -> Dict:
        """Perform statistical tests on predictions"""
        tests = {}
        
        try:
            if model_type == 'regression':
                # Durbin-Watson test for autocorrelation
                residuals = y_true - y_pred
                dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
                tests['durbin_watson'] = float(dw_stat)
                
                # Ljung-Box test
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_stat, lb_pvalue = acorr_ljungbox(residuals, lags=10, return_df=False)
                tests['ljung_box'] = {
                    'statistic': float(lb_stat.iloc[-1]),
                    'pvalue': float(lb_pvalue.iloc[-1])
                }
            
            elif model_type == 'classification':
                # McNemar's test (if applicable)
                # Chi-square test for independence
                from scipy.stats import chi2_contingency
                cm = confusion_matrix(y_true, y_pred)
                chi2, p_val, dof, expected = chi2_contingency(cm)
                tests['chi2_independence'] = {
                    'statistic': float(chi2),
                    'pvalue': float(p_val),
                    'degrees_of_freedom': int(dof)
                }
        
        except Exception as e:
            logger.warning(f"Statistical tests failed: {e}")
            tests['error'] = str(e)
        
        return tests
    
    def _detect_bias(self, y_true, y_pred, X_test) -> Dict:
        """Detect potential bias in model predictions"""
        bias_analysis = {}
        
        try:
            # Overall prediction bias
            if len(np.unique(y_true)) > 2:  # Regression or multiclass
                overall_bias = np.mean(y_pred - y_true)
                bias_analysis['overall_bias'] = float(overall_bias)
            
            # Feature-based bias analysis
            feature_bias = {}
            
            for column in X_test.select_dtypes(include=['object', 'category']).columns:
                if X_test[column].nunique() < 10:  # Categorical with few categories
                    bias_by_category = {}
                    for category in X_test[column].unique():
                        mask = X_test[column] == category
                        if np.sum(mask) > 10:  # Sufficient samples
                            if len(np.unique(y_true)) == 2:  # Binary classification
                                pred_rate = np.mean(y_pred[mask])
                                true_rate = np.mean(y_true[mask])
                                bias_by_category[str(category)] = {
                                    'predicted_rate': float(pred_rate),
                                    'true_rate': float(true_rate),
                                    'bias': float(pred_rate - true_rate)
                                }
                            else:  # Regression
                                pred_mean = np.mean(y_pred[mask])
                                true_mean = np.mean(y_true[mask])
                                bias_by_category[str(category)] = {
                                    'predicted_mean': float(pred_mean),
                                    'true_mean': float(true_mean),
                                    'bias': float(pred_mean - true_mean)
                                }
                    
                    if bias_by_category:
                        feature_bias[column] = bias_by_category
            
            bias_analysis['feature_bias'] = feature_bias
        
        except Exception as e:
            logger.warning(f"Bias detection failed: {e}")
            bias_analysis['error'] = str(e)
        
        return bias_analysis
    
    def generate_report(self, model_name: str = None) -> str:
        """Generate comprehensive evaluation report"""
        if model_name and model_name in self.evaluation_results:
            results = {model_name: self.evaluation_results[model_name]}
        else:
            results = self.evaluation_results
        
        report = []
        report.append("=" * 80)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for name, result in results.items():
            report.append(f"Model: {name}")
            report.append("-" * 50)
            report.append(f"Model Type: {result['model_type']}")
            report.append(f"Test Samples: {result['dataset_info']['test_samples']}")
            report.append(f"Features: {result['dataset_info']['features']}")
            report.append("")
            
            # Main metrics
            if result['model_type'] == 'regression':
                metrics = result['regression_metrics']
                report.append("Regression Metrics:")
                report.append(f"  R²: {metrics['r2']:.4f}")
                report.append(f"  RMSE: {metrics['rmse']:.4f}")
                report.append(f"  MAE: {metrics['mae']:.4f}")
                report.append(f"  MAPE: {metrics['mape']:.2f}%")
            
            elif result['model_type'] == 'classification':
                metrics = result['classification_metrics']
                report.append("Classification Metrics:")
                report.append(f"  Accuracy: {metrics['accuracy']:.4f}")
                report.append(f"  F1 (Macro): {metrics['f1_macro']:.4f}")
                report.append(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
                report.append(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
            
            elif result['model_type'] == 'clustering':
                metrics = result['clustering_metrics']
                report.append("Clustering Metrics:")
                report.append(f"  Silhouette Score: {metrics.get('silhouette_score', 'N/A')}")
                report.append(f"  Number of Clusters: {metrics.get('n_clusters', 'N/A')}")
            
            # Cross-validation results
            if 'cross_validation' in result:
                report.append("\nCross-Validation Results:")
                cv = result['cross_validation']
                for metric, scores in cv.items():
                    if isinstance(scores, dict):
                        report.append(f"  {metric}: {scores['test_mean']:.4f} ± {scores['test_std']:.4f}")
            
            report.append("\n" + "=" * 80)
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, filepath: str):
        """Save evaluation results to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2, default=str)
            logger.info(f"Evaluation results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def load_results(self, filepath: str) -> Dict:
        """Load evaluation results from JSON file"""
        try:
            with open(filepath, 'r') as f:
                self.evaluation_results = json.load(f)
            logger.info(f"Evaluation results loaded from {filepath}")
            return self.evaluation_results
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return {}
    
    def compare_models(self, models_results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple models performance"""
        comparison_data = []
        
        for model_name, results in models_results.items():
            model_type = results.get('model_type', 'unknown')
            row = {'Model': model_name, 'Type': model_type}
            
            if model_type == 'regression':
                metrics = results.get('regression_metrics', {})
                row.update({
                    'R²': metrics.get('r2', np.nan),
                    'RMSE': metrics.get('rmse', np.nan),
                    'MAE': metrics.get('mae', np.nan),
                    'MAPE': metrics.get('mape', np.nan)
                })
            
            elif model_type == 'classification':
                metrics = results.get('classification_metrics', {})
                row.update({
                    'Accuracy': metrics.get('accuracy', np.nan),
                    'F1_Macro': metrics.get('f1_macro', np.nan),
                    'Precision_Macro': metrics.get('precision_macro', np.nan),
                    'Recall_Macro': metrics.get('recall_macro', np.nan),
                    'ROC_AUC': metrics.get('roc_auc', np.nan)
                })
            
            elif model_type == 'clustering':
                metrics = results.get('clustering_metrics', {})
                row.update({
                    'Silhouette_Score': metrics.get('silhouette_score', np.nan),
                    'Calinski_Harabasz': metrics.get('calinski_harabasz_score', np.nan),
                    'Davies_Bouldin': metrics.get('davies_bouldin_score', np.nan),
                    'N_Clusters': metrics.get('n_clusters', np.nan)
                })
            
            # Add cross-validation info if available
            if 'cross_validation' in results:
                cv = results['cross_validation']
                if model_type == 'regression' and 'r2' in cv:
                    row['CV_R²_Mean'] = cv['r2'].get('test_mean', np.nan)
                    row['CV_R²_Std'] = cv['r2'].get('test_std', np.nan)
                elif model_type == 'classification' and 'accuracy' in cv:
                    row['CV_Accuracy_Mean'] = cv['accuracy'].get('test_mean', np.nan)
                    row['CV_Accuracy_Std'] = cv['accuracy'].get('test_std', np.nan)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, metric: str, 
                            title: str = None) -> None:
        """Plot model comparison for a specific metric"""
        if metric not in comparison_df.columns:
            logger.warning(f"Metric {metric} not found in comparison data")
            return
        
        plt.figure(figsize=self.config.figure_size)
        
        # Filter out NaN values
        valid_data = comparison_df.dropna(subset=[metric])
        
        if len(valid_data) == 0:
            logger.warning(f"No valid data for metric {metric}")
            return
        
        # Create bar plot
        bars = plt.bar(valid_data['Model'], valid_data[metric])
        
        # Color bars based on performance (higher is better for most metrics)
        if metric not in ['davies_bouldin_score', 'mse', 'rmse', 'mae', 'mape']:
            colors = plt.cm.RdYlGn(valid_data[metric] / valid_data[metric].max())
        else:
            colors = plt.cm.RdYlGn_r(valid_data[metric] / valid_data[metric].max())
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Models')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(title or f'Model Comparison - {metric.replace("_", " ").title()}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(os.path.join(self.config.plot_dir, f'model_comparison_{metric}.png'),
                       dpi=self.config.dpi, bbox_inches='tight')
        
        plt.show()
    
    def generate_model_ranking(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        """Generate model ranking based on multiple metrics"""
        ranking_data = []
        
        for _, row in comparison_df.iterrows():
            model_name = row['Model']
            model_type = row['Type']
            
            scores = []
            weights = []
            
            if model_type == 'regression':
                # Higher is better: R²
                if not pd.isna(row.get('R²')):
                    scores.append(row['R²'])
                    weights.append(0.4)
                
                # Lower is better: RMSE, MAE, MAPE (invert by taking 1/(1+value))
                for metric in ['RMSE', 'MAE', 'MAPE']:
                    if not pd.isna(row.get(metric)):
                        scores.append(1 / (1 + row[metric]))
                        weights.append(0.2)
            
            elif model_type == 'classification':
                # Higher is better: Accuracy, F1, Precision, Recall, ROC_AUC
                for metric in ['Accuracy', 'F1_Macro', 'Precision_Macro', 'Recall_Macro', 'ROC_AUC']:
                    if not pd.isna(row.get(metric)):
                        scores.append(row[metric])
                        weights.append(0.2)
            
            elif model_type == 'clustering':
                # Higher is better: Silhouette, Calinski_Harabasz
                for metric in ['Silhouette_Score', 'Calinski_Harabasz']:
                    if not pd.isna(row.get(metric)):
                        scores.append(row[metric])
                        weights.append(0.4)
                
                # Lower is better: Davies_Bouldin
                if not pd.isna(row.get('Davies_Bouldin')):
                    scores.append(1 / (1 + row['Davies_Bouldin']))
                    weights.append(0.2)
            
            # Calculate weighted average score
            if scores and weights:
                weights = np.array(weights)
                weights = weights / np.sum(weights)  # Normalize weights
                composite_score = np.average(scores, weights=weights)
            else:
                composite_score = 0
            
            ranking_data.append({
                'Model': model_name,
                'Type': model_type,
                'Composite_Score': composite_score,
                'Num_Metrics': len(scores)
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('Composite_Score', ascending=False)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        
        return ranking_df[['Rank', 'Model', 'Type', 'Composite_Score', 'Num_Metrics']]
    
    def export_detailed_report(self, filepath: str, include_plots: bool = True):
        """Export detailed HTML report with plots and analysis"""
        try:
            html_content = []
            html_content.append("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Evaluation Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .header { background-color: #f0f8ff; padding: 20px; border-radius: 10px; }
                    .model-section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 10px; }
                    .metrics-table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                    .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                    .metrics-table th { background-color: #f2f2f2; }
                    .plot-container { text-align: center; margin: 20px 0; }
                    .warning { color: #ff6b6b; font-style: italic; }
                    .success { color: #51cf66; font-weight: bold; }
                </style>
            </head>
            <body>
            """)
            
            # Header
            html_content.append(f"""
            <div class="header">
                <h1>Model Evaluation Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Total Models Evaluated:</strong> {len(self.evaluation_results)}</p>
            </div>
            """)
            
            # Model comparison table
            if len(self.evaluation_results) > 1:
                comparison_df = self.compare_models(self.evaluation_results)
                html_content.append("<h2>Model Comparison Overview</h2>")
                html_content.append(comparison_df.to_html(classes='metrics-table', escape=False))
                
                # Model ranking
                ranking_df = self.generate_model_ranking(comparison_df)
                html_content.append("<h2>Model Ranking</h2>")
                html_content.append(ranking_df.to_html(classes='metrics-table', escape=False))
            
            # Detailed results for each model
            html_content.append("<h2>Detailed Model Results</h2>")
            
            for model_name, results in self.evaluation_results.items():
                html_content.append(f'<div class="model-section">')
                html_content.append(f'<h3>{model_name}</h3>')
                html_content.append(f'<p><strong>Model Type:</strong> {results["model_type"]}</p>')
                html_content.append(f'<p><strong>Test Samples:</strong> {results["dataset_info"]["test_samples"]}</p>')
                html_content.append(f'<p><strong>Features:</strong> {results["dataset_info"]["features"]}</p>')
                
                # Main metrics
                model_type = results['model_type']
                if f'{model_type}_metrics' in results:
                    metrics = results[f'{model_type}_metrics']
                    html_content.append(f'<h4>{model_type.title()} Metrics</h4>')
                    
                    metrics_html = '<table class="metrics-table"><tr><th>Metric</th><th>Value</th></tr>'
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            metrics_html += f'<tr><td>{metric.replace("_", " ").title()}</td><td>{value:.6f}</td></tr>'
                    metrics_html += '</table>'
                    html_content.append(metrics_html)
                
                # Cross-validation results
                if 'cross_validation' in results:
                    html_content.append('<h4>Cross-Validation Results</h4>')
                    cv_html = '<table class="metrics-table"><tr><th>Metric</th><th>Test Mean</th><th>Test Std</th><th>Train Mean</th><th>Overfitting</th></tr>'
                    
                    for metric, scores in results['cross_validation'].items():
                        if isinstance(scores, dict) and 'test_mean' in scores:
                            cv_html += f'''<tr>
                                <td>{metric.replace("_", " ").title()}</td>
                                <td>{scores["test_mean"]:.6f}</td>
                                <td>{scores["test_std"]:.6f}</td>
                                <td>{scores.get("train_mean", "N/A")}</td>
                                <td>{scores.get("overfitting_score", "N/A")}</td>
                            </tr>'''
                    cv_html += '</table>'
                    html_content.append(cv_html)
                
                # Feature importance
                if 'feature_importance' in results:
                    html_content.append('<h4>Top 10 Important Features</h4>')
                    fi = results['feature_importance']
                    
                    if 'permutation_importance' in fi:
                        features = fi['permutation_importance']['features']
                        importance = fi['permutation_importance']['importance_mean']
                        
                        # Get top 10 features
                        top_indices = np.argsort(importance)[-10:][::-1]
                        
                        fi_html = '<table class="metrics-table"><tr><th>Feature</th><th>Importance</th></tr>'
                        for idx in top_indices:
                            fi_html += f'<tr><td>{features[idx]}</td><td>{importance[idx]:.6f}</td></tr>'
                        fi_html += '</table>'
                        html_content.append(fi_html)
                
                html_content.append('</div>')
            
            html_content.append("""
            </body>
            </html>
            """)
            
            # Write HTML file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_content))
            
            logger.info(f"Detailed HTML report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export detailed report: {e}")


class AdvancedModelEvaluator(ModelEvaluator):
    """Extended evaluator with advanced analysis capabilities"""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.ensemble_results = {}
        self.time_series_results = {}
    
    def evaluate_ensemble(self, models: List, model_names: List[str], 
                         X_test: pd.DataFrame, y_test: pd.Series,
                         ensemble_method: str = 'voting') -> Dict:
        """Evaluate ensemble of models"""
        logger.info(f"Evaluating ensemble with {len(models)} models...")
        
        # Get individual predictions
        predictions = []
        for model in models:
            pred = model.predict(X_test)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Create ensemble prediction
        if ensemble_method == 'voting':
            if len(np.unique(y_test)) < 20:  # Classification
                ensemble_pred = stats.mode(predictions, axis=0)[0].flatten()
            else:  # Regression
                ensemble_pred = np.mean(predictions, axis=0)
        elif ensemble_method == 'weighted_average':
            # Weight by individual model performance (simplified)
            weights = []
            for i, model in enumerate(models):
                pred = predictions[i]
                if len(np.unique(y_test)) < 20:  # Classification
                    score = accuracy_score(y_test, pred)
                else:  # Regression
                    score = r2_score(y_test, pred)
                weights.append(max(score, 0.01))  # Avoid zero weights
            
            weights = np.array(weights) / np.sum(weights)
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        # Evaluate ensemble
        ensemble_name = f"Ensemble_{ensemble_method}"
        ensemble_results = self.evaluate_model(
            type('EnsembleModel', (), {
                'predict': lambda self, X: ensemble_pred
            })(), 
            X_test, y_test, model_name=ensemble_name
        )
        
        # Add individual model comparisons
        ensemble_results['individual_models'] = {}
        for i, name in enumerate(model_names):
            individual_results = self.evaluate_model(
                models[i], X_test, y_test, model_name=name
            )
            ensemble_results['individual_models'][name] = individual_results
        
        self.ensemble_results[ensemble_name] = ensemble_results
        return ensemble_results
    
    def evaluate_time_series_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                                  time_column: str = None, forecast_horizon: int = 30) -> Dict:
        """Specialized evaluation for time series models"""
        logger.info("Performing time series model evaluation...")
        
        # Basic evaluation
        results = self.evaluate_model(model, X_test, y_test, model_name="TimeSeries_Model")
        
        # Time series specific metrics
        predictions = model.predict(X_test)
        
        # Directional accuracy
        if len(y_test) > 1:
            y_diff = np.diff(y_test)
            pred_diff = np.diff(predictions)
            directional_accuracy = np.mean(np.sign(y_diff) == np.sign(pred_diff))
            results['time_series_metrics'] = {
                'directional_accuracy': float(directional_accuracy)
            }
        
        # Forecast evaluation (if applicable)
        if hasattr(model, 'forecast') and forecast_horizon > 0:
            try:
                forecast = model.forecast(steps=forecast_horizon)
                results['forecast_info'] = {
                    'horizon': forecast_horizon,
                    'forecast_mean': float(np.mean(forecast)),
                    'forecast_std': float(np.std(forecast))
                }
            except:
                pass
        
        self.time_series_results["TimeSeries_Model"] = results
        return results


# Utility functions for evaluation
def quick_evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, 
                  model_name: str = "QuickEval") -> Dict:
    """Quick model evaluation with default settings"""
    config = EvaluationConfig(
        enable_shap_analysis=False,
        enable_learning_curves=False,
        save_plots=False
    )
    
    evaluator = ModelEvaluator(config)
    return evaluator.evaluate_model(model, X_test, y_test, model_name=model_name)


def evaluate_multiple_models(models: Dict, X_test: pd.DataFrame, y_test: pd.Series,
                            config: EvaluationConfig = None) -> pd.DataFrame:
    """Evaluate multiple models and return comparison DataFrame"""
    if config is None:
        config = EvaluationConfig()
    
    evaluator = ModelEvaluator(config)
    results = {}
    
    for name, model in models.items():
        results[name] = evaluator.evaluate_model(model, X_test, y_test, model_name=name)
    
    return evaluator.compare_models(results)


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression, make_classification
    
    # Example with regression
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(20)])
    y_reg = pd.Series(y_reg)
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Train model
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train_reg, y_train_reg)
    
    # Evaluate
    config = EvaluationConfig()
    evaluator = ModelEvaluator(config)
    
    results = evaluator.evaluate_model(
        rf_reg, X_test_reg, y_test_reg, 
        X_train_reg, y_train_reg,
        model_name="RandomForest_Regression"
    )
    
    # Generate report
    print(evaluator.generate_report())
    
    # Save results
    evaluator.save_results("evaluation_results.json")
    evaluator.export_detailed_report("evaluation_report.html")