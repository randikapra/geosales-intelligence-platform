# # backend/schemas/predictions.py
# """
# Prediction schemas
# """
# from pydantic import BaseModel
# from typing import Optional, List, Dict, Any
# from datetime import datetime


# class PredictionRequest(BaseModel):
#     territory: Optional[str] = None
#     dealer_code: Optional[str] = None
#     forecast_days: int = 30
#     confidence_level: float = 0.95


# class PredictionResponse(BaseModel):
#     prediction_id: str
#     model_version: str
#     confidence_score: float
#     generated_at: datetime


# class SalesForecast(BaseModel):
#     date: datetime
#     predicted_sales: float
#     confidence_interval_lower: float
#     confidence_interval_upper: float
#     territory_code: Optional[str] = None


# class DemandPrediction(BaseModel):
#     product_category: str
#     territory_code: str
#     predicted_demand: float
#     confidence_score: float
#     forecast_date: datetime


# class ChurnPrediction(BaseModel):
#     customer_code: str
#     churn_probability: float
#     risk_level: str  # low, medium, high
#     factors: List[str]
#     recommendation: str


# class RoutePrediction(BaseModel):
#     dealer_code: str
#     date: datetime
#     optimized_route: List[Dict[str, Any]]
#     estimated_distance: float
#     estimated_time: int  # minutes
#     potential_sales: float
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

class PredictionType(str, Enum):
    SALES_FORECAST = "sales_forecast"
    DEMAND_PREDICTION = "demand_prediction"
    CUSTOMER_CHURN = "customer_churn"
    DEALER_PERFORMANCE = "dealer_performance"
    INVENTORY_OPTIMIZATION = "inventory_optimization"
    ROUTE_OPTIMIZATION = "route_optimization"
    MARKET_EXPANSION = "market_expansion"
    PRICE_OPTIMIZATION = "price_optimization"

class ModelType(str, Enum):
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    ARIMA = "arima"
    LSTM = "lstm"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"

class PredictionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class PredictionBase(BaseModel):
    prediction_type: PredictionType
    model_type: ModelType
    target_variable: str
    features: List[str]
    prediction_horizon: int  # days
    confidence_level: float = Field(default=0.95, ge=0.1, le=0.99)

class SalesForecastRequest(PredictionBase):
    prediction_type: PredictionType = PredictionType.SALES_FORECAST
    dealer_code: Optional[str] = None
    customer_code: Optional[str] = None
    territory_code: Optional[str] = None
    product_category: Optional[str] = None
    include_seasonality: bool = True
    include_trends: bool = True
    external_factors: Dict[str, Any] = Field(default_factory=dict)

class DemandPredictionRequest(PredictionBase):
    prediction_type: PredictionType = PredictionType.DEMAND_PREDICTION
    product_codes: List[str]
    location_codes: List[str]
    include_promotions: bool = True
    include_weather: bool = False
    include_events: bool = False

class CustomerChurnRequest(PredictionBase):
    prediction_type: PredictionType = PredictionType.CUSTOMER_CHURN
    customer_codes: Optional[List[str]] = None
    risk_threshold: float = Field(default=0.7, ge=0.1, le=1.0)
    include_behavioral_features: bool = True
    include_transaction_features: bool = True

class DealerPerformancePredictionRequest(PredictionBase):
    prediction_type: PredictionType = PredictionType.DEALER_PERFORMANCE
    dealer_codes: Optional[List[str]] = None
    performance_metrics: List[str] = Field(default=["sales", "orders", "customers"])
    include_historical_performance: bool = True
    include_territory_factors: bool = True

class PredictionResult(BaseModel):
    prediction_id: str
    prediction_type: PredictionType
    model_type: ModelType
    status: PredictionStatus
    predicted_values: List[Dict[str, Any]]
    confidence_intervals: List[Dict[str, float]]
    feature_importance: Dict[str, float]
    model_accuracy: Dict[str, float]
    created_at: datetime
    completed_at: Optional[datetime] = None

class SalesForecastResult(PredictionResult):
    forecasted_sales: List[Dict[str, Any]]  # date, predicted_value, confidence_lower, confidence_upper
    seasonal_components: Optional[Dict[str, List[float]]] = None
    trend_components: Optional[List[float]] = None
    total_forecast: Decimal
    growth_rate: float

class DemandPredictionResult(PredictionResult):
    demand_forecast: List[Dict[str, Any]]  # product_code, location_code, predicted_demand, date
    inventory_recommendations: List[Dict[str, Any]]
    stockout_risk: Dict[str, float]
    overstock_risk: Dict[str, float]

class CustomerChurnResult(PredictionResult):
    churn_probabilities: List[Dict[str, Any]]  # customer_code, churn_probability, risk_level
    high_risk_customers: List[str]
    retention_recommendations: List[Dict[str, Any]]
    churn_factors: Dict[str, float]

class DealerPerformanceResult(PredictionResult):
    performance_predictions: List[Dict[str, Any]]  # dealer_code, metric, predicted_value, confidence
    improvement_opportunities: List[Dict[str, Any]]
    training_recommendations: List[Dict[str, Any]]
    territory_optimization: Optional[Dict[str, Any]] = None

class ModelTrainingRequest(BaseModel):
    model_name: str
    model_type: ModelType
    prediction_type: PredictionType
    training_data_period: Dict[str, date]  # start_date, end_date
    features: List[str]
    target_variable: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    cross_validation_folds: int = Field(default=5, ge=2, le=10)
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)

class ModelTrainingResult(BaseModel):
    model_id: str
    model_name: str
    model_type: ModelType
    training_status: str
    accuracy_metrics: Dict[str, float]
    validation_scores: List[float]
    feature_importance: Dict[str, float]
    training_time: float  # seconds
    model_size: int  # bytes
    hyperparameters_used: Dict[str, Any]
    created_at: datetime

class ModelEvaluationMetrics(BaseModel):
    model_id: str
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    r2_score: float  # R-squared
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None

class PredictionBatch(BaseModel):
    batch_id: str
    prediction_requests: List[Union[SalesForecastRequest, DemandPredictionRequest, CustomerChurnRequest, DealerPerformancePredictionRequest]]
    priority: int = Field(default=1, ge=1, le=5)
    callback_url: Optional[str] = None

class PredictionBatchResult(BaseModel):
    batch_id: str
    total_predictions: int
    completed_predictions: int
    failed_predictions: int
    results: List[PredictionResult]
    batch_status: PredictionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None

class ModelComparisonRequest(BaseModel):
    model_ids: List[str] = Field(..., min_items=2, max_items=10)
    evaluation_period: Dict[str, date]
    comparison_metrics: List[str] = Field(default=["accuracy", "precision", "recall"])

class ModelComparisonResult(BaseModel):
    comparison_id: str
    models_compared: List[str]
    best_model: str
    performance_comparison: Dict[str, Dict[str, float]]
    recommendations: List[str]
    detailed_results: List[ModelEvaluationMetrics]

class AutoMLRequest(BaseModel):
    dataset_name: str
    target_variable: str
    prediction_type: PredictionType
    max_training_time: int = Field(default=3600, ge=300, le=86400)  # seconds
    optimization_metric: str = "accuracy"
    feature_selection: bool = True
    hyperparameter_tuning: bool = True

class AutoMLResult(BaseModel):
    experiment_id: str
    best_model: ModelTrainingResult
    model_leaderboard: List[Dict[str, Any]]
    feature_engineering_summary: Dict[str, Any]
    optimization_history: List[Dict[str, Any]]
    recommendations: List[str]

class PredictionExplanation(BaseModel):
    prediction_id: str
    instance_id: str
    predicted_value: float
    confidence: float
    feature_contributions: Dict[str, float]
    similar_instances: List[Dict[str, Any]]
    explanation_text: str
    visualization_data: Optional[Dict[str, Any]] = None

class ModelDeploymentRequest(BaseModel):
    model_id: str
    deployment_name: str
    environment: str = Field(default="production")
    auto_scaling: bool = True
    monitoring_enabled: bool = True
    performance_threshold: Dict[str, float] = Field(default_factory=dict)

class ModelDeploymentStatus(BaseModel):
    deployment_id: str
    model_id: str
    deployment_name: str
    status: str
    endpoint_url: Optional[str] = None
    health_check: Dict[str, Any]
    performance_metrics: Dict[str, float]
    deployed_at: datetime
    last_updated: datetime

class PredictionMonitoring(BaseModel):
    model_id: str
    monitoring_period: Dict[str, date]
    prediction_count: int
    average_confidence: float
    accuracy_drift: float
    feature_drift: Dict[str, float]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]

class RealTimePredictionRequest(BaseModel):
    model_id: str
    input_features: Dict[str, Any]
    return_explanation: bool = False
    confidence_threshold: float = Field(default=0.8, ge=0.1, le=1.0)

class RealTimePredictionResponse(BaseModel):
    prediction_id: str
    predicted_value: Union[float, str, List[float]]
    confidence: float
    processing_time_ms: float
    model_version: str
    explanation: Optional[PredictionExplanation] = None
    warnings: List[str] = Field(default_factory=list)

class PredictionPipeline(BaseModel):
    pipeline_id: str
    pipeline_name: str
    steps: List[Dict[str, Any]]
    schedule: str  # cron expression
    is_active: bool
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None

class PredictionAudit(BaseModel):
    audit_id: str
    prediction_id: str
    model_id: str
    user_id: str
    action: str
    timestamp: datetime
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)