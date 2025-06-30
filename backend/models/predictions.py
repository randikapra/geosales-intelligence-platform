"""
Predictions model for storing ML model predictions, forecasts, and confidence scores.
Supports various prediction types including sales forecasting, demand prediction,
customer segmentation, and route optimization.
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON, Enum, Index
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func
from enum import Enum as PyEnum
import json

from .base import BaseModel


class PredictionType(PyEnum):
    """Enumeration for different types of predictions"""
    SALES_FORECAST = "sales_forecast"
    DEMAND_FORECAST = "demand_forecast"
    CUSTOMER_CHURN = "customer_churn"
    CUSTOMER_SEGMENTATION = "customer_segmentation"
    DEALER_PERFORMANCE = "dealer_performance"
    ROUTE_OPTIMIZATION = "route_optimization"
    TERRITORY_OPTIMIZATION = "territory_optimization"
    SALES_ANOMALY = "sales_anomaly"
    ROUTE_ANOMALY = "route_anomaly"
    MARKET_OPPORTUNITY = "market_opportunity"
    DEMAND_CLUSTERING = "demand_clustering"


class PredictionStatus(PyEnum):
    """Status of prediction"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class Prediction(BaseModel):
    """
    Model for storing ML predictions and forecasts
    """
    __tablename__ = 'predictions'
    
    # Prediction Metadata
    prediction_type = Column(Enum(PredictionType), nullable=False, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    status = Column(Enum(PredictionStatus), default=PredictionStatus.PENDING, nullable=False)
    
    # Prediction Target Information
    target_entity_type = Column(String(50), nullable=False)  # customer, dealer, route, territory
    target_entity_id = Column(Integer, nullable=True, index=True)  # ID of target entity
    target_date = Column(DateTime(timezone=True), nullable=True, index=True)  # When prediction is for
    
    # Prediction Results
    prediction_value = Column(Float, nullable=True)  # Main prediction value
    confidence_score = Column(Float, nullable=False)  # Confidence 0-1
    probability_scores = Column(JSON, nullable=True)  # Class probabilities for classification
    
    # Detailed Results
    prediction_data = Column(JSON, nullable=False)  # Detailed prediction results
    feature_importance = Column(JSON, nullable=True)  # Feature importance scores
    explanation = Column(Text, nullable=True)  # Human-readable explanation
    
    # Time Information
    prediction_horizon = Column(Integer, nullable=True)  # Prediction horizon in days
    valid_from = Column(DateTime(timezone=True), nullable=False, default=func.now())
    valid_until = Column(DateTime(timezone=True), nullable=True)
    predicted_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Model Performance
    model_accuracy = Column(Float, nullable=True)  # Model accuracy score
    model_metrics = Column(JSON, nullable=True)  # Additional model metrics
    
    # Business Context
    business_impact = Column(String(20), nullable=True)  # high, medium, low
    action_required = Column(Boolean, default=False, nullable=False)
    alert_threshold_breached = Column(Boolean, default=False, nullable=False)
    
    # Input Data References
    input_data_hash = Column(String(64), nullable=True)  # Hash of input data
    input_features = Column(JSON, nullable=True)  # Input feature values
    data_sources = Column(JSON, nullable=True)  # Data sources used
    
    # Validation and Feedback
    actual_value = Column(Float, nullable=True)  # Actual outcome for validation
    accuracy_validated = Column(Boolean, nullable=True)
    user_feedback = Column(Text, nullable=True)
    feedback_score = Column(Integer, nullable=True)  # 1-5 rating
    
    # Relationships
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=True, index=True)
    dealer_id = Column(Integer, ForeignKey('dealers.id'), nullable=True, index=True)
    
    customer = relationship("Customer", back_populates="predictions")
    dealer = relationship("Dealer", back_populates="predictions")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_pred_type_date', 'prediction_type', 'target_date'),
        Index('idx_pred_entity', 'target_entity_type', 'target_entity_id'),
        Index('idx_pred_model_version', 'model_name', 'model_version'),
        Index('idx_pred_status_date', 'status', 'predicted_at'),
        Index('idx_pred_valid_period', 'valid_from', 'valid_until'),
        Index('idx_pred_confidence', 'confidence_score', 'business_impact'),
    )
    
    @hybrid_property
    def is_valid(self) -> bool:
        """Check if prediction is still valid"""
        now = datetime.utcnow()
        return (self.valid_from <= now and 
                (self.valid_until is None or self.valid_until > now))
    
    @hybrid_property
    def is_high_confidence(self) -> bool:
        """Check if prediction has high confidence"""
        return self.confidence_score >= 0.8
    
    @hybrid_property
    def days_until_target(self) -> Optional[int]:
        """Calculate days until target date"""
        if not self.target_date:
            return None
        delta = self.target_date - datetime.utcnow()
        return delta.days
    
    @hybrid_property
    def age_hours(self) -> float:
        """Get age of prediction in hours"""
        delta = datetime.utcnow() - self.predicted_at
        return delta.total_seconds() / 3600
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of prediction results"""
        return {
            'prediction_type': self.prediction_type.value,
            'model_name': self.model_name,
            'target_entity': f"{self.target_entity_type}_{self.target_entity_id}",
            'prediction_value': self.prediction_value,
            'confidence_score': self.confidence_score,
            'business_impact': self.business_impact,
            'status': self.status.value,
            'predicted_at': self.predicted_at.isoformat(),
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
            'is_valid': self.is_valid,
            'action_required': self.action_required
        }
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """Get detailed prediction results"""
        return {
            **self.get_prediction_summary(),
            'prediction_data': self.prediction_data,
            'feature_importance': self.feature_importance,
            'probability_scores': self.probability_scores,
            'model_metrics': self.model_metrics,
            'input_features': self.input_features,
            'explanation': self.explanation
        }
    
    def validate_prediction(self, actual_value: float) -> Dict[str, Any]:
        """
        Validate prediction against actual outcome
        Returns validation metrics
        """
        if self.prediction_value is None:
            return {'error': 'No prediction value to validate'}
        
        self.actual_value = actual_value
        
        # Calculate accuracy metrics
        absolute_error = abs(self.prediction_value - actual_value)
        relative_error = absolute_error / max(abs(actual_value), 1e-6)
        percentage_error = relative_error * 100
        
        # Determine if prediction was accurate within threshold
        accuracy_threshold = 0.1  # 10% threshold
        self.accuracy_validated = relative_error <= accuracy_threshold
        
        validation_metrics = {
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'percentage_error': percentage_error,
            'is_accurate': self.accuracy_validated,
            'predicted_value': self.prediction_value,
            'actual_value': actual_value,
            'confidence_score': self.confidence_score
        }
        
        return validation_metrics
    
    def update_confidence(self, new_confidence: float, reason: str = None):
        """Update confidence score with reason"""
        old_confidence = self.confidence_score
        self.confidence_score = max(0.0, min(1.0, new_confidence))
        
        # Log confidence change
        if self.prediction_data is None:
            self.prediction_data = {}
        
        if 'confidence_history' not in self.prediction_data:
            self.prediction_data['confidence_history'] = []
        
        self.prediction_data['confidence_history'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'old_confidence': old_confidence,
            'new_confidence': self.confidence_score,
            'reason': reason
        })
    
    def add_feedback(self, feedback_text: str, score: int = None):
        """Add user feedback to prediction"""
        self.user_feedback = feedback_text
        if score is not None:
            self.feedback_score = max(1, min(5, score))
    
    def is_expired(self) -> bool:
        """Check if prediction has expired"""
        if self.valid_until:
            return datetime.utcnow() > self.valid_until
        return False
    
    def expire_prediction(self):
        """Mark prediction as expired"""
        self.status = PredictionStatus.EXPIRED
        self.valid_until = datetime.utcnow()
    
    @classmethod
    def get_active_predictions(cls, session, prediction_type: PredictionType = None,
                             entity_type: str = None, entity_id: int = None) -> List['Prediction']:
        """Get active predictions with optional filters"""
        query = session.query(cls).filter(
            cls.status == PredictionStatus.COMPLETED,
            cls.valid_from <= func.now(),
            (cls.valid_until.is_(None) | (cls.valid_until > func.now()))
        )
        
        if prediction_type:
            query = query.filter(cls.prediction_type == prediction_type)
        
        if entity_type:
            query = query.filter(cls.target_entity_type == entity_type)
            
        if entity_id:
            query = query.filter(cls.target_entity_id == entity_id)
        
        return query.order_by(cls.predicted_at.desc()).all()
    
    @classmethod
    def get_high_impact_predictions(cls, session, hours: int = 24) -> List['Prediction']:
        """Get high-impact predictions from recent hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return session.query(cls).filter(
            cls.business_impact == 'high',
            cls.confidence_score >= 0.7,
            cls.predicted_at >= cutoff_time,
            cls.status == PredictionStatus.COMPLETED
        ).order_by(cls.confidence_score.desc()).all()
    
    @classmethod
    def get_predictions_needing_action(cls, session) -> List['Prediction']:
        """Get predictions that require business action"""
        return session.query(cls).filter(
            cls.action_required == True,
            cls.status == PredictionStatus.COMPLETED,
            cls.valid_from <= func.now(),
            (cls.valid_until.is_(None) | (cls.valid_until > func.now()))
        ).order_by(cls.confidence_score.desc()).all()
    
    @classmethod
    def get_model_performance_stats(cls, session, model_name: str, 
                                  days: int = 30) -> Dict[str, Any]:
        """Get performance statistics for a specific model"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        predictions = session.query(cls).filter(
            cls.model_name == model_name,
            cls.predicted_at >= cutoff_date,
            cls.actual_value.isnot(None)
        ).all()
        
        if not predictions:
            return {'error': 'No validated predictions found'}
        
        # Calculate performance metrics
        total_predictions = len(predictions)
        accurate_predictions = sum(1 for p in predictions if p.accuracy_validated)
        avg_confidence = sum(p.confidence_score for p in predictions) / total_predictions
        avg_feedback = sum(p.feedback_score for p in predictions if p.feedback_score) / max(1, sum(1 for p in predictions if p.feedback_score))
        
        return {
            'model_name': model_name,
            'evaluation_period_days': days,
            'total_predictions': total_predictions,
            'accurate_predictions': accurate_predictions,
            'accuracy_rate': accurate_predictions / total_predictions,
            'average_confidence': avg_confidence,
            'average_user_feedback': avg_feedback,
            'performance_trend': 'improving' if accurate_predictions > total_predictions * 0.8 else 'needs_improvement'
        }
    
    def __repr__(self):
        return (f"<Prediction(type='{self.prediction_type.value}', "
                f"model='{self.model_name}', "
                f"confidence={self.confidence_score:.2f}, "
                f"status='{self.status.value}')>")