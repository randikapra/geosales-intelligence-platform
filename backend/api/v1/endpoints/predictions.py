# # backend/api/v1/endpoints/predictions.py
# """
# Predictions and forecasting endpoints
# """
# from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
# from sqlalchemy.orm import Session
# from typing import List, Optional, Dict, Any
# from datetime import datetime, timedelta

# from core.database import get_db
# from services.prediction_service import PredictionService
# from schemas.predictions import (
#     SalesForecast, DemandPrediction, ChurnPrediction,
#     RoutePrediction, PredictionRequest, PredictionResponse
# )

# router = APIRouter()


# @router.post("/sales/forecast", response_model=List[SalesForecast])
# async def predict_sales_forecast(
#     request: PredictionRequest,
#     db: Session = Depends(get_db)
# ):
#     """Generate sales forecast for specified period"""
#     prediction_service = PredictionService(db)
#     return await prediction_service.predict_sales_forecast(request)


# @router.post("/demand/forecast", response_model=List[DemandPrediction])
# async def predict_demand(
#     territory: Optional[str] = Query(None),
#     product_category: Optional[str] = Query(None),
#     forecast_days: int = Query(30, ge=1, le=365),
#     db: Session = Depends(get_db)
# ):
#     """Predict product demand by territory"""
#     prediction_service = PredictionService(db)
#     return await prediction_service.predict_demand(territory, product_category, forecast_days)


# @router.post("/churn/prediction", response_model=List[ChurnPrediction])
# async def predict_customer_churn(
#     territory: Optional[str] = Query(None),
#     risk_threshold: float = Query(0.7, ge=0.0, le=1.0),
#     db: Session = Depends(get_db)
# ):
#     """Predict customer churn risk"""
#     prediction_service = PredictionService(db)
#     return await prediction_service.predict_customer_churn(territory, risk_threshold)


# @router.post("/routes/optimize", response_model=List[RoutePrediction])
# async def optimize_routes(
#     dealer_code: str,
#     date: datetime = Query(default_factory=datetime.now),
#     max_customers: int = Query(20, ge=1, le=50),
#     db: Session = Depends(get_db)
# ):
#     """Optimize dealer routes for given date"""
#     prediction_service = PredictionService(db)
#     return await prediction_service.optimize_routes(dealer_code, date, max_customers)


# @router.post("/anomaly/detect")
# async def detect_anomalies(
#     data_type: str = Query("sales", regex="^(sales|routes|gps)$"),
#     start_date: Optional[datetime] = Query(None),
#     end_date: Optional[datetime] = Query(None),
#     db: Session = Depends(get_db)
# ):
#     """Detect anomalies in sales, routes, or GPS data"""
#     prediction_service = PredictionService(db)
#     return await prediction_service.detect_anomalies(data_type, start_date, end_date)


# @router.post("/retrain/models")
# async def retrain_models(
#     background_tasks: BackgroundTasks,
#     model_type: Optional[str] = Query(None),
#     db: Session = Depends(get_db)
# ):
#     """Trigger model retraining"""
#     prediction_service = PredictionService(db)
#     task_id = await prediction_service.schedule_retraining(background_tasks, model_type)
#     return {"task_id": task_id, "status": "scheduled"}




# api/v1/endpoints/predictions.py
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from core.dependencies import get_db, get_current_user
from models.sales import SalesOrder
from models.dealer import Dealer
from datetime import datetime, date, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

router = APIRouter()

@router.get("/sales-forecast/{dealer_code}")
def predict_sales_forecast(
    dealer_code: str,
    forecast_days: int = Query(30, ge=7, le=90),
    historical_days: int = Query(90, ge=30, le=365),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Predict sales forecast for a dealer"""
    # Get historical sales data
    start_date = datetime.now() - timedelta(days=historical_days)
    
    sales_data = db.query(
        func.date(SalesOrder.date).label('date'),
        func.sum(SalesOrder.final_value).label('daily_sales'),
        func.count(SalesOrder.code).label('order_count')
    ).filter(
        SalesOrder.user_code == dealer_code,
        SalesOrder.date >= start_date
    ).group_by(func.date(SalesOrder.date)).order_by('date').all()
    
    if len(sales_data) < 14:
        raise HTTPException(
            status_code=400,
            detail="Insufficient historical data for prediction (minimum 14 days required)"
        )
    
    # Prepare data for ML model
    df = pd.DataFrame([
        {
            'date': row.date,
            'daily_sales': float(row.daily_sales),
            'order_count': row.order_count,
            'day_of_week': row.date.weekday(),
            'day_of_month': row.date.day
        }
        for row in sales_data
    ])
    
    # Create features
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Prepare training data
    X = df[['days_since_start', 'day_of_week', 'day_of_month', 'is_weekend', 'order_count']].values
    y = df['daily_sales'].values
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Generate predictions
    predictions = []
    last_date = df['date'].max()
    avg_order_count = df['order_count'].mean()
    
    for i in range(1, forecast_days + 1):
        future_date = last_date + timedelta(days=i)
        days_since_start = (future_date - df['date'].min()).days
        
        features = np.array([[
            days_since_start,
            future_date.weekday(),
            future_date.day,
            1 if future_date.weekday() in [5, 6] else 0,
            avg_order_count
        ]])
        
        predicted_sales = model.predict(features)[0]
        
        predictions.append({
            "date": str(future_date),
            "predicted_sales": round(max(0, predicted_sales), 2),
            "confidence": "medium"  # Simplified confidence scoring
        })
    
    # Calculate accuracy metrics on historical data
    historical_predictions = model.predict(X)
    mae = np.mean(np.abs(historical_predictions - y))
    
    return {
        "dealer_code": dealer_code,
        "forecast_period_days": forecast_days,
        "historical_period_days": historical_days,
        "model_accuracy": {
            "mean_absolute_error": round(mae, 2),
            "r2_score": round(model.score(X, y), 3)
        },
        "predictions": predictions,
        "total_predicted_sales": round(sum(p["predicted_sales"] for p in predictions), 2)
    }

@router.get("/demand-forecast")
def predict_demand_forecast(
    days: int = Query(30, ge=7, le=90),
    territory_code: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Predict overall demand forecast"""
    # Get historical data
    start_date = datetime.now() - timedelta(days=90)  # Use 90 days of history
    
    query = db.query(
        func.date(SalesOrder.date).label('date'),
        func.sum(SalesOrder.final_value).label('daily_sales'),
        func.count(SalesOrder.code).label('order_count'),
        func.count(func.distinct(SalesOrder.user_code)).label('active_dealers')
    ).filter(SalesOrder.date >= start_date)
    
    if territory_code:
        query = query.join(Dealer, SalesOrder.user_code == Dealer.user_code).filter(
            Dealer.territory_code == territory_code
        )
    
    sales_data = query.group_by(func.date(SalesOrder.date)).order_by('date').all()
    
    if len(sales_data) < 21:
        raise HTTPException(
            status_code=400,
            detail="Insufficient data for demand forecasting"
        )
    
    # Simple linear regression for trend
    df = pd.DataFrame([
        {
            'date': row.date,
            'daily_sales': float(row.daily_sales),
            'order_count': row.order_count,
            'active_dealers': row.active_dealers
        }
        for row in sales_data
    ])
    
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    df['day_of_week'] = df['date'].apply(lambda x: x.weekday())
    
    # Train model
    X = df[['days_since_start', 'day_of_week', 'active_dealers']].values
    y = df['daily_sales'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate predictions
    predictions = []
    last_date = df['date'].max()
    avg_active_dealers = df['active_dealers'].mean()
    
    for i in range(1, days + 1):
        future_date = last_date + timedelta(days=i)
        days_since_start = (future_date - df['date'].min()).days
        
        features = np.array([[
            days_since_start,
            future_date.weekday(),
            avg_active_dealers
        ]])
        
        predicted_demand = model.predict(features)[0]
        
        predictions.append({
            "date": str(future_date),
            "predicted_demand": round(max(0, predicted_demand), 2),
            "day_of_week": future_date.strftime("%A")
        })
    
    return {
        "territory_code": territory_code,
        "forecast_period_days": days,
        "predictions": predictions,
        "total_predicted_demand": round(sum(p["predicted_demand"] for p in predictions), 2),
        "average_daily_demand": round(sum(p["predicted_demand"] for p in predictions) / len(predictions), 2)
    }

@router.get("/dealer-performance-prediction/{dealer_code}")
def predict_dealer_performance(
    dealer_code: str,
    prediction_days: int = Query(30, ge=7, le=90),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Predict dealer performance metrics"""
    # Get historical performance data
    start_date = datetime.now() - timedelta(days=120)
    
    performance_data = db.query(
        func.date(SalesOrder.date).label('date'),
        func.sum(SalesOrder.final_value).label('daily_sales'),
        func.count(SalesOrder.code).label('order_count'),
        func.count(func.distinct(SalesOrder.distributor_code)).label('unique_customers')
    ).filter(
        SalesOrder.user_code == dealer_code,
        SalesOrder.date >= start_date
    ).group_by(func.date(SalesOrder.date)).order_by('date').all()
    
    if len(performance_data) < 21:
        raise HTTPException(
            status_code=400,
            detail="Insufficient performance data for prediction"
        )
    
    # Calculate moving averages and trends
    df = pd.DataFrame([
        {
            'date': row.date,
            'daily_sales': float(row.daily_sales),
            'order_count': row.order_count,
            'unique_customers': row.unique_customers
        }
        for row in performance_data
    ])
    
    # Calculate 7-day moving averages
    df['sales_ma7'] = df['daily_sales'].rolling(window=7).mean()
    df['orders_ma7'] = df['order_count'].rolling(window=7).mean()
    
    # Simple trend analysis
    recent_data = df.tail(14)  # Last 2 weeks
    older_data = df.tail(28).head(14)  # Previous 2 weeks
    
    sales_trend = (recent_data['daily_sales'].mean() - older_data['daily_sales'].mean()) / older_data['daily_sales'].mean()
    order_trend = (recent_data['order_count'].mean() - older_data['order_count'].mean()) / older_data['order_count'].mean()
    
    # Project future performance
    base_daily_sales = recent_data['daily_sales'].mean()
    base_daily_orders = recent_data['order_count'].mean()
    
    predictions = []
    for i in range(1, prediction_days + 1):
        # Apply trend with diminishing effect over time
        trend_factor = max(0.1, 1 - (i / prediction_days) * 0.5)
        
        predicted_sales = base_daily_sales * (1 + sales_trend * trend_factor)
        predicted_orders = base_daily_orders * (1 + order_trend * trend_factor)
        
        future_date = df['date'].max() + timedelta(days=i)
        
        predictions.append({
            "date": str(future_date),
            "predicted_sales": round(max(0, predicted_sales), 2),
            "predicted_orders": round(max(0, predicted_orders)),
            "confidence_score": round(max(0.3, 1 - (i / prediction_days) * 0.7), 2)
        })
    
    return {
        "dealer_code": dealer_code,
        "prediction_period_days": prediction_days,
        "current_trends": {
            "sales_trend_percentage": round(sales_trend * 100, 2),
            "order_trend_percentage": round(order_trend * 100, 2)
        },
        "predictions": predictions,
        "summary": {
            "total_predicted_sales": round(sum(p["predicted_sales"] for p in predictions), 2),
            "total_predicted_orders": sum(p["predicted_orders"] for p in predictions),
            "average_daily_sales": round(sum(p["predicted_sales"] for p in predictions) / len(predictions), 2)
        }
    }

@router.get("/market-opportunities")
def identify_market_opportunities(
    territory_code: Optional[str] = Query(None),
    min_potential_value: float = Query(10000, ge=1000),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Identify market opportunities and underperforming areas"""
    # Get dealer performance data
    start_date = datetime.now() - timedelta(days=60)
    
    query = db.query(
        Dealer.user_code,
        Dealer.user_name,
        Dealer.territory_code,
        Dealer.division_code,
        func.coalesce(func.sum(SalesOrder.final_value), 0).label('total_sales'),
        func.count(SalesOrder.code).label('order_count')
    ).outerjoin(
        SalesOrder, and_(
            Dealer.user_code == SalesOrder.user_code,
            SalesOrder.date >= start_date
        )
    )
    
    if territory_code:
        query = query.filter(Dealer.territory_code == territory_code)
    
    dealer_performance = query.group_by(
        Dealer.user_code, Dealer.user_name, Dealer.territory_code, Dealer.division_code
    ).all()
    
    # Calculate territory averages
    territory_stats = {}
    for dealer in dealer_performance:
        territory = dealer.territory_code
        if territory not in territory_stats:
            territory_stats[territory] = {'sales': [], 'orders': []}
        territory_stats[territory]['sales'].append(float(dealer.total_sales))
        territory_stats[territory]['orders'].append(dealer.order_count)
    
    # Calculate averages
    for territory in territory_stats:
        territory_stats[territory]['avg_sales'] = np.mean(territory_stats[territory]['sales'])
        territory_stats[territory]['avg_orders'] = np.mean(territory_stats[territory]['orders'])
    
    # Identify opportunities
    opportunities = []
    underperformers = []
    
    for dealer in dealer_performance:
        territory_avg_sales = territory_stats[dealer.territory_code]['avg_sales']
        performance_ratio = float(dealer.total_sales) / territory_avg_sales if territory_avg_sales > 0 else 0
        
        potential_increase = territory_avg_sales - float(dealer.total_sales)
        
        if performance_ratio < 0.7 and potential_increase >= min_potential_value:
            opportunities.append({
                "dealer_code": dealer.user_code,
                "dealer_name": dealer.user_name,
                "territory_code": dealer.territory_code,
                "current_sales": float(dealer.total_sales),
                "territory_average": round(territory_avg_sales, 2),
                "potential_increase": round(potential_increase, 2),
                "performance_ratio": round(performance_ratio, 2),
                "opportunity_score": round((1 - performance_ratio) * potential_increase / 1000, 2)
            })
        
        if performance_ratio < 0.5:
            underperformers.append({
                "dealer_code": dealer.user_code,
                "dealer_name": dealer.user_name,
                "territory_code": dealer.territory_code,
                "current_sales": float(dealer.total_sales),
                "performance_ratio": round(performance_ratio, 2),
                "support_priority": "high" if performance_ratio < 0.3 else "medium"
            })
    
    # Sort by opportunity score
    opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
    underperformers.sort(key=lambda x: x['performance_ratio'])
    
    return {
        "territory_code": territory_code,
        "analysis_period_days": 60,
        "min_potential_value": min_potential_value,
        "opportunities": opportunities[:20],  # Top 20 opportunities
        "underperformers": underperformers[:15],  # Top 15 underperformers
        "summary": {
            "total_opportunities": len(opportunities),
            "total_potential_value": round(sum(opp['potential_increase'] for opp in opportunities), 2),
            "dealers_needing_support": len(underperformers)
        }
    }
    