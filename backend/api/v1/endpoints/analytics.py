

# # api/v1/endpoints/analytics.py
# from typing import List, Optional
# from fastapi import APIRouter, Depends, HTTPException, Query
# from sqlalchemy.orm import Session
# from sqlalchemy import func, desc, and_
# from core.dependencies import get_db, get_current_user
# from models.sales import SalesOrder
# from models.dealer import Dealer
# from models.customer import Customer
# from datetime import datetime, date, timedelta
# import calendar

# router = APIRouter()

# @router.get("/dashboard")
# def get_dashboard_data(
#     days: int = Query(30, ge=1, le=365),
#     db: Session = Depends(get_db),
#     current_user = Depends(get_current_user)
# ):
#     """Get dashboard overview data"""
#     start_date = datetime.now() - timedelta(days=days)
    
#     # Total sales
#     total_sales = db.query(func.sum(SalesOrder.final_value)).filter(
#         SalesOrder.date >= start_date
#     ).scalar() or 0
    
#     # Total orders
#     total_orders = db.query(SalesOrder).filter(
#         SalesOrder.date >= start_date
#     ).count()
    
#     # Active dealers
#     active_dealers = db.query(func.count(func.distinct(SalesOrder.user_code))).filter(
#         SalesOrder.date >= start_date
#     ).scalar() or 0
    
#     # Total customers
#     total_customers = db.query(Customer).count()
    
#     # Average order value
#     avg_order_value = total_sales / total_orders if total_orders > 0 else 0
    
#     # Previous period comparison
#     prev_start_date = start_date - timedelta(days=days)
#     prev_total_sales = db.query(func.sum(SalesOrder.final_value)).filter(
#         and_(SalesOrder.date >= prev_start_date, SalesOrder.date < start_date)
#     ).scalar() or 0
    
#     sales_growth = ((total_sales - prev_total_sales) / prev_total_sales * 100) if prev_total_sales > 0 else 0
    
#     return {
#         "total_sales": float(total_sales),
#         "total_orders": total_orders,
#         "active_dealers": active_dealers,
#         "total_customers": total_customers,
#         "avg_order_value": float(avg_order_value),
#         "sales_growth_percentage": round(sales_growth, 2),
#         "period_days": days
#     }

# @router.get("/kpis")
# def get_kpis(
#     start_date: Optional[date] = Query(None),
#     end_date: Optional[date] = Query(None),
#     db: Session = Depends(get_db),
#     current_user = Depends(get_current_user)
# ):
#     """Get Key Performance Indicators"""
#     if not start_date:
#         start_date = datetime.now().date() - timedelta(days=30)
#     if not end_date:
#         end_date = datetime.now().date()
    
#     # Sales KPIs
#     sales_data = db.query(
#         func.sum(SalesOrder.final_value).label('total_revenue'),
#         func.count(SalesOrder.code).label('total_orders'),
#         func.avg(SalesOrder.final_value).label('avg_order_value'),
#         func.count(func.distinct(SalesOrder.user_code)).label('active_dealers'),
#         func.count(func.distinct(SalesOrder.distributor_code)).label('active_distributors')
#     ).filter(
#         and_(SalesOrder.date >= start_date, SalesOrder.date <= end_date)
#     ).first()
    
#     # Top performing dealers
#     top_dealers = db.query(
#         SalesOrder.user_code,
#         SalesOrder.user_name,
#         func.sum(SalesOrder.final_value).label('total_sales'),
#         func.count(SalesOrder.code).label('order_count')
#     ).filter(
#         and_(SalesOrder.date >= start_date, SalesOrder.date <= end_date)
#     ).group_by(SalesOrder.user_code, SalesOrder.user_name).order_by(
#         desc('total_sales')
#     ).limit(10).all()
    
#     # Sales by territory/division
#     territory_sales = db.query(
#         Dealer.territory_code,
#         Dealer.division_code,
#         func.sum(SalesOrder.final_value).label('total_sales')
#     ).join(
#         SalesOrder, Dealer.user_code == SalesOrder.user_code
#     ).filter(
#         and_(SalesOrder.date >= start_date, SalesOrder.date <= end_date)
#     ).group_by(Dealer.territory_code, Dealer.division_code).all()
    
#     return {
#         "period": {
#             "start_date": str(start_date),
#             "end_date": str(end_date)
#         },
#         "sales_kpis": {
#             "total_revenue": float(sales_data.total_revenue or 0),
#             "total_orders": sales_data.total_orders or 0,
#             "avg_order_value": float(sales_data.avg_order_value or 0),
#             "active_dealers": sales_data.active_dealers or 0,
#             "active_distributors": sales_data.active_distributors or 0
#         },
#         "top_dealers": [
#             {
#                 "dealer_code": dealer.user_code,
#                 "dealer_name": dealer.user_name,
#                 "total_sales": float(dealer.total_sales),
#                 "order_count": dealer.order_count
#             }
#             for dealer in top_dealers
#         ],
#         "territory_performance": [
#             {
#                 "territory_code": territory.territory_code,
#                 "division_code": territory.division_code,
#                 "total_sales": float(territory.total_sales)
#             }
#             for territory in territory_sales
#         ]
#     }

# @router.get("/trends")
# def get_trends(
#     period: str = Query("daily", regex="^(daily|weekly|monthly)$"),
#     days: int = Query(90, ge=7, le=365),
#     db: Session = Depends(get_db),
#     current_user = Depends(get_current_user)
# ):
#     """Get sales trends over time"""
#     start_date = datetime.now() - timedelta(days=days)
    
#     if period == "daily":
#         trends = db.query(
#             func.date(SalesOrder.date).label('period'),
#             func.sum(SalesOrder.final_value).label('total_sales'),
#             func.count(SalesOrder.code).label('order_count')
#         ).filter(
#             SalesOrder.date >= start_date
#         ).group_by(func.date(SalesOrder.date)).order_by('period').all()
    
#     elif period == "weekly":
#         trends = db.query(
#             func.date_format(SalesOrder.date, '%Y-%u').label('period'),
#             func.sum(SalesOrder.final_value).label('total_sales'),
#             func.count(SalesOrder.code).label('order_count')
#         ).filter(
#             SalesOrder.date >= start_date
#         ).group_by(func.date_format(SalesOrder.date, '%Y-%u')).order_by('period').all()
    
#     elif period == "monthly":
#         trends = db.query(
#             func.date_format(SalesOrder.date, '%Y-%m').label('period'),
#             func.sum(SalesOrder.final_value).label('total_sales'),
#             func.count(SalesOrder.code).label('order_count')
#         ).filter(
#             SalesOrder.date >= start_date
#         ).group_by(func.date_format(SalesOrder.date, '%Y-%m')).order_by('period').all()
    
#     return {
#         "period_type": period,
#         "trends": [
#             {
#                 "period": str(trend.period),
#                 "total_sales": float(trend.total_sales),
#                 "order_count": trend.order_count
#             }
#             for trend in trends
#         ]
#     }

# @router.get("/dealer-performance")
# def get_dealer_performance_analytics(
#     limit: int = Query(20, ge=1, le=100),
#     days: int = Query(30, ge=1, le=365),
#     db: Session = Depends(get_db),
#     current_user = Depends(get_current_user)
# ):
#     """Get comprehensive dealer performance analytics"""
#     start_date = datetime.now() - timedelta(days=days)
    
#     performance_data = db.query(
#         SalesOrder.user_code,
#         SalesOrder.user_name,
#         func.sum(SalesOrder.final_value).label('total_sales'),
#         func.count(SalesOrder.code).label('total_orders'),
#         func.avg(SalesOrder.final_value).label('avg_order_value'),
#         func.count(func.distinct(SalesOrder.distributor_code)).label('unique_customers'),
#         func.count(func.distinct(func.date(SalesOrder.date))).label('active_days')
#     ).filter(
#         SalesOrder.date >= start_date
#     ).group_by(
#         SalesOrder.user_code, SalesOrder.user_name
#     ).order_by(
#         desc('total_sales')
#     ).limit(limit).all()
    
#     return {
#         "period_days": days,
#         "dealer_performance": [
#             {
#                 "dealer_code": perf.user_code,
#                 "dealer_name": perf.user_name,
#                 "total_sales": float(perf.total_sales),
#                 "total_orders": perf.total_orders,
#                 "avg_order_value": float(perf.avg_order_value),
#                 "unique_customers": perf.unique_customers,
#                 "active_days": perf.active_days,
#                 "sales_per_day": float(perf.total_sales / perf.active_days) if perf.active_days > 0 else 0
#             }
#             for perf in performance_data
#         ]
#     }

from typing import List, Optional, Dict, Any
from datetime import date, datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from core.dependencies import get_db, get_current_user
from models.user import User
from schemas.analytics import (
    DashboardResponse,
    KPIResponse,
    TrendAnalysisResponse,
    GeographicAnalysisResponse,
    PerformanceAnalysisResponse,
    CustomerAnalysisResponse,
    TimeSeriesData,
    HeatmapData,
    DrillDownRequest
)
from services.analytics_service import AnalyticsService
from utils.date_utils import get_date_range

router = APIRouter()

@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard_data(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    division_code: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive dashboard data
    """
    analytics_service = AnalyticsService(db)
    
    # Set default date range if not provided
    if not start_date or not end_date:
        start_date, end_date = get_date_range("last_30_days")
    
    dashboard_data = analytics_service.get_dashboard_data(
        start_date=start_date,
        end_date=end_date,
        division_code=division_code
    )
    
    return DashboardResponse(**dashboard_data)

@router.get("/kpis", response_model=List[KPIResponse])
async def get_kpis(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    division_code: Optional[str] = Query(None),
    compare_previous: bool = Query(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get key performance indicators
    """
    analytics_service = AnalyticsService(db)
    
    # Set default date range if not provided
    if not start_date or not end_date:
        start_date, end_date = get_date_range("current_month")
    
    kpis = analytics_service.get_kpis(
        start_date=start_date,
        end_date=end_date,
        division_code=division_code,
        compare_previous=compare_previous
    )
    
    return [KPIResponse(**kpi) for kpi in kpis]

@router.get("/trends/sales", response_model=TrendAnalysisResponse)
async def get_sales_trends(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    granularity: str = Query("daily", regex="^(daily|weekly|monthly)$"),
    division_code: Optional[str] = Query(None),
    dealer_code: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get sales trend analysis
    """
    analytics_service = AnalyticsService(db)
    
    # Set default date range if not provided
    if not start_date or not end_date:
        start_date, end_date = get_date_range("last_90_days")
    
    trends = analytics_service.get_sales_trends(
        start_date=start_date,
        end_date=end_date,
        granularity=granularity,
        division_code=division_code,
        dealer_code=dealer_code
    )
    
    return TrendAnalysisResponse(**trends)

@router.get("/geographic/heatmap", response_model=List[HeatmapData])
async def get_geographic_heatmap(
    start_date: Optional[])