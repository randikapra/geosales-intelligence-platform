

# # api/v1/endpoints/sales.py
# from typing import List, Optional
# from fastapi import APIRouter, Depends, HTTPException, Query
# from sqlalchemy.orm import Session
# from sqlalchemy import func, desc, and_
# from core.dependencies import get_db, get_current_user
# from models.sales import SalesOrder
# from schemas.sales import SalesOrderCreate, SalesOrderResponse, SalesOrderUpdate
# from datetime import datetime, date

# router = APIRouter()

# @router.get("/", response_model=List[SalesOrderResponse])
# def get_sales_orders(
#     skip: int = Query(0, ge=0),
#     limit: int = Query(100, ge=1, le=1000),
#     dealer_code: Optional[str] = Query(None),
#     distributor_code: Optional[str] = Query(None),
#     start_date: Optional[date] = Query(None),
#     end_date: Optional[date] = Query(None),
#     min_value: Optional[float] = Query(None),
#     db: Session = Depends(get_db),
#     current_user = Depends(get_current_user)
# ):
#     """Get sales orders with filtering"""
#     query = db.query(SalesOrder)
    
#     if dealer_code:
#         query = query.filter(SalesOrder.user_code == dealer_code)
    
#     if distributor_code:
#         query = query.filter(SalesOrder.distributor_code == distributor_code)
    
#     if start_date:
#         query = query.filter(SalesOrder.date >= start_date)
    
#     if end_date:
#         query = query.filter(SalesOrder.date <= end_date)
    
#     if min_value:
#         query = query.filter(SalesOrder.final_value >= min_value)
    
#     orders = query.order_by(desc(SalesOrder.date)).offset(skip).limit(limit).all()
#     return orders

# @router.get("/{order_code}", response_model=SalesOrderResponse)
# def get_sales_order(
#     order_code: str,
#     db: Session = Depends(get_db),
#     current_user = Depends(get_current_user)
# ):
#     """Get sales order by code"""
#     order = db.query(SalesOrder).filter(SalesOrder.code == order_code).first()
#     if not order:
#         raise HTTPException(status_code=404, detail="Sales order not found")
#     return order

# @router.post("/", response_model=SalesOrderResponse)
# def create_sales_order(
#     order: SalesOrderCreate,
#     db: Session = Depends(get_db),
#     current_user = Depends(get_current_user)
# ):
#     """Create new sales order"""
#     db_order = SalesOrder(**order.dict())
#     db.add(db_order)
#     db.commit()
#     db.refresh(db_order)
#     return db_order

# @router.put("/{order_code}", response_model=SalesOrderResponse)
# def update_sales_order(
#     order_code: str,
#     order: SalesOrderUpdate,
#     db: Session = Depends(get_db),
#     current_user = Depends(get_current_user)
# ):
#     """Update sales order"""
#     db_order = db.query(SalesOrder).filter(SalesOrder.code == order_code).first()
#     if not db_order:
#         raise HTTPException(status_code=404, detail="Sales order not found")
    
#     for field, value in order.dict(exclude_unset=True).items():
#         setattr(db_order, field, value)
    
#     db.commit()
#     db.refresh(db_order)
#     return db_order

# @router.delete("/{order_code}")
# def delete_sales_order(
#     order_code: str,
#     db: Session = Depends(get_db),
#     current_user = Depends(get_current_user)
# ):
#     """Delete sales order"""
#     db_order = db.query(SalesOrder).filter(SalesOrder.code == order_code).first()
#     if not db_order:
#         raise HTTPException(status_code=404, detail="Sales order not found")
    
#     db.delete(db_order)
#     db.commit()
#     return {"message": "Sales order deleted successfully"}

# @router.get("/aggregations/summary")
# def get_sales_summary(
#     start_date: Optional[date] = Query(None),
#     end_date: Optional[date] = Query(None),
#     group_by: str = Query("day", regex="^(day|week|month|dealer|distributor)$"),
#     db: Session = Depends(get_db),
#     current_user = Depends(get_current_user)
# ):
#     """Get sales summary with aggregations"""
#     query = db.query(SalesOrder)
    
#     if start_date:
#         query = query.filter(SalesOrder.date >= start_date)
#     if end_date:
#         query = query.filter(SalesOrder.date <= end_date)
    
#     if group_by == "day":
#         results = query.with_entities(
#             func.date(SalesOrder.date).label('period'),
#             func.sum(SalesOrder.final_value).label('total_sales'),
#             func.count(SalesOrder.code).label('order_count'),
#             func.avg(SalesOrder.final_value).label('avg_order_value')
#         ).group_by(func.date(SalesOrder.date)).order_by('period').all()
    
#     elif group_by == "month":
#         results = query.with_entities(
#             func.date_format(SalesOrder.date, '%Y-%m').label('period'),
#             func.sum(SalesOrder.final_value).label('total_sales'),
#             func.count(SalesOrder.code).label('order_count'),
#             func.avg(SalesOrder.final_value).label('avg_order_value')
#         ).group_by(func.date_format(SalesOrder.date, '%Y-%m')).order_by('period').all()
    
#     elif group_by == "dealer":
#         results = query.with_entities(
#             SalesOrder.user_code.label('period'),
#             SalesOrder.user_name.label('dealer_name'),
#             func.sum(SalesOrder.final_value).label('total_sales'),
#             func.count(SalesOrder.code).label('order_count'),
#             func.avg(SalesOrder.final_value).label('avg_order_value')
#         ).group_by(SalesOrder.user_code, SalesOrder.user_name).order_by(desc('total_sales')).all()
    
#     elif group_by == "distributor":
#         results = query.with_entities(
#             SalesOrder.distributor_code.label('period'),
#             func.sum(SalesOrder.final_value).label('total_sales'),
#             func.count(SalesOrder.code).label('order_count'),
#             func.avg(SalesOrder.final_value).label('avg_order_value')
#         ).group_by(SalesOrder.distributor_code).order_by(desc('total_sales')).all()
    
#     return [
#         {
#             "period": str(result.period),
#             "total_sales": float(result.total_sales),
#             "order_count": result.order_count,
#             "avg_order_value": float(result.avg_order_value),
#             **({"dealer_name": result.dealer_name} if hasattr(result, 'dealer_name') else {})
#         }
#         for result in results
#     ]
from typing import List, Optional
from datetime import date, datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from core.dependencies import get_db, get_current_user
from models.user import User
from models.sales import Sales, SalesOrder
from schemas.sales import (
    SalesOrderCreate,
    SalesOrderUpdate,
    SalesOrderResponse,
    SalesOrderFilter,
    SalesAggregationResponse,
    SalesSummary,
    SalesExportRequest
)
from services.sales_service import SalesService
from utils.date_utils import get_date_range

router = APIRouter()

@router.get("/orders", response_model=List[SalesOrderResponse])
async def get_sales_orders(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    dealer_code: Optional[str] = Query(None),
    distributor_code: Optional[str] = Query(None),
    division_code: Optional[str] = Query(None),
    min_amount: Optional[float] = Query(None),
    max_amount: Optional[float] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get sales orders with optional filtering
    """
    sales_service = SalesService(db)
    
    filters = SalesOrderFilter(
        start_date=start_date,
        end_date=end_date,
        dealer_code=dealer_code,
        distributor_code=distributor_code,
        division_code=division_code,
        min_amount=min_amount,
        max_amount=max_amount
    )
    
    orders = sales_service.get_sales_orders(
        skip=skip,
        limit=limit,
        filters=filters
    )
    
    return [SalesOrderResponse.from_orm(order) for order in orders]

@router.get("/orders/count")
async def get_sales_orders_count(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    dealer_code: Optional[str] = Query(None),
    distributor_code: Optional[str] = Query(None),
    division_code: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get total count of sales orders
    """
    sales_service = SalesService(db)
    
    filters = SalesOrderFilter(
        start_date=start_date,
        end_date=end_date,
        dealer_code=dealer_code,
        distributor_code=distributor_code,
        division_code=division_code
    )
    
    count = sales_service.get_sales_orders_count(filters)
    return {"total_orders": count}

@router.get("/orders/{order_code}", response_model=SalesOrderResponse)
async def get_sales_order(
    order_code: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get sales order by code
    """
    sales_service = SalesService(db)
    
    order = sales_service.get_sales_order_by_code(order_code)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sales order not found"
        )
    
    return SalesOrderResponse.from_orm(order)

@router.post("/orders", response_model=SalesOrderResponse, status_code=status.HTTP_201_CREATED)
async def create_sales_order(
    order_data: SalesOrderCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new sales order
    """
    sales_service = SalesService(db)
    
    # Check if order code already exists
    existing_order = sales_service.get_sales_order_by_code(order_data.code)
    if existing_order:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sales order code already exists"
        )
    
    order = sales_service.create_sales_order(order_data)
    return SalesOrderResponse.from_orm(order)

@router.put("/orders/{order_code}", response_model=SalesOrderResponse)
async def update_sales_order(
    order_code: str,
    order_data: SalesOrderUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update sales order information
    """
    sales_service = SalesService(db)
    
    order = sales_service.get_sales_order_by_code(order_code)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sales order not found"
        )
    
    updated_order = sales_service.update_sales_order(order_code, order_data)
    return SalesOrderResponse.from_orm(updated_order)

@router.delete("/orders/{order_code}")
async def delete_sales_order(
    order_code: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a sales order
    """
    sales_service = SalesService(db)
    
    order = sales_service.get_sales_order_by_code(order_code)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sales order not found"
        )
    
    sales_service.delete_sales_order(order_code)
    return {"message": "Sales order deleted successfully"}

@router.get("/summary", response_model=SalesSummary)
async def get_sales_summary(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    dealer_code: Optional[str] = Query(None),
    division_code: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get sales summary statistics
    """
    sales_service = SalesService(db)
    
    # Set default date range if not provided
    if not start_date or not end_date:
        start_date, end_date = get_date_range("current_month")
    
    filters = SalesOrderFilter(
        start_date=start_date,
        end_date=end_date,
        dealer_code=dealer_code,
        division_code=division_code
    )
    
    summary = sales_service.get_sales_summary(filters)
    return SalesSummary(**summary)

@router.get("/aggregation/daily", response_model=List[SalesAggregationResponse])
async def get_daily_sales_aggregation(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    dealer_code: Optional[str] = Query(None),
    division_code: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get daily sales aggregation
    """
    sales_service = SalesService(db)
    
    # Set default date range if not provided
    if not start_date or not end_date:
        start_date, end_date = get_date_range("last_30_days")
    
    filters = SalesOrderFilter(
        start_date=start_date,
        end_date=end_date,
        dealer_code=dealer_code,
        division_code=division_code
    )
    
    aggregation = sales_service.get_daily_sales_aggregation(filters)
    return [SalesAggregationResponse(**item) for item in aggregation]

@router.get("/aggregation/monthly", response_model=List[SalesAggregationResponse])
async def get_monthly_sales_aggregation(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    dealer_code: Optional[str] = Query(None),
    division_code: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get monthly sales aggregation
    """
    sales_service = SalesService(db)
    
    # Set default date range if not provided
    if not start_date or not end_date:
        start_date, end_date = get_date_range("last_12_months")
    
    filters = SalesOrderFilter(
        start_date=start_date,
        end_date=end_date,
        dealer_code=dealer_code,
        division_code=division_code
    )
    
    aggregation = sales_service.get_monthly_sales_aggregation(filters)
    return [SalesAggregationResponse(**item) for item in aggregation]

@router.get("/aggregation/dealer")
async def get_sales_by_dealer(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    division_code: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get sales aggregation by dealer
    """
    sales_service = SalesService(db)
    
    # Set default date range if not provided
    if not start_date or not end_date:
        start_date, end_date = get_date_range("current_month")
    
    filters = SalesOrderFilter(
        start_date=start_date,
        end_date=end_date,
        division_code=division_code
    )
    
    aggregation = sales_service.get_sales_by_dealer(filters, limit)
    return {"dealer_sales": aggregation}

@router.get("/aggregation/distributor")
async def get_sales_by_distributor(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    division_code: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get sales aggregation by distributor
    """
    sales_service = SalesService(db)
    
    # Set default date range if not provided
    if not start_date or not end_date:
        start_date, end_date = get_date_range("current_month")
    
    filters = SalesOrderFilter(
        start_date=start_date,
        end_date=end_date,
        division_code=division_code
    )
    
    aggregation = sales_service.get_sales_by_distributor(filters, limit)
    return {"distributor_sales": aggregation}

@router.get("/trends/comparison")
async def get_sales_trends_comparison(
    period: str = Query("monthly", regex="^(daily|weekly|monthly)$"),
    current_start: date = Query(...),
    current_end: date = Query(...),
    previous_start: date = Query(...),
    previous_end: date = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get sales trends comparison between two periods
    """
    sales_service = SalesService(db)
    
    comparison = sales_service.get_sales_trends_comparison(
        period=period,
        current_start=current_start,
        current_end=current_end,
        previous_start=previous_start,
        previous_end=previous_end
    )
    
    return comparison

@router.get("/performance/top-performers")
async def get_top_sales_performers(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    metric: str = Query("total_sales", regex="^(total_sales|order_count|avg_order_value)$"),
    limit: int = Query(10, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get top sales performers
    """
    sales_service = SalesService(db)
    
    # Set default date range if not provided
    if not start_date or not end_date:
        start_date, end_date = get_date_range("current_month")
    
    performers = sales_service.get_top_sales_performers(
        start_date=start_date,
        end_date=end_date,
        metric=metric,
        limit=limit
    )
    
    return {"top_performers": performers, "metric": metric}

@router.post("/import/bulk", status_code=status.HTTP_201_CREATED)
async def bulk_import_sales_orders(
    orders_data: List[SalesOrderCreate],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Bulk import sales orders from a list
    """
    sales_service = SalesService(db)
    
    results = sales_service.bulk_import_sales_orders(orders_data)
    
    return {
        "message": "Bulk import completed",
        "imported_count": results["imported_count"],
        "failed_count": results["failed_count"],
        "errors": results["errors"]
    }

@router.post("/export/csv")
async def export_sales_csv(
    export_request: SalesExportRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Export sales data as CSV
    """
    sales_service = SalesService(db)
    
    csv_data = sales_service.export_sales_csv(export_request)
    
    from fastapi.responses import Response
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sales_data.csv"}
    )

@router.get("/metrics/kpis")
async def get_sales_kpis(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    division_code: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get key performance indicators for sales
    """
    sales_service = SalesService(db)
    
    # Set default date range if not provided
    if not start_date or not end_date:
        start_date, end_date = get_date_range("current_month")
    
    kpis = sales_service.get_sales_kpis(start_date, end_date, division_code)
    return kpis

@router.get("/analysis/seasonal")
async def get_seasonal_analysis(
    year: int = Query(..., ge=2020, le=2030),
    division_code: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get seasonal sales analysis
    """
    sales_service = SalesService(db)
    
    analysis = sales_service.get_seasonal_analysis(year, division_code)
    return analysis

@router.get("/forecasting/next-period")
async def get_sales_forecast(
    periods: int = Query(3, ge=1, le=12),
    model_type: str = Query("linear", regex="^(linear|exponential|seasonal)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get sales forecast for next periods
    """
    sales_service = SalesService(db)
    
    forecast = sales_service.get_sales_forecast(periods, model_type)
    return forecast