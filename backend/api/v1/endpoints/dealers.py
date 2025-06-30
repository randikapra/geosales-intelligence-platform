

# # api/v1/endpoints/dealers.py
# from typing import List, Optional
# from fastapi import APIRouter, Depends, HTTPException, Query
# from sqlalchemy.orm import Session
# from sqlalchemy import func, desc
# from core.dependencies import get_db, get_current_user
# from models.dealer import Dealer
# from models.sales import SalesOrder
# from schemas.dealer import DealerCreate, DealerResponse, DealerPerformance
# from datetime import datetime, timedelta

# router = APIRouter()

# @router.get("/", response_model=List[DealerResponse])
# def get_dealers(
#     skip: int = Query(0, ge=0),
#     limit: int = Query(100, ge=1, le=1000),
#     territory: Optional[str] = Query(None),
#     division: Optional[str] = Query(None),
#     db: Session = Depends(get_db),
#     current_user = Depends(get_current_user)
# ):
#     """Get all dealers with optional filtering"""
#     query = db.query(Dealer)
    
#     if territory:
#         query = query.filter(Dealer.territory_code == territory)
    
#     if division:
#         query = query.filter(Dealer.division_code == division)
    
#     dealers = query.offset(skip).limit(limit).all()
#     return dealers

# @router.get("/{dealer_code}", response_model=DealerResponse)
# def get_dealer(
#     dealer_code: str,
#     db: Session = Depends(get_db),
#     current_user = Depends(get_current_user)
# ):
#     """Get dealer by code"""
#     dealer = db.query(Dealer).filter(Dealer.user_code == dealer_code).first()
#     if not dealer:
#         raise HTTPException(status_code=404, detail="Dealer not found")
#     return dealer

# @router.post("/", response_model=DealerResponse)
# def create_dealer(
#     dealer: DealerCreate,
#     db: Session = Depends(get_db),
#     current_user = Depends(get_current_user)
# ):
#     """Create new dealer"""
#     db_dealer = Dealer(**dealer.dict())
#     db.add(db_dealer)
#     db.commit()
#     db.refresh(db_dealer)
#     return db_dealer

# @router.get("/{dealer_code}/performance")
# def get_dealer_performance(
#     dealer_code: str,
#     days: int = Query(30, ge=1, le=365),
#     db: Session = Depends(get_db),
#     current_user = Depends(get_current_user)
# ):
#     """Get dealer performance metrics"""
#     dealer = db.query(Dealer).filter(Dealer.user_code == dealer_code).first()
#     if not dealer:
#         raise HTTPException(status_code=404, detail="Dealer not found")
    
#     start_date = datetime.now() - timedelta(days=days)
    
#     # Get sales performance
#     sales_query = db.query(SalesOrder).filter(
#         SalesOrder.user_code == dealer_code,
#         SalesOrder.date >= start_date
#     )
    
#     total_sales = sales_query.with_entities(func.sum(SalesOrder.final_value)).scalar() or 0
#     total_orders = sales_query.count()
#     avg_order_value = total_sales / total_orders if total_orders > 0 else 0
    
#     # Get daily sales trend
#     daily_sales = db.query(
#         func.date(SalesOrder.date).label('date'),
#         func.sum(SalesOrder.final_value).label('total_sales'),
#         func.count(SalesOrder.code).label('order_count')
#     ).filter(
#         SalesOrder.user_code == dealer_code,
#         SalesOrder.date >= start_date
#     ).group_by(func.date(SalesOrder.date)).order_by('date').all()
    
#     return {
#         "dealer_code": dealer_code,
#         "dealer_name": dealer.user_name,
#         "period_days": days,
#         "total_sales": float(total_sales),
#         "total_orders": total_orders,
#         "avg_order_value": float(avg_order_value),
#         "daily_sales": [
#             {
#                 "date": str(day.date),
#                 "total_sales": float(day.total_sales),
#                 "order_count": day.order_count
#             }
#         ]
#     }

from typing import List, Optional
from datetime import date, datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from core.dependencies import get_db, get_current_user
from models.user import User
from models.dealer import Dealer
from schemas.dealer import (
    DealerCreate,
    DealerUpdate,
    DealerResponse,
    DealerPerformanceResponse,
    DealerLocationResponse,
    TerritoryAssignment,
    PerformanceMetrics
)
from services.dealer_service import DealerService
from utils.date_utils import get_date_range

router = APIRouter()

@router.get("/", response_model=List[DealerResponse])
async def get_dealers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000),
    division_code: Optional[str] = Query(None),
    territory_code: Optional[str] = Query(None),
    active_only: bool = Query(True),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all dealers with optional filtering
    """
    dealer_service = DealerService(db)
    
    filters = {}
    if division_code:
        filters['division_code'] = division_code
    if territory_code:
        filters['territory_code'] = territory_code
    if active_only:
        filters['is_active'] = True
    if search:
        filters['search'] = search
    
    dealers = dealer_service.get_dealers(
        skip=skip,
        limit=limit,
        filters=filters
    )
    
    return [DealerResponse.from_orm(dealer) for dealer in dealers]

@router.get("/count")
async def get_dealers_count(
    division_code: Optional[str] = Query(None),
    territory_code: Optional[str] = Query(None),
    active_only: bool = Query(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get total count of dealers
    """
    dealer_service = DealerService(db)
    
    filters = {}
    if division_code:
        filters['division_code'] = division_code
    if territory_code:
        filters['territory_code'] = territory_code
    if active_only:
        filters['is_active'] = True
    
    count = dealer_service.get_dealers_count(filters)
    return {"total_dealers": count}

@router.get("/{dealer_code}", response_model=DealerResponse)
async def get_dealer(
    dealer_code: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get dealer by code
    """
    dealer_service = DealerService(db)
    
    dealer = dealer_service.get_dealer_by_code(dealer_code)
    if not dealer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dealer not found"
        )
    
    return DealerResponse.from_orm(dealer)

@router.post("/", response_model=DealerResponse, status_code=status.HTTP_201_CREATED)
async def create_dealer(
    dealer_data: DealerCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new dealer
    """
    dealer_service = DealerService(db)
    
    # Check if dealer code already exists
    existing_dealer = dealer_service.get_dealer_by_code(dealer_data.user_code)
    if existing_dealer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dealer code already exists"
        )
    
    dealer = dealer_service.create_dealer(dealer_data)
    return DealerResponse.from_orm(dealer)

@router.put("/{dealer_code}", response_model=DealerResponse)
async def update_dealer(
    dealer_code: str,
    dealer_data: DealerUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update dealer information
    """
    dealer_service = DealerService(db)
    
    dealer = dealer_service.get_dealer_by_code(dealer_code)
    if not dealer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dealer not found"
        )
    
    updated_dealer = dealer_service.update_dealer(dealer_code, dealer_data)
    return DealerResponse.from_orm(updated_dealer)

@router.delete("/{dealer_code}")
async def delete_dealer(
    dealer_code: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a dealer (soft delete - marks as inactive)
    """
    dealer_service = DealerService(db)
    
    dealer = dealer_service.get_dealer_by_code(dealer_code)
    if not dealer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dealer not found"
        )
    
    dealer_service.delete_dealer(dealer_code)
    return {"message": "Dealer deleted successfully"}

@router.get("/{dealer_code}/performance", response_model=DealerPerformanceResponse)
async def get_dealer_performance(
    dealer_code: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get dealer performance metrics
    """
    dealer_service = DealerService(db)
    
    dealer = dealer_service.get_dealer_by_code(dealer_code)
    if not dealer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dealer not found"
        )
    
    # Set default date range if not provided
    if not start_date or not end_date:
        start_date, end_date = get_date_range("last_30_days")
    
    performance = dealer_service.get_dealer_performance(
        dealer_code, start_date, end_date
    )
    
    return DealerPerformanceResponse(**performance)

@router.get("/performance/ranking")
async def get_dealers_performance_ranking(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    division_code: Optional[str] = Query(None),
    metric: str = Query("total_sales", regex="^(total_sales|order_count|avg_order_value)$"),
    limit: int = Query(20, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get dealer performance ranking
    """
    dealer_service = DealerService(db)
    
    # Set default date range if not provided
    if not start_date or not end_date:
        start_date, end_date = get_date_range("last_30_days")
    
    ranking = dealer_service.get_performance_ranking(
        start_date=start_date,
        end_date=end_date,
        division_code=division_code,
        metric=metric,
        limit=limit
    )
    
    return {"ranking": ranking, "metric": metric, "period": f"{start_date} to {end_date}"}

@router.get("/{dealer_code}/territory", response_model=List[str])
async def get_dealer_territory(
    dealer_code: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get dealer's assigned territory codes
    """
    dealer_service = DealerService(db)
    
    dealer = dealer_service.get_dealer_by_code(dealer_code)
    if not dealer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dealer not found"
        )
    
    territories = dealer_service.get_dealer_territories(dealer_code)
    return territories

@router.post("/{dealer_code}/territory/assign")
async def assign_territory_to_dealer(
    dealer_code: str,
    assignment: TerritoryAssignment,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Assign territory to dealer
    """
    dealer_service = DealerService(db)
    
    dealer = dealer_service.get_dealer_by_code(dealer_code)
    if not dealer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dealer not found"
        )
    
    success = dealer_service.assign_territory(dealer_code, assignment.territory_codes)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to assign territories"
        )
    
    return {"message": "Territories assigned successfully"}

@router.delete("/{dealer_code}/territory/{territory_code}")
async def remove_territory_from_dealer(
    dealer_code: str,
    territory_code: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Remove territory assignment from dealer
    """
    dealer_service = DealerService(db)
    
    dealer = dealer_service.get_dealer_by_code(dealer_code)
    if not dealer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dealer not found"
        )
    
    success = dealer_service.remove_territory(dealer_code, territory_code)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to remove territory assignment"
        )
    
    return {"message": "Territory assignment removed successfully"}

@router.get("/{dealer_code}/locations", response_model=List[DealerLocationResponse])
async def get_dealer_locations(
    dealer_code: str,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(1000, le=5000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get dealer's GPS location history
    """
    dealer_service = DealerService(db)
    
    dealer = dealer_service.get_dealer_by_code(dealer_code)
    if not dealer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dealer not found"
        )
    
    # Set default date range if not provided
    if not start_date or not end_date:
        start_date, end_date = get_date_range("today")
    
    locations = dealer_service.get_dealer_locations(
        dealer_code, start_date, end_date, limit
    )
    
    return [DealerLocationResponse.from_orm(location) for location in locations]

@router.get("/{dealer_code}/route-history")
async def get_dealer_route_history(
    dealer_code: str,
    date_filter: date = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get dealer's route history for a specific date
    """
    dealer_service = DealerService(db)
    
    dealer = dealer_service.get_dealer_by_code(dealer_code)
    if not dealer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dealer not found"
        )
    
    route_history = dealer_service.get_route_history(dealer_code, date_filter)
    return {
        "dealer_code": dealer_code,
        "date": date_filter,
        "route_data": route_history
    }

@router.get("/{dealer_code}/customers")
async def get_dealer_customers(
    dealer_code: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get customers assigned to a dealer
    """
    dealer_service = DealerService(db)
    
    dealer = dealer_service.get_dealer_by_code(dealer_code)
    if not dealer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dealer not found"
        )
    
    customers = dealer_service.get_dealer_customers(dealer_code)
    return {"dealer_code": dealer_code, "customers": customers}

@router.get("/divisions/list")
async def get_divisions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get list of all unique division codes
    """
    dealer_service = DealerService(db)
    
    divisions = dealer_service.get_unique_divisions()
    return {"divisions": divisions}

@router.get("/stats/division")
async def get_dealers_by_division_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get dealer count statistics by division
    """
    dealer_service = DealerService(db)
    
    stats = dealer_service.get_dealer_stats_by_division()
    return {"division_stats": stats}

@router.get("/performance/metrics")
async def get_overall_performance_metrics(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    division_code: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get overall dealer performance metrics
    """
    dealer_service = DealerService(db)
    
    # Set default date range if not provided
    if not start_date or not end_date:
        start_date, end_date = get_date_range("last_30_days")
    
    metrics = dealer_service.get_overall_performance_metrics(
        start_date, end_date, division_code
    )
    
    return PerformanceMetrics(**metrics)

@router.post("/import/bulk", status_code=status.HTTP_201_CREATED)
async def bulk_import_dealers(
    dealers_data: List[DealerCreate],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Bulk import dealers from a list
    """
    dealer_service = DealerService(db)
    
    results = dealer_service.bulk_import_dealers(dealers_data)
    
    return {
        "message": "Bulk import completed",
        "imported_count": results["imported_count"],
        "failed_count": results["failed_count"],
        "errors": results["errors"]
    }

@router.get("/export/csv")
async def export_dealers_csv(
    division_code: Optional[str] = Query(None),
    active_only: bool = Query(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Export dealers data as CSV
    """
    dealer_service = DealerService(db)
    
    filters = {}
    if division_code:
        filters['division_code'] = division_code
    if active_only:
        filters['is_active'] = True
    
    csv_data = dealer_service.export_dealers_csv(filters)
    
    from fastapi.responses import Response
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=dealers.csv"}
    )

@router.patch("/{dealer_code}/status")
async def update_dealer_status(
    dealer_code: str,
    is_active: bool,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update dealer active status
    """
    dealer_service = DealerService(db)
    
    dealer = dealer_service.get_dealer_by_code(dealer_code)
    if not dealer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dealer not found"
        )
    
    dealer_service.update_dealer_status(dealer_code, is_active)
    status_text = "activated" if is_active else "deactivated"
    
    return {"message": f"Dealer {status_text} successfully"}