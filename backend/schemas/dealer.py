from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

class DealerStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

class DealerBase(BaseModel):
    user_code: str = Field(..., min_length=1, max_length=20)
    user_name: str = Field(..., min_length=1, max_length=100)
    division_code: str = Field(..., min_length=1, max_length=10)
    territory_code: Optional[str] = Field(None, max_length=10)
    status: DealerStatus = DealerStatus.ACTIVE
    phone: Optional[str] = Field(None, max_length=15)
    email: Optional[str] = Field(None, max_length=100)
    hire_date: Optional[date] = None

class DealerCreate(DealerBase):
    pass

class DealerUpdate(BaseModel):
    user_name: Optional[str] = Field(None, min_length=1, max_length=100)
    division_code: Optional[str] = Field(None, min_length=1, max_length=10)
    territory_code: Optional[str] = Field(None, max_length=10)
    status: Optional[DealerStatus] = None
    phone: Optional[str] = Field(None, max_length=15)
    email: Optional[str] = Field(None, max_length=100)
    hire_date: Optional[date] = None

class DealerInDB(DealerBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class Dealer(DealerInDB):
    pass

class DealerPerformanceMetrics(BaseModel):
    dealer_id: int
    user_code: str
    user_name: str
    period_start: date
    period_end: date
    total_sales: Decimal = Field(default=0)
    order_count: int = Field(default=0)
    unique_customers: int = Field(default=0)
    avg_order_value: Decimal = Field(default=0)
    distance_traveled_km: Optional[float] = Field(default=0)
    working_days: int = Field(default=0)
    sales_per_day: Decimal = Field(default=0)
    conversion_rate: float = Field(default=0.0, ge=0, le=100)

class DealerWithPerformance(Dealer):
    current_month_sales: Decimal = Field(default=0)
    current_month_orders: int = Field(default=0)
    last_activity: Optional[datetime] = None
    performance_score: Optional[float] = Field(default=0.0, ge=0, le=100)
    rank_in_division: Optional[int] = None

class DealerTerritoryAssignment(BaseModel):
    dealer_id: int
    user_code: str
    territory_code: str
    assigned_date: date
    coverage_area_km2: Optional[float] = None
    customer_count: int = Field(default=0)
    target_monthly_sales: Optional[Decimal] = None

class DealerActivitySummary(BaseModel):
    dealer_id: int
    user_code: str
    date: date
    gps_points_count: int = Field(default=0)
    distance_traveled_km: float = Field(default=0.0)
    unique_locations_visited: int = Field(default=0)
    orders_created: int = Field(default=0)
    total_sales: Decimal = Field(default=0)
    first_activity: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    working_hours: Optional[float] = Field(default=0.0)

class DealerSearchFilter(BaseModel):
    division_code: Optional[str] = None
    territory_code: Optional[str] = None
    status: Optional[DealerStatus] = None
    user_name: Optional[str] = None
    performance_min: Optional[float] = Field(None, ge=0, le=100)
    performance_max: Optional[float] = Field(None, ge=0, le=100)
    hire_date_from: Optional[date] = None
    hire_date_to: Optional[date] = None

class DealerStats(BaseModel):
    total_dealers: int
    active_dealers: int
    dealers_by_division: Dict[str, int]
    dealers_by_territory: Dict[str, int]
    avg_performance_score: float
    top_performers: List[Dict[str, any]]
    bottom_performers: List[Dict[str, any]]

class DealerRanking(BaseModel):
    rank: int
    dealer_id: int
    user_code: str
    user_name: str
    division_code: str
    total_sales: Decimal
    order_count: int
    performance_score: float

class DealerComparisonRequest(BaseModel):
    dealer_ids: List[int] = Field(..., min_items=2, max_items=10)
    period_start: date
    period_end: date
    metrics: List[str] = Field(default=["total_sales", "order_count", "unique_customers"])

class DealerComparisonResponse(BaseModel):
    period_start: date
    period_end: date
    dealers: List[DealerPerformanceMetrics]
    comparison_summary: Dict[str, any]