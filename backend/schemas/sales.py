from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

class OrderStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class SalesOrderBase(BaseModel):
    code: str = Field(..., min_length=1, max_length=50)
    distributor_code: str = Field(..., min_length=1, max_length=20)
    user_code: str = Field(..., min_length=1, max_length=20)
    user_name: str = Field(..., min_length=1, max_length=100)
    final_value: Decimal = Field(..., gt=0)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    erp_order_number: Optional[str] = Field(None, max_length=50)
    tour_code: Optional[str] = Field(None, max_length=20)
    days: Optional[str] = Field(None, max_length=20)
    total: Optional[Decimal] = Field(None, gt=0)

class SalesOrderCreate(SalesOrderBase):
    date: datetime = Field(default_factory=datetime.now)

class SalesOrderUpdate(BaseModel):
    distributor_code: Optional[str] = Field(None, min_length=1, max_length=20)
    user_code: Optional[str] = Field(None, min_length=1, max_length=20)
    user_name: Optional[str] = Field(None, min_length=1, max_length=100)
    final_value: Optional[Decimal] = Field(None, gt=0)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    erp_order_number: Optional[str] = Field(None, max_length=50)
    status: Optional[OrderStatus] = None
    tour_code: Optional[str] = Field(None, max_length=20)
    days: Optional[str] = Field(None, max_length=20)
    total: Optional[Decimal] = Field(None, gt=0)

class SalesOrderInDB(SalesOrderBase):
    id: int
    date: datetime
    creation_date: datetime
    submitted_date: Optional[datetime]
    status: OrderStatus = OrderStatus.DRAFT
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class SalesOrder(SalesOrderInDB):
    pass

class SalesOrderWithDetails(SalesOrder):
    customer_city: Optional[str] = None
    dealer_division: Optional[str] = None
    distance_from_customer: Optional[float] = None
    order_processing_time: Optional[int] = None  # minutes

class SalesAggregation(BaseModel):
    period: str
    total_sales: Decimal
    order_count: int
    unique_customers: int
    unique_dealers: int
    avg_order_value: Decimal
    max_order_value: Decimal
    min_order_value: Decimal

class SalesByPeriod(BaseModel):
    date: date
    total_sales: Decimal
    order_count: int
    avg_order_value: Decimal

class SalesByDealer(BaseModel):
    user_code: str
    user_name: str
    division_code: str
    total_sales: Decimal
    order_count: int
    unique_customers: int
    avg_order_value: Decimal
    performance_score: float

class SalesByCustomer(BaseModel):
    distributor_code: str
    customer_city: Optional[str]
    total_purchases: Decimal
    order_count: int
    avg_order_value: Decimal
    last_order_date: datetime
    customer_value_score: float

class SalesFilter(BaseModel):
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    user_code: Optional[str] = None
    distributor_code: Optional[str] = None
    division_code: Optional[str] = None
    territory_code: Optional[str] = None
    min_value: Optional[Decimal] = Field(None, gt=0)
    max_value: Optional[Decimal] = Field(None, gt=0)
    status: Optional[OrderStatus] = None
    has_coordinates: Optional[bool] = None
    city: Optional[str] = None
    
    @validator('date_to')
    def validate_date_range(cls, v, values):
        if v and 'date_from' in values and values['date_from']:
            if v < values['date_from']:
                raise ValueError('date_to must be after date_from')
        return v
    
    @validator('max_value')
    def validate_value_range(cls, v, values):
        if v and 'min_value' in values and values['min_value']:
            if v < values['min_value']:
                raise ValueError('max_value must be greater than min_value')
        return v

class SalesStats(BaseModel):
    total_sales: Decimal
    total_orders: int
    avg_order_value: Decimal
    unique_customers: int
    unique_dealers: int
    orders_with_coordinates: int
    geographic_coverage: int
    top_selling_dealers: List[SalesByDealer]
    top_customers: List[SalesByCustomer]

class SalesTrend(BaseModel):
    period: str
    sales_data: List[SalesByPeriod]
    growth_rate: float
    trend_direction: str  # "up", "down", "stable"
    seasonal_pattern: Optional[Dict[str, float]] = None

class RegionalSalesReport(BaseModel):
    region: str
    total_sales: Decimal
    order_count: int
    dealer_count: int
    customer_count: int
    avg_order_value: Decimal
    market_share: float
    growth_rate: float

class SalesTargetVsActual(BaseModel):
    period: str
    target: Decimal
    actual: Decimal
    achievement_rate: float
    variance: Decimal
    status: str  # "above", "on_target", "below"

class BulkOrderRequest(BaseModel):
    orders: List[SalesOrderCreate] = Field(..., min_items=1, max_items=100)
    
class BulkOrderResponse(BaseModel):
    successful_orders: List[SalesOrder]
    failed_orders: List[Dict[str, str]]
    success_count: int
    failure_count: int

class OrderValidationError(BaseModel):
    field: str
    message: str
    value: any

class SalesOrderValidation(BaseModel):
    is_valid: bool
    errors: List[OrderValidationError]
    warnings: List[str]