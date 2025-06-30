from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from decimal import Decimal

class CustomerBase(BaseModel):
    city: str = Field(..., min_length=1, max_length=100)
    contact: Optional[str] = Field(None, max_length=50)
    telex_no: Optional[str] = Field(None, max_length=20)
    document_sending_profile: Optional[str] = Field(None, max_length=100)
    ship_to_code: Optional[str] = Field(None, max_length=20)
    our_account_no: Optional[str] = Field(None, max_length=30)
    territory_code: Optional[str] = Field(None, max_length=10)
    global_dimension_1_code: Optional[str] = Field(None, max_length=20)
    global_dimension_2_code: Optional[str] = Field(None, max_length=20)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)

class CustomerCreate(CustomerBase):
    pass

class CustomerUpdate(BaseModel):
    city: Optional[str] = Field(None, min_length=1, max_length=100)
    contact: Optional[str] = Field(None, max_length=50)
    telex_no: Optional[str] = Field(None, max_length=20)
    document_sending_profile: Optional[str] = Field(None, max_length=100)
    ship_to_code: Optional[str] = Field(None, max_length=20)
    our_account_no: Optional[str] = Field(None, max_length=30)
    territory_code: Optional[str] = Field(None, max_length=10)
    global_dimension_1_code: Optional[str] = Field(None, max_length=20)
    global_dimension_2_code: Optional[str] = Field(None, max_length=20)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)

class CustomerInDB(CustomerBase):
    id: int
    no: str
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class Customer(CustomerInDB):
    pass

class CustomerWithSales(Customer):
    total_sales: Optional[Decimal] = Field(default=0)
    order_count: Optional[int] = Field(default=0)
    last_order_date: Optional[datetime] = None
    avg_order_value: Optional[Decimal] = Field(default=0)

class CustomerLocationResponse(BaseModel):
    id: int
    no: str
    city: str
    latitude: Optional[float]
    longitude: Optional[float]
    distance_km: Optional[float] = None

class CustomerSearchFilter(BaseModel):
    city: Optional[str] = None
    territory_code: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    radius_km: Optional[float] = Field(None, gt=0, le=1000)
    has_coordinates: Optional[bool] = None
    
    @validator('radius_km')
    def validate_radius_with_coordinates(cls, v, values):
        if v is not None:
            if 'latitude' not in values or 'longitude' not in values:
                raise ValueError('Latitude and longitude required when using radius filter')
            if values.get('latitude') is None or values.get('longitude') is None:
                raise ValueError('Latitude and longitude must be provided when using radius filter')
        return v

class CustomerStats(BaseModel):
    total_customers: int
    customers_with_coordinates: int
    customers_by_city: dict
    customers_by_territory: dict
    geographic_coverage: dict