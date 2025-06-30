
# # # api/v1/endpoints/customers.py
# # from typing import List, Optional
# # from fastapi import APIRouter, Depends, HTTPException, Query
# # from sqlalchemy.orm import Session
# # from sqlalchemy import func, and_
# # from core.dependencies import get_db, get_current_user
# # from models.customer import Customer
# # from schemas.customer import CustomerCreate, CustomerUpdate, CustomerResponse, CustomerSearch
# # import math

# # router = APIRouter()

# # @router.get("/", response_model=List[CustomerResponse])
# # def get_customers(
# #     skip: int = Query(0, ge=0),
# #     limit: int = Query(100, ge=1, le=1000),
# #     city: Optional[str] = Query(None),
# #     search: Optional[str] = Query(None),
# #     db: Session = Depends(get_db),
# #     current_user = Depends(get_current_user)
# # ):
# #     """Get all customers with optional filtering"""
# #     query = db.query(Customer)
    
# #     if city:
# #         query = query.filter(Customer.city.ilike(f"%{city}%"))
    
# #     if search:
# #         query = query.filter(
# #             func.or_(
# #                 Customer.contact.ilike(f"%{search}%"),
# #                 Customer.city.ilike(f"%{search}%")
# #             )
# #         )
    
# #     customers = query.offset(skip).limit(limit).all()
# #     return customers

# # @router.get("/{customer_id}", response_model=CustomerResponse)
# # def get_customer(
# #     customer_id: int,
# #     db: Session = Depends(get_db),
# #     current_user = Depends(get_current_user)
# # ):
# #     """Get customer by ID"""
# #     customer = db.query(Customer).filter(Customer.no == customer_id).first()
# #     if not customer:
# #         raise HTTPException(status_code=404, detail="Customer not found")
# #     return customer

# # @router.post("/", response_model=CustomerResponse)
# # def create_customer(
# #     customer: CustomerCreate,
# #     db: Session = Depends(get_db),
# #     current_user = Depends(get_current_user)
# # ):
# #     """Create new customer"""
# #     db_customer = Customer(**customer.dict())
# #     db.add(db_customer)
# #     db.commit()
# #     db.refresh(db_customer)
# #     return db_customer

# # @router.put("/{customer_id}", response_model=CustomerResponse)
# # def update_customer(
# #     customer_id: int,
# #     customer: CustomerUpdate,
# #     db: Session = Depends(get_db),
# #     current_user = Depends(get_current_user)
# # ):
# #     """Update customer"""
# #     db_customer = db.query(Customer).filter(Customer.no == customer_id).first()
# #     if not db_customer:
# #         raise HTTPException(status_code=404, detail="Customer not found")
    
# #     for field, value in customer.dict(exclude_unset=True).items():
# #         setattr(db_customer, field, value)
    
# #     db.commit()
# #     db.refresh(db_customer)
# #     return db_customer

# # @router.delete("/{customer_id}")
# # def delete_customer(
# #     customer_id: int,
# #     db: Session = Depends(get_db),
# #     current_user = Depends(get_current_user)
# # ):
# #     """Delete customer"""
# #     db_customer = db.query(Customer).filter(Customer.no == customer_id).first()
# #     if not db_customer:
# #         raise HTTPException(status_code=404, detail="Customer not found")
    
# #     db.delete(db_customer)
# #     db.commit()
# #     return {"message": "Customer deleted successfully"}

# # @router.get("/nearby/{latitude}/{longitude}")
# # def get_nearby_customers(
# #     latitude: float,
# #     longitude: float,
# #     radius_km: float = Query(10, ge=0.1, le=100),
# #     db: Session = Depends(get_db),
# #     current_user = Depends(get_current_user)
# # ):
# #     """Get customers within specified radius"""
# #     # Haversine formula for distance calculation
# #     customers = db.query(Customer).filter(
# #         Customer.latitude.isnot(None),
# #         Customer.longitude.isnot(None)
# #     ).all()
    
# #     nearby_customers = []
# #     for customer in customers:
# #         distance = calculate_distance(latitude, longitude, customer.latitude, customer.longitude)
# #         if distance <= radius_km:
# #             customer_dict = customer.__dict__.copy()
# #             customer_dict['distance_km'] = round(distance, 2)
# #             nearby_customers.append(customer_dict)
    
# #     return sorted(nearby_customers, key=lambda x: x['distance_km'])

# # def calculate_distance(lat1, lon1, lat2, lon2):
# #     """Calculate distance between two points using Haversine formula"""
# #     R = 6371  # Earth's radius in kilometers
    
# #     lat1_rad = math.radians(lat1)
# #     lon1_rad = math.radians(lon1)
# #     lat2_rad = math.radians(lat2)
# #     lon2_rad = math.radians(lon2)
    
# #     dlat = lat2_rad - lat1_rad
# #     dlon = lon2_rad - lon1_rad
    
# #     a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
# #     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
# #     return R * c

# # api/v1/endpoints/customers.py
# from typing import List, Optional
# from fastapi import APIRouter, Depends, HTTPException, status, Query
# from sqlalchemy.orm import Session
# from sqlalchemy import and_, or_, func
# from geopy.distance import geodesic
# import math

# from core.dependencies import get_db, get_current_user
# from models.user import User
# from models.customer import Customer
# from schemas.customer import CustomerCreate, CustomerUpdate, CustomerResponse, CustomerSearch
# from schemas.base import APIResponse, PaginatedResponse
# from utils.pagination import paginate
# from utils.location import calculate_distance, get_nearby_locations

# router = APIRouter(prefix="/customers", tags=["Customers"])

# @router.get("/", response_model=PaginatedResponse[CustomerResponse])
# async def get_customers(
#     page: int = Query(1, ge=1),
#     per_page: int = Query(10, ge=1, le=100),
#     search: Optional[str] = Query(None),
#     city: Optional[str] = Query(None),
#     territory_code: Optional[str] = Query(None),
#     active_only: bool = Query(True),
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """
#     Get paginated list of customers with filtering options
#     """
#     try:
#         query = db.query(Customer)
        
#         # Apply filters
#         if active_only:
#             query = query.filter(Customer.is_active == True)
        
#         if search:
#             search_term = f"%{search}%"
#             query = query.filter(
#                 or_(
#                     Customer.name.ilike(search_term),
#                     Customer.contact.ilike(search_term),
#                     Customer.city.ilike(search_term),
#                     Customer.no.ilike(search_term)
#                 )
#             )
        
#         if city:
#             query = query.filter(Customer.city.ilike(f"%{city}%"))
        
#         if territory_code:
#             query = query.filter(Customer.territory_code == territory_code)
        
#         # For dealers, filter by their territory
#         if current_user.role == "dealer" and current_user.territory_code:
#             query = query.filter(Customer.territory_code == current_user.territory_code)
        
#         # Paginate results
#         paginated_result = paginate(query, page, per_page)
        
#         return PaginatedResponse(
#             success=True,
#             message="Customers retrieved successfully",
#             data=[CustomerResponse.from_orm(customer) for customer in paginated_result.items],
#             pagination=paginated_result.pagination
#         )
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to retrieve customers: {str(e)}"
#         )

# @router.get("/search", response_model=APIResponse[List[CustomerResponse]])
# async def search_customers(
#     q: str = Query(..., min_length=2),
#     limit: int = Query(20, ge=1, le=100),
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """
#     Quick search customers by name, contact, or city
#     """
#     try:
#         search_term = f"%{q}%"
#         query = db.query(Customer).filter(
#             and_(
#                 Customer.is_active == True,
#                 or_(
#                     Customer.name.ilike(search_term),
#                     Customer.contact.ilike(search_term),
#                     Customer.city.ilike(search_term),
#                     Customer.no.ilike(search_term)
#                 )
#             )
#         )
        
#         # For dealers, filter by their territory
#         if current_user.role == "dealer" and current_user.territory_code:
#             query = query.filter(Customer.territory_code == current_user.territory_code)
        
#         customers = query.limit(limit).all()
        
#         return APIResponse(
#             success=True,
#             message=f"Found {len(customers)} customers",
#             data=[CustomerResponse.from_orm(customer) for customer in customers]
#         )
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Search failed: {str(e)}"
#         )

# @router.get("/nearby", response_model=APIResponse[List[CustomerResponse]])
# async def get_nearby_customers(
#     latitude: float = Query(..., ge=-90, le=90),
#     longitude: float = Query(..., ge=-180, le=180),
#     radius_km: float = Query(10, ge=0.1, le=100),
#     limit: int = Query(50, ge=1, le=100),
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """
#     Get customers within specified radius of given coordinates
#     """
#     try:
#         # Get all customers with coordinates
#         query = db.query(Customer).filter(
#             and_(
#                 Customer.is_active == True,
#                 Customer.latitude.isnot(None),
#                 Customer.longitude.isnot(None)
#             )
#         )
        
#         # For dealers, filter by their territory
#         if current_user.role == "dealer" and current_user.territory_code:
#             query = query.filter(Customer.territory_code == current_user.territory_code)
        
#         customers = query.all()
        
#         # Calculate distances and filter
#         nearby_customers = []
#         for customer in customers:
#             try:
#                 distance = geodesic(
#                     (latitude, longitude),
#                     (customer.latitude, customer.longitude)
#                 ).kilometers
                
#                 if distance <= radius_km:
#                     customer_data = CustomerResponse.from_orm(customer)
#                     customer_data.distance_km = round(distance, 2)
#                     nearby_customers.append(customer_data)
#             except:
#                 continue
        
#         # Sort by distance and limit
#         nearby_customers.sort(key=lambda x: x.distance_km or float('inf'))
#         nearby_customers = nearby_customers[:limit]
        
#         return APIResponse(
#             success=True,
#             message=f"Found {len(nearby_customers)} customers within {radius_km}km",
#             data=nearby_customers
#         )
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to find nearby customers: {str(e)}"
#         )

# @router.get("/by-city", response_model=APIResponse[List[dict]])
# async def get_customers_by_city(
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """
#     Get customer count grouped by city
#     """
#     try:
#         query = db.query(
#             Customer.city,
#             func.count(Customer.id).label('count')
#         ).filter(Customer.is_active == True)
        
#         # For dealers, filter by their territory
#         if current_user.role == "dealer" and current_user.territory_code:
#             query = query.filter(Customer.territory_code == current_user.territory_code)
        
#         city_counts = query.group_by(Customer.city).order_by(func.count(Customer.id).desc()).all()
        
#         result = [
#             {
#                 "city": city,
#                 "customer_count": count
#             }
#             for city, count in city_counts
#         ]
        
#         return APIResponse(
#             success=True,
#             message="Customer distribution by city retrieved successfully",
#             data=result
#         )
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to get customer distribution: {str(e)}"
#         )

# @router.get("/{customer_id}", response_model=APIResponse[CustomerResponse])
# async def get_customer_by_id(
#     customer_id: int,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """
#     Get customer by ID
#     """
#     try:
#         query = db.query(Customer).filter(Customer.id == customer_id)
        
#         # For dealers, filter by their territory
#         if current_user.role == "dealer" and current_user.territory_code:
#             query = query.filter(Customer.territory_code == current_user.territory_code)
        
#         customer = query.first()
        
#         if not customer:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="Customer not found"
#             )
        
#         return APIResponse(
#             success=True,
#             message="Customer retrieved successfully",
#             data=CustomerResponse.from_orm(customer)
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to retrieve customer: {str(e)}"
#         )

# @router.post("/", response_model=APIResponse[CustomerResponse])
# async def create_customer(
#     customer_data: CustomerCreate,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """
#     Create new customer (Admin only)
#     """
#     try:
#         if current_user.role != "admin":
#             raise HTTPException(
#                 status_code=status.HTTP_403_FORBIDDEN,
#                 detail="Only admins can create customers"
#             )
        
#         # Check if customer with same contact already exists
#         existing_customer = db.query(Customer).filter(
#             Customer.contact == customer_data.contact
#         ).first()
        
#         if existing_customer:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Customer with this contact already exists"
#             )
        
#         # Create new customer
#         new_customer = Customer(**customer_data.dict())
#         db.add(new_customer)
#         db.commit()
#         db.refresh(new_customer)
        
#         return APIResponse(
#             success=True,
#             message="Customer created successfully",
#             data=CustomerResponse.from_orm(new_customer)
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to create customer: {str(e)}"
#         )

# @router.put("/{customer_id}", response_model=APIResponse[CustomerResponse])
# async def update_customer(
#     customer_id: int,
#     customer_data: CustomerUpdate,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """
#     Update customer information (Admin only)
#     """
#     try:
#         if current_user.role != "admin":
#             raise HTTPException(
#                 status_code=status.HTTP_403_FORBIDDEN,
#                 detail="Only admins can update customers"
#             )
        
#         customer = db.query(Customer).filter(Customer.id == customer_id).first()
        
#         if not customer:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="Customer not found"
#             )
        
#         # Update customer fields
#         update_data = customer_data.dict(exclude_unset=True)
#         for field, value in update_data.items():
#             setattr(customer, field, value)
        
#         db.commit()
#         db.refresh(customer)
        
#         return APIResponse(
#             success=True,
#             message="Customer updated successfully",
#             data=CustomerResponse.from_orm(customer)
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to update customer: {str(e)}"
#         )

# @router.delete("/{customer_id}", response_model=APIResponse[dict])
# async def delete_customer(
#     customer_id: int,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """
#     Soft delete customer (Admin only)
#     """
#     try:
#         if current_user.role != "admin":
#             raise HTTPException(
#                 status_code=status.HTTP_403_FORBIDDEN,
#                 detail="Only admins can delete customers"
#             )
        
#         customer = db.query(Customer).filter(Customer.id == customer_id).first()
        
#         if not customer:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="Customer not found"
#             )
        
#         # Soft delete
#         customer.is_active = False
#         db.commit()
        
#         return APIResponse(
#             success=True,
#             message="Customer deleted successfully",
#             data={"message": "Customer has been deactivated"}
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to delete customer: {str(e)}"
#         )

# @router.get("/{customer_id}/location-history", response_model=APIResponse[List[dict]])
# async def get_customer_location_history(
#     customer_id: int,
#     days: int = Query(30, ge=1, le=365),
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """
#     Get customer's location update history
#     """
#     try:
#         # This would require a customer_location_history table
#         # For now, return the current location
#         customer = db.query(Customer).filter(Customer.id == customer_id).first()
        
#         if not customer:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="Customer not found"
#             )
        
#         # For dealers, check territory access
#         if current_user.role == "dealer" and current_user.territory_code:
#             if customer.territory_code != current_user.territory_code:
#                 raise HTTPException(
#                     status_code=status.HTTP_403_FORBIDDEN,
#                     detail="Access denied to this customer"
#                 )
        
#         location_history = []
#         if customer.latitude and customer.longitude:
#             location_history.append({
#                 "latitude": customer.latitude,
#                 "longitude": customer.longitude,
#                 "city": customer.city,
#                 "updated_at": customer.updated_at or customer.created_at,
#                 "address": f"{customer.city}"
#             })
        
#         return APIResponse(
#             success=True,
#             message="Customer location history retrieved successfully",
#             data=location_history
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to retrieve location history: {str(e)}"
#         )

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from geopy.distance import geodesic

from core.dependencies import get_db, get_current_user
from models.user import User
from models.customer import Customer
from schemas.customer import (
    CustomerCreate,
    CustomerUpdate,
    CustomerResponse,
    CustomerSearchParams,
    CustomerLocationResponse,
    NearbyCustomersRequest
)
from services.customer_service import CustomerService
from utils.geo_utils import calculate_distance, get_nearby_locations

router = APIRouter()

@router.get("/", response_model=List[CustomerResponse])
async def get_customers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000),
    city: Optional[str] = Query(None),
    territory_code: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all customers with optional filtering
    """
    customer_service = CustomerService(db)
    
    filters = {}
    if city:
        filters['city'] = city
    if territory_code:
        filters['territory_code'] = territory_code
    if search:
        filters['search'] = search
    
    customers = customer_service.get_customers(
        skip=skip,
        limit=limit,
        filters=filters
    )
    
    return [CustomerResponse.from_orm(customer) for customer in customers]

@router.get("/count")
async def get_customers_count(
    city: Optional[str] = Query(None),
    territory_code: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get total count of customers
    """
    customer_service = CustomerService(db)
    
    filters = {}
    if city:
        filters['city'] = city
    if territory_code:
        filters['territory_code'] = territory_code
    
    count = customer_service.get_customers_count(filters)
    return {"total_customers": count}

@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer(
    customer_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get customer by ID
    """
    customer_service = CustomerService(db)
    
    customer = customer_service.get_customer_by_id(customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    return CustomerResponse.from_orm(customer)

@router.post("/", response_model=CustomerResponse, status_code=status.HTTP_201_CREATED)
async def create_customer(
    customer_data: CustomerCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new customer
    """
    customer_service = CustomerService(db)
    
    # Check if customer ID already exists
    existing_customer = customer_service.get_customer_by_id(customer_data.customer_id)
    if existing_customer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Customer ID already exists"
        )
    
    customer = customer_service.create_customer(customer_data)
    return CustomerResponse.from_orm(customer)

@router.put("/{customer_id}", response_model=CustomerResponse)
async def update_customer(
    customer_id: str,
    customer_data: CustomerUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update customer information
    """
    customer_service = CustomerService(db)
    
    customer = customer_service.get_customer_by_id(customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    updated_customer = customer_service.update_customer(customer_id, customer_data)
    return CustomerResponse.from_orm(updated_customer)

@router.delete("/{customer_id}")
async def delete_customer(
    customer_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a customer
    """
    customer_service = CustomerService(db)
    
    customer = customer_service.get_customer_by_id(customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    customer_service.delete_customer(customer_id)
    return {"message": "Customer deleted successfully"}

@router.get("/search/advanced", response_model=List[CustomerResponse])
async def advanced_customer_search(
    search_params: CustomerSearchParams = Depends(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Advanced customer search with multiple filters
    """
    customer_service = CustomerService(db)
    
    customers = customer_service.advanced_search(search_params)
    return [CustomerResponse.from_orm(customer) for customer in customers]

@router.get("/location/nearby", response_model=List[CustomerLocationResponse])
async def get_nearby_customers(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(10, gt=0, le=100),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get customers within a specified radius from a location
    """
    customer_service = CustomerService(db)
    
    nearby_customers = customer_service.get_nearby_customers(
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        limit=limit
    )
    
    return [
        CustomerLocationResponse(
            **customer.__dict__,
            distance_km=calculate_distance(
                latitude, longitude,
                customer.latitude, customer.longitude
            )
        )
        for customer in nearby_customers
    ]

@router.get("/cities/list")
async def get_customer_cities(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get list of all unique cities where customers are located
    """
    customer_service = CustomerService(db)
    
    cities = customer_service.get_unique_cities()
    return {"cities": cities}

@router.get("/territories/list")
async def get_territories(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get list of all unique territory codes
    """
    customer_service = CustomerService(db)
    
    territories = customer_service.get_unique_territories()
    return {"territories": territories}

@router.get("/stats/city")
async def get_customers_by_city_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get customer count statistics by city
    """
    customer_service = CustomerService(db)
    
    stats = customer_service.get_customer_stats_by_city()
    return {"city_stats": stats}

@router.get("/stats/territory")
async def get_customers_by_territory_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get customer count statistics by territory
    """
    customer_service = CustomerService(db)
    
    stats = customer_service.get_customer_stats_by_territory()
    return {"territory_stats": stats}

@router.post("/import/bulk", status_code=status.HTTP_201_CREATED)
async def bulk_import_customers(
    customers_data: List[CustomerCreate],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Bulk import customers from a list
    """
    customer_service = CustomerService(db)
    
    results = customer_service.bulk_import_customers(customers_data)
    
    return {
        "message": "Bulk import completed",
        "imported_count": results["imported_count"],
        "failed_count": results["failed_count"],
        "errors": results["errors"]
    }

@router.get("/{customer_id}/orders-summary")
async def get_customer_orders_summary(
    customer_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get order summary for a specific customer
    """
    customer_service = CustomerService(db)
    
    customer = customer_service.get_customer_by_id(customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    summary = customer_service.get_customer_orders_summary(customer_id)
    return summary

@router.get("/location/clusters")
async def get_customer_location_clusters(
    min_cluster_size: int = Query(5, ge=2),
    max_distance_km: float = Query(5, gt=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get customer location clusters for visualization
    """
    customer_service = CustomerService(db)
    
    clusters = customer_service.get_location_clusters(
        min_cluster_size=min_cluster_size,
        max_distance_km=max_distance_km
    )
    
    return {"clusters": clusters}

@router.get("/export/csv")
async def export_customers_csv(
    city: Optional[str] = Query(None),
    territory_code: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Export customers data as CSV
    """
    customer_service = CustomerService(db)
    
    filters = {}
    if city:
        filters['city'] = city
    if territory_code:
        filters['territory_code'] = territory_code
    
    csv_data = customer_service.export_customers_csv(filters)
    
    from fastapi.responses import Response
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=customers.csv"}
    )

@router.post("/geocode/update")
async def update_customer_geocoding(
    customer_id: Optional[str] = Query(None),
    force_update: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update geocoding (latitude/longitude) for customers
    """
    customer_service = CustomerService(db)
    
    if customer_id:
        # Update specific customer
        customer = customer_service.get_customer_by_id(customer_id)
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        updated = customer_service.update_customer_geocoding(customer_id, force_update)
        return {"message": f"Geocoding updated for customer {customer_id}", "updated": updated}
    else:
        # Update all customers
        results = customer_service.bulk_update_geocoding(force_update)
        return {
            "message": "Bulk geocoding update completed",
            "updated_count": results["updated_count"],
            "failed_count": results["failed_count"]
        }