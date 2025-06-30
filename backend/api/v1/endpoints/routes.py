
# api/v1/endpoints/routes.py
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from core.dependencies import get_db, get_current_user
from models.gps import GPSData
from models.customer import Customer
from schemas.route import RouteOptimization, GPSTrackingResponse
from datetime import datetime, date, timedelta
import math

router = APIRouter()

@router.get("/gps-tracking/{dealer_code}")
def get_gps_tracking(
    dealer_code: str,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get GPS tracking data for a dealer"""
    query = db.query(GPSData).filter(GPSData.user_code == dealer_code)
    
    if start_date:
        query = query.filter(GPSData.received_date >= start_date)
    if end_date:
        query = query.filter(GPSData.received_date <= end_date)
    
    gps_data = query.order_by(GPSData.received_date).all()
    
    if not gps_data:
        raise HTTPException(status_code=404, detail="No GPS data found for dealer")
    
    return {
        "dealer_code": dealer_code,
        "dealer_name": gps_data[0].user_name if gps_data else None,
        "total_points": len(gps_data),
        "tracking_data": [
            {
                "latitude": float(point.latitude),
                "longitude": float(point.longitude),
                "timestamp": point.received_date,
                "tour_code": point.tour_code,
                "division_code": point.division_code
            }
            for point in gps_data
        ]
    }

@router.get("/route-history/{dealer_code}")
def get_route_history(
    dealer_code: str,
    days: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get route history for a dealer"""
    start_date = datetime.now() - timedelta(days=days)
    
    # Get GPS data grouped by tour/day
    routes = db.query(
        GPSData.tour_code,
        func.date(GPSData.received_date).label('date'),
        func.min(GPSData.received_date).label('start_time'),
        func.max(GPSData.received_date).label('end_time'),
        func.count(GPSData.id).label('total_points')
    ).filter(
        GPSData.user_code == dealer_code,
        GPSData.received_date >= start_date
    ).group_by(
        GPSData.tour_code, func.date(GPSData.received_date)
    ).order_by(desc('date')).all()
    
    route_details = []
    for route in routes:
        # Get detailed GPS points for this route
        gps_points = db.query(GPSData).filter(
            GPSData.user_code == dealer_code,
            GPSData.tour_code == route.tour_code,
            func.date(GPSData.received_date) == route.date
        ).order_by(GPSData.received_date).all()
        
        # Calculate total distance
        total_distance = calculate_route_distance(gps_points)
        
        route_details.append({
            "tour_code": route.tour_code,
            "date": str(route.date),
            "start_time": route.start_time,
            "end_time": route.end_time,
            "duration_hours": (route.end_time - route.start_time).total_seconds() / 3600,
            "total_points": route.total_points,
            "total_distance_km": round(total_distance, 2),
            "gps_points": [
                {
                    "latitude": float(point.latitude),
                    "longitude": float(point.longitude),
                    "timestamp": point.received_date
                }
                for point in gps_points
            ]
        })
    
    return {
        "dealer_code": dealer_code,
        "period_days": days,
        "routes": route_details
    }

@router.post("/optimize-route")
def optimize_route(
    dealer_code: str,
    customer_codes: List[str],
    start_location: Optional[dict] = None,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Optimize route for visiting customers"""
    # Get customer locations
    customers = db.query(Customer).filter(
        Customer.no.in_(customer_codes),
        Customer.latitude.isnot(None),
        Customer.longitude.isnot(None)
    ).all()
    
    if len(customers) != len(customer_codes):
        raise HTTPException(
            status_code=400, 
            detail="Some customers not found or missing location data"
        )
    
    # If no start location provided, use first customer
    if not start_location:
        start_location = {
            "latitude": float(customers[0].latitude),
            "longitude": float(customers[0].longitude)
        }
    
    # Simple nearest neighbor optimization
    optimized_route = nearest_neighbor_tsp(
        customers, 
        start_location["latitude"], 
        start_location["longitude"]
    )
    
    # Calculate total distance and estimated time
    total_distance = 0
    route_points = [start_location]
    
    for i, customer in enumerate(optimized_route):
        if i == 0:
            prev_lat, prev_lon = start_location["latitude"], start_location["longitude"]
        else:
            prev_lat, prev_lon = float(optimized_route[i-1].latitude), float(optimized_route[i-1].longitude)
        
        distance = calculate_distance(prev_lat, prev_lon, float(customer.latitude), float(customer.longitude))
        total_distance += distance
        
        route_points.append({
            "latitude": float(customer.latitude),
            "longitude": float(customer.longitude),
            "customer_id": customer.no,
            "customer_name": customer.contact,
            "city": customer.city,
            "distance_from_previous": round(distance, 2)
        })
    
    estimated_time_hours = total_distance / 40  # Assuming 40 km/h average speed
    
    return {
        "dealer_code": dealer_code,
        "total_customers": len(customers),
        "total_distance_km": round(total_distance, 2),
        "estimated_time_hours": round(estimated_time_hours, 2),
        "optimized_route": route_points
    }

@router.get("/live-tracking/{dealer_code}")
def get_live_tracking(
    dealer_code: str,
    minutes: int = Query(60, ge=5, le=1440),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get live tracking data for the last N minutes"""
    start_time = datetime.now() - timedelta(minutes=minutes)
    
    gps_data = db.query(GPSData).filter(
        GPSData.user_code == dealer_code,
        GPSData.received_date >= start_time
    ).order_by(desc(GPSData.received_date)).all()
    
    if not gps_data:
        return {
            "dealer_code": dealer_code,
            "status": "offline",
            "last_seen": None,
            "current_location": None,
            "recent_path": []
        }
    
    latest_point = gps_data[0]
    
    return {
        "dealer_code": dealer_code,
        "dealer_name": latest_point.user_name,
        "status": "online" if (datetime.now() - latest_point.received_date).total_seconds() < 300 else "offline",
        "last_seen": latest_point.received_date,
        "current_location": {
            "latitude": float(latest_point.latitude),
            "longitude": float(latest_point.longitude),
            "tour_code": latest_point.tour_code
        },
        "recent_path": [
            {
                "latitude": float(point.latitude),
                "longitude": float(point.longitude),
                "timestamp": point.received_date
            }
            for point in gps_data[:50]  # Last 50 points
        ]
    }

def calculate_route_distance(gps_points):
    """Calculate total distance of a route"""
    if len(gps_points) < 2:
        return 0
    
    total_distance = 0
    for i in range(1, len(gps_points)):
        distance = calculate_distance(
            float(gps_points[i-1].latitude), float(gps_points[i-1].longitude),
            float(gps_points[i].latitude), float(gps_points[i].longitude)
        )
        total_distance += distance
    
    return total_distance

def nearest_neighbor_tsp(customers, start_lat, start_lon):
    """Simple nearest neighbor algorithm for TSP"""
    unvisited = customers.copy()
    route = []
    current_lat, current_lon = start_lat, start_lon
    
    while unvisited:
        nearest_customer = None
        nearest_distance = float('inf')
        
        for customer in unvisited:
            distance = calculate_distance(
                current_lat, current_lon,
                float(customer.latitude), float(customer.longitude)
            )
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_customer = customer
        
        route.append(nearest_customer)
        unvisited.remove(nearest_customer)
        current_lat, current_lon = float(nearest_customer.latitude), float(nearest_customer.longitude)
    
    return route

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

