from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, text, desc, asc, extract, literal_column
from datetime import datetime, timedelta
from models.gps_data import GPSData
from models.dealer import Dealer
from models.customer import Customer
from .base import CRUDBase, BaseRepository
from schemas.gps_data import GPSDataCreate, GPSDataUpdate
import math
from decimal import Decimal


class GPSRepository(CRUDBase[GPSData, GPSDataCreate, GPSDataUpdate]):
    def __init__(self):
        super().__init__(GPSData)

    def get_dealer_locations(
        self,
        db: Session,
        user_code: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000
    ) -> List[GPSData]:
        """Get GPS tracking data for a specific dealer within date range"""
        return (
            db.query(self.model)
            .filter(and_(
                self.model.user_code == user_code,
                self.model.received_date >= start_date,
                self.model.received_date <= end_date
            ))
            .order_by(self.model.received_date)
            .limit(limit)
            .all()
        )

    def get_dealers_near_location(
        self,
        db: Session,
        latitude: float,
        longitude: float,
        radius_km: float = 5.0,
        timestamp: datetime = None
    ) -> List[Dict[str, Any]]:
        """Find dealers within a specific radius of a location"""
        # Using Haversine formula for distance calculation
        # Distance = 6371 * acos(cos(radians(lat1)) * cos(radians(lat2)) * cos(radians(lon2) - radians(lon1)) + sin(radians(lat1)) * sin(radians(lat2)))
        
        distance_formula = func.acos(
            func.cos(func.radians(latitude)) *
            func.cos(func.radians(self.model.latitude)) *
            func.cos(func.radians(self.model.longitude) - func.radians(longitude)) +
            func.sin(func.radians(latitude)) *
            func.sin(func.radians(self.model.latitude))
        ) * 6371  # Earth's radius in kilometers
        
        query = (
            db.query(
                self.model.user_code,
                self.model.user_name,
                self.model.latitude,
                self.model.longitude,
                self.model.received_date,
                distance_formula.label('distance_km')
            )
            .filter(distance_formula <= radius_km)
        )
        
        if timestamp:
            # Get locations within 1 hour of specified timestamp
            time_window = timedelta(hours=1)
            query = query.filter(and_(
                self.model.received_date >= timestamp - time_window,
                self.model.received_date <= timestamp + time_window
            ))
        
        results = query.order_by(distance_formula).all()
        
        return [
            {
                "user_code": result.user_code,
                "user_name": result.user_name,
                "latitude": float(result.latitude),
                "longitude": float(result.longitude),
                "distance_km": float(result.distance_km),
                "timestamp": result.received_date.isoformat()
            }
            for result in results
        ]

    def get_dealer_route_analysis(
        self,
        db: Session,
        user_code: str,
        date: datetime.date
    ) -> Dict[str, Any]:
        """Analyze dealer's route for a specific day"""
        start_date = datetime.combine(date, datetime.min.time())
        end_date = datetime.combine(date, datetime.max.time())
        
        # Get all GPS points for the day
        gps_points = (
            db.query(self.model)
            .filter(and_(
                self.model.user_code == user_code,
                self.model.received_date >= start_date,
                self.model.received_date <= end_date
            ))
            .order_by(self.model.received_date)
            .all()
        )
        
        if not gps_points:
            return {"error": "No GPS data found for the specified date"}
        
        # Calculate route metrics
        total_distance = 0
        speeds = []
        stops = []
        
        for i in range(1, len(gps_points)):
            prev_point = gps_points[i-1]
            curr_point = gps_points[i]
            
            # Calculate distance between consecutive points
            distance = self._calculate_distance(
                prev_point.latitude, prev_point.longitude,
                curr_point.latitude, curr_point.longitude
            )
            total_distance += distance
            
            # Calculate speed
            time_diff = (curr_point.received_date - prev_point.received_date).total_seconds()
            if time_diff > 0:
                speed_kmh = (distance / time_diff) * 3600  # Convert to km/h
                speeds.append(speed_kmh)
            
            # Detect stops (same location for more than 5 minutes)
            if distance < 0.05 and time_diff > 300:  # 50m radius, 5 minutes
                stops.append({
                    "latitude": float(curr_point.latitude),
                    "longitude": float(curr_point.longitude),
                    "start_time": prev_point.received_date.isoformat(),
                    "end_time": curr_point.received_date.isoformat(),
                    "duration_minutes": time_diff / 60
                })
        
        # Calculate summary statistics
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        max_speed = max(speeds) if speeds else 0
        
        # Working hours calculation
        first_point = gps_points[0]
        last_point = gps_points[-1]
        working_duration = (last_point.received_date - first_point.received_date).total_seconds() / 3600
        
        return {
            "user_code": user_code,
            "date": date.isoformat(),
            "summary": {
                "total_distance_km": round(total_distance, 2),
                "working_hours": round(working_duration, 2),
                "avg_speed_kmh": round(avg_speed, 2),
                "max_speed_kmh": round(max_speed, 2),
                "total_stops": len(stops),
                "total_gps_points": len(gps_points)
            },
            "route_points": [
                {
                    "latitude": float(point.latitude),
                    "longitude": float(point.longitude),
                    "timestamp": point.received_date.isoformat(),
                    "tour_code": point.tour_code
                }
                for point in gps_points[::10]  # Sample every 10th point for performance
            ],
            "stops": stops,
            "start_location": {
                "latitude": float(first_point.latitude),
                "longitude": float(first_point.longitude),
                "time": first_point.received_date.isoformat()
            },
            "end_location": {
                "latitude": float(last_point.latitude),
                "longitude": float(last_point.longitude),
                "time": last_point.received_date.isoformat()
            }
        }

    def get_territory_coverage(
        self,
        db: Session,
        division_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Analyze territory coverage by division"""
        # Get all GPS points for division
        gps_data = (
            db.query(self.model)
            .filter(and_(
                self.model.division_code == division_code,
                self.model.received_date >= start_date,
                self.model.received_date <= end_date
            ))
            .all()
        )
        
        if not gps_data:
            return {"error": "No GPS data found for the specified division"}
        
        # Calculate coverage area bounds
        latitudes = [float(point.latitude) for point in gps_data]
        longitudes = [float(point.longitude) for point in gps_data]
        
        bounds = {
            "north": max(latitudes),
            "south": min(latitudes),
            "east": max(longitudes),
            "west": min(longitudes)
        }
        
        # Calculate area coverage (approximate)
        lat_distance = self._calculate_distance(bounds["north"], bounds["west"], bounds["south"], bounds["west"])
        lon_distance = self._calculate_distance(bounds["north"], bounds["west"], bounds["north"], bounds["east"])
        coverage_area = lat_distance * lon_distance
        
        # Dealer activity analysis
        dealer_activity = {}
        for point in gps_data:
            if point.user_code not in dealer_activity:
                dealer_activity[point.user_code] = {
                    "user_name": point.user_name,
                    "points_count": 0,
                    "unique_days": set(),
                    "first_activity": point.received_date,
                    "last_activity": point.received_date
                }
            
            activity = dealer_activity[point.user_code]
            activity["points_count"] += 1
            activity["unique_days"].add(point.received_date.date())
            
            if point.received_date < activity["first_activity"]:
                activity["first_activity"] = point.received_date
            if point.received_date > activity["last_activity"]:
                activity["last_activity"] = point.received_date
        
        # Convert to summary format
        dealer_summary = []
        for user_code, activity in dealer_activity.items():
            dealer_summary.append({
                "user_code": user_code,
                "user_name": activity["user_name"],
                "gps_points": activity["points_count"],
                "active_days": len(activity["unique_days"]),
                "first_activity": activity["first_activity"].isoformat(),
                "last_activity": activity["last_activity"].isoformat()
            })
        
        return {
            "division_code": division_code,
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "coverage_summary": {
                "total_gps_points": len(gps_data),
                "estimated_area_km2": round(coverage_area, 2),
                "active_dealers": len(dealer_activity),
                "coverage_bounds": bounds
            },
            "dealer_activity": sorted(dealer_summary, key=lambda x: x["gps_points"], reverse=True)
        }

    def get_customer_visit_analysis(
        self,
        db: Session,
        user_code: str,
        start_date: datetime,
        end_date: datetime,
        visit_radius_m: float = 100
    ) -> List[Dict[str, Any]]:
        """Detect potential customer visits based on GPS proximity"""
        # Get dealer GPS data
        dealer_gps = (
            db.query(self.model)
            .filter(and_(
                self.model.user_code == user_code,
                self.model.received_date >= start_date,
                self.model.received_date <= end_date
            ))
            .order_by(self.model.received_date)
            .all()
        )
        
        # Get customer locations
        customers = db.query(Customer).all()
        customer_locations = {
            customer.code: {
                "latitude": customer.latitude,
                "longitude": customer.longitude,
                "city": customer.city
            }
            for customer in customers
            if customer.latitude and customer.longitude
        }
        
        visits = []
        for gps_point in dealer_gps:
            for customer_code, location in customer_locations.items():
                distance = self._calculate_distance(
                    float(gps_point.latitude), float(gps_point.longitude),
                    location["latitude"], location["longitude"]
                ) * 1000  # Convert to meters
                
                if distance <= visit_radius_m:
                    visits.append({
                        "customer_code": customer_code,
                        "customer_city": location["city"],
                        "visit_time": gps_point.received_date.isoformat(),
                        "distance_meters": round(distance, 2),
                        "dealer_location": {
                            "latitude": float(gps_point.latitude),
                            "longitude": float(gps_point.longitude)
                        },
                        "customer_location": {
                            "latitude": location["latitude"],
                            "longitude": location["longitude"]
                        }
                    })
        
        # Group consecutive visits to same customer
        grouped_visits = []
        if visits:
            current_visit = visits[0]
            visit_start = current_visit["visit_time"]
            
            for i in range(1, len(visits)):
                if (visits[i]["customer_code"] == current_visit["customer_code"] and
                    (datetime.fromisoformat(visits[i]["visit_time"]) - 
                     datetime.fromisoformat(current_visit["visit_time"])).total_seconds() < 3600):
                    # Continue same visit
                    current_visit = visits[i]
                else:
                    # End current visit, start new one
                    grouped_visits.append({
                        **current_visit,
                        "visit_duration_minutes": (
                            datetime.fromisoformat(current_visit["visit_time"]) - 
                            datetime.fromisoformat(visit_start)
                        ).total_seconds() / 60,
                        "visit_start": visit_start
                    })
                    current_visit = visits[i]
                    visit_start = current_visit["visit_time"]
            
            # Add last visit
            grouped_visits.append({
                **current_visit,
                "visit_duration_minutes": (
                    datetime.fromisoformat(current_visit["visit_time"]) - 
                    datetime.fromisoformat(visit_start)
                ).total_seconds() / 60,
                "visit_start": visit_start
            })
        
        return grouped_visits

    def get_movement_patterns(
        self,
        db: Session,
        user_code: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Analyze dealer movement patterns over time"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Get GPS data
        gps_data = (
            db.query(self.model)
            .filter(and_(
                self.model.user_code == user_code,
                self.model.received_date >= start_date,
                self.model.received_date <= end_date
            ))
            .order_by(self.model.received_date)
            .all()
        )
        
        if not gps_data:
            return {"error": "No GPS data found"}
        
        # Daily analysis
        daily_patterns = {}
        for point in gps_data:
            date_key = point.received_date.date().isoformat()
            if date_key not in daily_patterns:
                daily_patterns[date_key] = {
                    "points": [],
                    "total_distance": 0,
                    "start_time": point.received_date,
                    "end_time": point.received_date
                }
            
            pattern = daily_patterns[date_key]
            pattern["points"].append(point)
            
            if point.received_date < pattern["start_time"]:
                pattern["start_time"] = point.received_date
            if point.received_date > pattern["end_time"]:
                pattern["end_time"] = point.received_date
        
        # Calculate daily distances and working hours
        for date_key, pattern in daily_patterns.items():
            points = pattern["points"]
            total_distance = 0
            
            for i in range(1, len(points)):
                distance = self._calculate_distance(
                    float(points[i-1].latitude), float(points[i-1].longitude),
                    float(points[i].latitude), float(points[i].longitude)
                )
                total_distance += distance
            
            pattern["total_distance"] = total_distance
            pattern["working_hours"] = (pattern["end_time"] - pattern["start_time"]).total_seconds() / 3600
            pattern["points_count"] = len(points)
        
        # Calculate averages
        total_days = len(daily_patterns)
        avg_distance = sum(p["total_distance"] for p in daily_patterns.values()) / total_days
        avg_working_hours = sum(p["working_hours"] for p in daily_patterns.values()) / total_days
        avg_points = sum(p["points_count"] for p in daily_patterns.values()) / total_days
        
        return {
            "user_code": user_code,
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days_analyzed": total_days
            },
            "summary": {
                "avg_daily_distance_km": round(avg_distance, 2),
                "avg_working_hours": round(avg_working_hours, 2),
                "avg_gps_points_per_day": round(avg_points, 0),
                "total_distance_km": round(sum(p["total_distance"] for p in daily_patterns.values()), 2)
            },
            "daily_patterns": [
                {
                    "date": date_key,
                    "distance_km": round(pattern["total_distance"], 2),
                    "working_hours": round(pattern["working_hours"], 2),
                    "gps_points": pattern["points_count"],
                    "start_time": pattern["start_time"].strftime("%H:%M"),
                    "end_time": pattern["end_time"].strftime("%H:%M")
                }
                for date_key, pattern in sorted(daily_patterns.items())
            ]
        }

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c


# Create instance
gps_repository = GPSRepository()