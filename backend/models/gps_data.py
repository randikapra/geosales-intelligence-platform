# # # backend/models/gps_data.py
# # """
# # GPS tracking model
# # """
# # from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Integer
# # from sqlalchemy.orm import relationship
# # from geoalchemy2 import Geometry
# # from .base import BaseModel


# # class GPSData(BaseModel):
# #     __tablename__ = "gps_data"
    
# #     division_code = Column(String(50))
# #     user_code = Column(String(50), ForeignKey("dealers.user_code"))
# #     user_name = Column(String(255))
# #     latitude = Column(Float, nullable=False)
# #     longitude = Column(Float, nullable=False)
# #     tour_code = Column(String(100))
# #     received_date = Column(DateTime, nullable=False)
    
# #     # Geospatial data
# #     location = Column(Geometry('POINT'))
    
# #     # Derived fields
# #     speed = Column(Float)  # km/h
# #     distance_from_previous = Column(Float)  # meters
# #     stop_duration = Column(Integer)  # seconds
# #     is_moving = Column(Boolean, default=True)
    
# #     # Relationships
# #     dealer = relationship("Dealer", back_populates="gps_data")


# from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
# from sqlalchemy.orm import relationship
# from geoalchemy2 import Geography
# from .base import Base

# class GPSTrack(Base):
#     __tablename__ = "gps_tracks"
    
#     id = Column(Integer, primary_key=True, index=True)
#     dealer_id = Column(Integer, ForeignKey("dealers.id"))
#     division_code = Column(String)
#     user_code = Column(String)
#     latitude = Column(Float)
#     longitude = Column(Float)
#     location = Column(Geography(geometry_type='POINT', srid=4326))
#     tour_code = Column(String)
#     received_date = Column(DateTime)
#     speed = Column(Float)  # Calculated field
#     bearing = Column(Float)  # Direction
#     accuracy = Column(Float)  # GPS accuracy
    
#     # Relationships
#     dealer = relationship("Dealer", back_populates="gps_tracks")


"""
GPS Data model for tracking dealer movements and location history.
Based on SFA_GPSData.csv dataset structure.
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, ForeignKey, Index
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func
import json
import math

from .base import BaseModel


class GPSData(BaseModel):
    """
    GPS Data model for tracking dealer location and movement.
    Based on SFA_GPSData.csv dataset structure.
    """
    __tablename__ = 'gps_data'
    
    # Basic GPS Information (from SFA_GPSData)
    division_code = Column(String(20), nullable=False, index=True)  # DivisionCode from dataset
    user_code = Column(String(50), nullable=False, index=True)  # UserCode from dataset
    user_name = Column(String(255), nullable=False, index=True)  # UserName from dataset
    latitude = Column(Float, nullable=False, index=True)  # Latitude from dataset
    longitude = Column(Float, nullable=False, index=True)  # Longitude from dataset
    tour_code = Column(String(50), nullable=True, index=True)  # TourCode from dataset
    received_date = Column(DateTime(timezone=True), nullable=False, index=True)  # RecievedDate from dataset
    
    # Extended GPS Information
    dealer_id = Column(Integer, ForeignKey('dealers.id'), nullable=False, index=True)
    
    # Location Accuracy and Quality
    accuracy = Column(Float, nullable=True)  # GPS accuracy in meters
    altitude = Column(Float, nullable=True)  # Altitude in meters
    speed = Column(Float, nullable=True)  # Speed in km/h
    heading = Column(Float, nullable=True)  # Direction in degrees (0-360)
    
    # Device Information
    device_id = Column(String(100), nullable=True)
    battery_level = Column(Integer, nullable=True)  # Battery percentage
    signal_strength = Column(Integer, nullable=True)  # Signal strength
    
    # Movement Analytics
    is_moving = Column(Boolean, default=False, nullable=False)
    movement_type = Column(String(20), nullable=True)  # walking, driving, stationary
    distance_from_previous = Column(Float, nullable=True)  # Distance in meters
    time_since_previous = Column(Integer, nullable=True)  # Time in seconds
    
    # Location Context
    location_type = Column(String(50), nullable=True)  # customer_location, office, home, transit
    nearby_customer_id = Column(Integer, ForeignKey('customers.id'), nullable=True, index=True)
    address = Column(String(500), nullable=True)  # Reverse geocoded address
    place_name = Column(String(255), nullable=True)  # Point of interest name
    
    # Visit Information
    is_customer_visit = Column(Boolean, default=False, nullable=False)
    visit_duration = Column(Integer, nullable=True)  # Duration in minutes
    visit_start_time = Column(DateTime(timezone=True), nullable=True)
    visit_end_time = Column(DateTime(timezone=True), nullable=True)
    
    # Route Information
    route_sequence = Column(Integer, nullable=True)  # Sequence in planned route
    is_planned_location = Column(Boolean, default=False, nullable=False)
    deviation_from_route = Column(Float, nullable=True)  # Distance in meters
    
    # Data Quality
    is_valid_location = Column(Boolean, default=True, nullable=False)
    # Data Quality (continuing from where you left off)
    is_outlier = Column(Boolean, default=False, nullable=False)
    confidence_score = Column(Float, nullable=True)  # Location confidence 0-1
    
    # Metadata
    raw_data = Column(Text, nullable=True)  # Store original GPS data as JSON
    processed_at = Column(DateTime(timezone=True), default=func.now())
    processing_version = Column(String(20), nullable=True)
    
    # Relationships
    dealer = relationship("Dealer", back_populates="gps_data")
    nearby_customer = relationship("Customer", foreign_keys=[nearby_customer_id])
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_gps_user_date', 'user_code', 'received_date'),
        Index('idx_gps_location', 'latitude', 'longitude'),
        Index('idx_gps_dealer_date', 'dealer_id', 'received_date'),
        Index('idx_gps_tour_code', 'tour_code', 'received_date'),
        Index('idx_gps_spatial', 'latitude', 'longitude', 'received_date'),
    )
    
    @hybrid_property
    def coordinates(self) -> Tuple[float, float]:
        """Return coordinates as (latitude, longitude) tuple"""
        return (self.latitude, self.longitude)
    
    @hybrid_property
    def is_recent(self) -> bool:
        """Check if GPS data is from the last hour"""
        return self.received_date >= datetime.utcnow() - timedelta(hours=1)
    
    @hybrid_property
    def location_accuracy_level(self) -> str:
        """Categorize GPS accuracy level"""
        if not self.accuracy:
            return "unknown"
        elif self.accuracy <= 5:
            return "high"
        elif self.accuracy <= 15:
            return "medium"
        else:
            return "low"
    
    def calculate_distance_to(self, lat: float, lon: float) -> float:
        """
        Calculate distance to another point using Haversine formula
        Returns distance in meters
        """
        return self._haversine_distance(self.latitude, self.longitude, lat, lon)
    
    def calculate_speed_from_previous(self, previous_gps: 'GPSData') -> Optional[float]:
        """
        Calculate speed based on previous GPS point
        Returns speed in km/h
        """
        if not previous_gps:
            return None
            
        distance = self.calculate_distance_to(previous_gps.latitude, previous_gps.longitude)
        time_diff = (self.received_date - previous_gps.received_date).total_seconds()
        
        if time_diff <= 0:
            return None
            
        # Convert m/s to km/h
        speed_ms = distance / time_diff
        return speed_ms * 3.6
    
    def is_near_location(self, lat: float, lon: float, radius_meters: float = 100) -> bool:
        """Check if GPS point is within specified radius of a location"""
        distance = self.calculate_distance_to(lat, lon)
        return distance <= radius_meters
    
    def get_movement_pattern(self, window_minutes: int = 30) -> Dict[str, Any]:
        """
        Analyze movement pattern within a time window
        Returns pattern analysis
        """
        return {
            'is_stationary': not self.is_moving,
            'movement_type': self.movement_type,
            'speed': self.speed,
            'location_type': self.location_type,
            'accuracy_level': self.location_accuracy_level,
            'visit_info': {
                'is_customer_visit': self.is_customer_visit,
                'visit_duration': self.visit_duration,
                'nearby_customer_id': self.nearby_customer_id
            }
        }
    
    def to_geojson_feature(self) -> Dict[str, Any]:
        """Convert GPS point to GeoJSON feature format"""
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [self.longitude, self.latitude]
            },
            "properties": {
                "user_code": self.user_code,
                "user_name": self.user_name,
                "division_code": self.division_code,
                "tour_code": self.tour_code,
                "received_date": self.received_date.isoformat(),
                "speed": self.speed,
                "heading": self.heading,
                "accuracy": self.accuracy,
                "is_moving": self.is_moving,
                "movement_type": self.movement_type,
                "location_type": self.location_type,
                "is_customer_visit": self.is_customer_visit,
                "visit_duration": self.visit_duration
            }
        }
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        Returns distance in meters
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in meters
        r = 6371000
        return c * r
    
    @classmethod
    def get_recent_locations(cls, session, user_code: str, hours: int = 24) -> List['GPSData']:
        """Get recent GPS locations for a user"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return session.query(cls).filter(
            cls.user_code == user_code,
            cls.received_date >= cutoff_time,
            cls.is_valid_location == True
        ).order_by(cls.received_date.desc()).all()
    
    @classmethod
    def get_locations_in_area(cls, session, center_lat: float, center_lon: float, 
                            radius_km: float = 1.0) -> List['GPSData']:
        """Get GPS locations within a circular area"""
        # Simple bounding box filter (more efficient than exact circle)
        lat_delta = radius_km / 111.32  # Approximate km per degree latitude
        lon_delta = radius_km / (111.32 * math.cos(math.radians(center_lat)))
        
        return session.query(cls).filter(
            cls.latitude.between(center_lat - lat_delta, center_lat + lat_delta),
            cls.longitude.between(center_lon - lon_delta, center_lon + lon_delta),
            cls.is_valid_location == True
        ).all()
    
    @classmethod
    def get_route_trace(cls, session, user_code: str, tour_code: str) -> List['GPSData']:
        """Get complete route trace for a specific tour"""
        return session.query(cls).filter(
            cls.user_code == user_code,
            cls.tour_code == tour_code,
            cls.is_valid_location == True
        ).order_by(cls.received_date).all()
    
    @classmethod
    def get_customer_visits(cls, session, user_code: str, date_from: datetime, 
                          date_to: datetime) -> List['GPSData']:
        """Get customer visit locations within date range"""
        return session.query(cls).filter(
            cls.user_code == user_code,
            cls.is_customer_visit == True,
            cls.received_date.between(date_from, date_to)
        ).order_by(cls.received_date).all()
    
    def __repr__(self):
        return (f"<GPSData(user_code='{self.user_code}', "
                f"lat={self.latitude}, lon={self.longitude}, "
                f"date='{self.received_date}')>")