# # # # backend/models/dealer.py
# # # """
# # # Dealer model
# # # """
# # # from sqlalchemy import Column, String, Float, Integer
# # # from sqlalchemy.orm import relationship
# # # from geoalchemy2 import Geometry
# # # from .base import BaseModel


# # # class Dealer(BaseModel):
# # #     __tablename__ = "dealers"
    
# # #     user_code = Column(String(50), unique=True, index=True)
# # #     user_name = Column(String(255), nullable=False)
# # #     division_code = Column(String(50))
# # #     territory_code = Column(String(50))
# # #     phone = Column(String(20))
# # #     email = Column(String(100))
    
# # #     # Performance metrics
# # #     total_sales = Column(Float, default=0.0)
# # #     total_customers = Column(Integer, default=0)
# # #     avg_daily_distance = Column(Float, default=0.0)
# # #     efficiency_score = Column(Float, default=0.0)
    
# # #     # Relationships
# # #     gps_data = relationship("GPSData", back_populates="dealer")
# # #     sales = relationship("Sale", back_populates="dealer")
# # #     routes = relationship("Route", back_populates="dealer")


# # from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
# # from sqlalchemy.orm import relationship
# # from geoalchemy2 import Geography
# # from .base import Base

# # class Dealer(Base):
# #     __tablename__ = "dealers"
    
# #     id = Column(Integer, primary_key=True, index=True)
# #     user_code = Column(String, unique=True, index=True)
# #     user_name = Column(String)
# #     division_code = Column(String)
# #     territory_code = Column(String)
# #     phone = Column(String)
# #     email = Column(String)
# #     is_active = Column(Boolean, default=True)
# #     performance_score = Column(Float)
# #     created_at = Column(DateTime)
    
# #     # Relationships
# #     gps_tracks = relationship("GPSTrack", back_populates="dealer")
# #     orders = relationship("Order", back_populates="dealer")
# #     visits = relationship("DealerVisit", back_populates="dealer")


# """
# Dealer model for dealer information, territory, performance metrics, and GPS tracking.
# """
# import enum
# from decimal import Decimal
# from typing import Optional, List
# from datetime import datetime, timedelta
# from sqlalchemy import Column, Integer, String, Numeric, DateTime, Boolean, Text, Enum, ForeignKey
# from sqlalchemy.orm import relationship
# from sqlalchemy.ext.hybrid import hybrid_property
# from geoalchemy2 import Geography, Geometry
# from .base import BaseModel


# class DealerStatus(enum.Enum):
#     """Dealer status enumeration."""
#     ACTIVE = "active"
#     INACTIVE = "inactive"
#     SUSPENDED = "suspended"
#     TERMINATED = "terminated"
#     ON_LEAVE = "on_leave"


# class DealerType(enum.Enum):
#     """Dealer type enumeration."""
#     SALES_REP = "sales_rep"
#     DISTRIBUTOR = "distributor"
#     FIELD_OFFICER = "field_officer"
#     SUPERVISOR = "supervisor"
#     MANAGER = "manager"


# class VehicleType(enum.Enum):
#     """Vehicle type enumeration."""
#     MOTORCYCLE = "motorcycle"
#     CAR = "car"
#     VAN = "van"
#     TRUCK = "truck"
#     BICYCLE = "bicycle"
#     WALKING = "walking"


# class Dealer(BaseModel):
#     """
#     Dealer model based on SFA_GPSData and SFA_Orders datasets.
#     Stores dealer information, territory, and performance metrics.
#     """
    
#     __tablename__ = "dealer"
    
#     # Primary dealer identifier (from SFA_GPSData: "UserCode")
#     user_code = Column(String(20), unique=True, nullable=False, index=True)
    
#     # Basic Information (from SFA_GPSData: "UserName")
#     user_name = Column(String(100), nullable=False)
#     full_name = Column(String(150), nullable=True)
#     employee_id = Column(String(50), unique=True, nullable=True)
    
#     # Contact Information
#     phone_primary = Column(String(20), nullable=True)
#     phone_secondary = Column(String(20), nullable=True)
#     mobile = Column(String(20), nullable=True)
#     email = Column(String(100), nullable=True)
#     emergency_contact = Column(String(100), nullable=True)
#     emergency_phone = Column(String(20), nullable=True)
    
#     # Address Information
#     address_line_1 = Column(String(200), nullable=True)
#     address_line_2 = Column(String(200), nullable=True)
#     city = Column(String(100), nullable=True)
#     postal_code = Column(String(20), nullable=True)
#     state_province = Column(String(100), nullable=True)
#     country = Column(String(100), default="Sri Lanka", nullable=False)
    
#     # Territory Assignment (from SFA_GPSData: "DivisionCode")
#     division_code = Column(String(20), nullable=False, index=True)
#     territory_code = Column(String(20), nullable=True, index=True)
#     region_code = Column(String(20), nullable=True, index=True)
#     beat_code = Column(String(20), nullable=True)
    
#     # Dealer Classification
#     dealer_type = Column(Enum(DealerType), default=DealerType.SALES_REP, nullable=False)
#     dealer_status = Column(Enum(DealerStatus), default=DealerStatus.ACTIVE, nullable=False)
    
#     # Employment Details
#     join_date = Column(DateTime(timezone=True), nullable=True)
#     termination_date = Column(DateTime(timezone=True), nullable=True)
#     last_working_date = Column(DateTime(timezone=True), nullable=True)
#     supervisor_user_code = Column(String(20), ForeignKey('dealer.user_code'), nullable=True)
#     reporting_manager = Column(String(100), nullable=True)
    
#     # Performance Metrics
#     sales_target_monthly = Column(Numeric(12, 2), default=0, nullable=False)
#     sales_target_yearly = Column(Numeric(15, 2), default=0, nullable=False)
#     sales_achieved_mtd = Column(Numeric(12, 2), default=0, nullable=False)
#     sales_achieved_ytd = Column(Numeric(15, 2), default=0, nullable=False)
    
#     # Visit Metrics
#     planned_visits_monthly = Column(Integer, default=0, nullable=False)
#     actual_visits_mtd = Column(Integer, default=0, nullable=False)
#     customers_visited_mtd = Column(Integer, default=0, nullable=False)
#     new_customers_acquired_mtd = Column(Integer, default=0, nullable=False)
    
#     # Order Metrics
#     orders_count_mtd = Column(Integer, default=0, nullable=False)
#     orders_count_ytd = Column(Integer, default=0, nullable=False)
#     average_order_value = Column(Numeric(10, 2), default=0, nullable=False)
#     last_order_date = Column(DateTime(timezone=True), nullable=True)
    
#     # GPS and Movement Tracking
#     current_latitude = Column(Numeric(10, 8), nullable=True)
#     current_longitude = Column(Numeric(11, 8), nullable=True)
#     current_location = Column(Geography('POINT', srid=4326), nullable=True)
#     last_gps_update = Column(DateTime(timezone=True), nullable=True)
#     gps_accuracy = Column(Numeric(8, 2), nullable=True)
    
#     # Current Tour Information (from SFA_GPSData: "TourCode")
#     current_tour_code = Column(String(50), nullable=True, index=True)
#     tour_start_time = Column(DateTime(timezone=True), nullable=True)
#     tour_end_time = Column(DateTime(timezone=True), nullable=True)
#     tour_status = Column(String(20), default="not_started", nullable=False)  # not_started, active, completed, cancelled
    
#     # Vehicle Information
#     vehicle_type = Column(Enum(VehicleType), default=VehicleType.MOTORCYCLE, nullable=False)
#     vehicle_number = Column(String(20), nullable=True)
#     fuel_allowance_monthly = Column(Numeric(8, 2), default=0, nullable=False)
    
#     # Working Hours and Schedule
#     work_start_time = Column(String(8), default="08:00", nullable=False)  # HH:MM format
#     work_end_time = Column(String(8), default="17:00", nullable=False)    # HH:MM format
#     working_days = Column(String(20), default="MON-SAT", nullable=False)  # Days of week
    
#     # Performance Ratings
#     performance_rating = Column(Numeric(3, 2), nullable=True)  # 1-5 scale
#     customer_feedback_score = Column(Numeric(3, 2), nullable=True)  # 1-5 scale
#     punctuality_score = Column(Numeric(3, 2), nullable=True)  # 1-5 scale
    
#     # Financial Information
#     basic_salary = Column(Numeric(10, 2), default=0, nullable=False)
#     commission_rate = Column(Numeric(5, 2), default=0, nullable=False)  # Percentage
#     incentive_earned_mtd = Column(Numeric(8, 2), default=0, nullable=False)
#     travel_allowance = Column(Numeric(8, 2), default=0, nullable=False)
    
#     # Additional Fields
#     bank_account_no = Column(String(50), nullable=True)
#     bank_name = Column(String(100), nullable=True)
#     nic_number = Column(String(20), nullable=True)
#     driving_license_no = Column(String(50), nullable=True)
#     notes = Column(Text, nullable=True)
    
#     # Device and Technology
#     device_imei = Column(String(20), nullable=True)
#     app_version = Column(String(20), nullable=True)
#     last_app_sync = Column(DateTime(timezone=True), nullable=True)
    
#     # Relationships
#     subordinates = relationship("Dealer", backref="supervisor", remote_side=[user_code])
#     gps_tracks = relationship("GPSData", back_populates="dealer", lazy="dynamic")
#     sales_orders = relationship("Sales", back_populates="dealer", lazy="dynamic")
#     dealer_visits = relationship("CustomerVisit", back_populates="dealer")
#     route_plans = relationship("RoutePlan", back_populates="dealer")
    
#     @hybrid_property
#     def is_active_dealer(self) -> bool:
#         """Check if dealer is active and available for work."""
#         return (self.is_active and 
#                 self.dealer_status == DealerStatus.ACTIVE and
#                 (not self.termination_date or self.termination_date > datetime.utcnow()))
    
#     @hybrid_property
#     def sales_achievement_percentage(self) -> Optional[float]:
#         """Calculate sales achievement percentage for current month."""
#         if self.sales_target_monthly > 0:
#             return float((self.sales_achieved_mtd / self.sales_target_monthly) * 100)
#         return None
    
#     @hybrid_property
#     def visit_completion_rate(self) -> Optional[float]:
#         """Calculate visit completion rate for current month."""
#         if self.planned_visits_monthly > 0:
#             return float((self.actual_visits_mtd / self.planned_visits_monthly) * 100)
#         return None
    
#     @hybrid_property
#     def days_since_last_gps_update(self) -> Optional[int]:
#         """Calculate days since last GPS update."""
#         if self.last_gps_update:
#             delta = datetime.utcnow() - self.last_gps_update
#             return delta.days
#         return None
    
#     def update_current_location(self, latitude: float, longitude: float, 
#                               accuracy: float = None, tour_code: str = None) -> None:
#         """Update dealer's current location from GPS data."""
#         from geoalchemy2.elements import WKTElement
        
#         self.current_latitude = Decimal(str(latitude))
#         self.current_longitude = Decimal(str(longitude))
#         self.current_location = WKTElement(f'POINT({longitude} {latitude})', srid=4326)
#         self.last_gps_update = datetime.utcnow()
#         self.gps_accuracy = Decimal(str(accuracy)) if accuracy else None
        
#         if tour_code:
#             self.current_tour_code = tour_code
#             if self.tour_status == "not_started":
#                 self.tour_status = "active"
#                 self.tour_start_time = datetime.utcnow()
    
#     def start_tour(self, tour_code: str) -> None:
#         """Start a new tour."""
#         self.current_tour_code = tour_code
#         self.tour_status = "active"
#         self.tour_start_time = datetime.utcnow()
#         self.tour_end_time = None
    
#     def end_tour(self) -> None:
#         """End current tour."""
#         self.tour_status = "completed"
#         self.tour_end_time = datetime.utcnow()
    
#     def calculate_distance_traveled_today(self) -> Optional[float]:
#         """Calculate total distance traveled today in kilometers."""
#         from sqlalchemy import func, and_
#         from datetime import date
        
#         # This would require the GPS tracking data
#         # Implementation depends on the session context
#         return None
    
#     def update_sales_metrics(self, order_value: Decimal) -> None:
#         """Update sales metrics when new order is created."""
#         self.sales_achieved_mtd += order_value
#         self.sales_achieved_ytd += order_value
#         self.orders_count_mtd += 1
#         self.orders_count_ytd += 1
#         self.last_order_date = datetime.utcnow()
        
#         # Recalculate average order value
#         if self.orders_count_ytd > 0:
#             self.average_order_value = self.sales_achieved_ytd / self.orders_count_ytd
    
#     def reset_monthly_metrics(self) -> None:
#         """Reset monthly metrics at the beginning of each month."""
#         self.sales_achieved_mtd = Decimal('0')
#         self.actual_visits_mtd = 0
#         self.customers_visited_mtd = 0
#         self.new_customers_acquired_mtd = 0
#         self.orders_count_mtd = 0
#         self.incentive_earned_mtd = Decimal('0')
    
#     def calculate_incentive(self) -> Decimal:
#         """Calculate incentive based on sales achievement."""
#         if self.sales_achievement_percentage and self.sales_achievement_percentage >= 100:
#             excess_percentage = self.sales_achievement_percentage - 100
#             base_incentive = self.sales_achieved_mtd * (self.commission_rate / 100)
#             bonus_incentive = (self.sales_achieved_mtd * excess_percentage / 100) * 0.02  # 2% bonus
#             return base_incentive + bonus_incentive
#         elif self.sales_achievement_percentage and self.sales_achievement_percentage >= 80:
#             return self.sales_achieved_mtd * (self.commission_rate / 100)
#         return Decimal('0')
    
#     def get_performance_grade(self) -> str:
#         """Get performance grade based on various metrics."""
#         sales_score = min(100, self.sales_achievement_percentage or 0) / 100
#         visit_score = min(100, self.visit_completion_rate or 0) / 100
#         rating_score = (self.performance_rating or 3) / 5
        
#         overall_score = (sales_score * 0.5) + (visit_score * 0.3) + (rating_score * 0.2)
        
#         if overall_score >= 0.9:
#             return "A+"
#         elif overall_score >= 0.8:
#             return "A"
#         elif overall_score >= 0.7:
#             return "B+"
#         elif overall_score >= 0.6:
#             return "B"
#         elif overall_score >= 0.5:
#             return "C"
#         else:
#             return "D"
    
#     def is_within_working_hours(self) -> bool:
#         """Check if current time is within working hours."""
#         from datetime import datetime, time
        
#         current_time = datetime.now().time()
#         start_time = datetime.strptime(self.work_start_time, "%H:%M").time()
#         end_time = datetime.strptime(self.work_end_time, "%H:%M").time()
        
#         return start_time <= current_time <= end_time
    
#     def get_territory_customers(self):
#         """Get customers assigned to this dealer's territory."""
#         from .customer import Customer
#         return Customer.query.filter(
#             Customer.territory_code == self.territory_code,
#             Customer.is_active == True
#         )


# class DealerTarget(BaseModel):
#     """Dealer targets and goals."""
    
#     __tablename__ = "dealer_target"
    
#     dealer_user_code = Column(String(20), ForeignKey('dealer.user_code'), nullable=False)
#     target_year = Column(Integer, nullable=False)
#     target_month = Column(Integer, nullable=False)
    
#     # Sales Targets
#     sales_target = Column(Numeric(12, 2), nullable=False)
#     volume_target = Column(Numeric(10, 2), nullable=True)
    
#     # Visit Targets
#     visit_target = Column(Integer, nullable=False)
#     new_customer_target = Column(Integer, default=0, nullable=False)
    
#     # Product-wise targets (JSON or separate table)
#     product_targets = Column(Text, nullable=True)  # JSON string
    
#     # Achievement tracking
#     sales_achieved = Column(Numeric(12, 2), default=0, nullable=False)
#     visits_achieved = Column(Integer, default=0, nullable=False)
#     new_customers_achieved = Column(Integer, default=0, nullable=False)
    
#     # Relationships
#     dealer = relationship("Dealer")


# class DealerPerformanceHistory(BaseModel):
#     """Historical performance tracking for dealers."""
    
#     __tablename__ = "dealer_performance_history"
    
#     dealer_user_code = Column(String(20), ForeignKey('dealer.user_code'), nullable=False)
#     evaluation_date = Column(DateTime(timezone=True), nullable=False)
#     evaluation_period = Column(String(20), nullable=False)  # monthly, quarterly, yearly
    
#     # Performance Metrics
#     sales_achievement_percentage = Column(Numeric(5, 2), nullable=True)
#     visit_completion_rate = Column(Numeric(5, 2), nullable=True)
#     customer_satisfaction_score = Column(Numeric(3, 2), nullable=True)
#     punctuality_score = Column(Numeric(3, 2), nullable=True)
#     overall_rating = Column(Numeric(3, 2), nullable=True)
    
#     # Financial Metrics
#     revenue_generated = Column(Numeric(12, 2), nullable=True)
#     commission_earned = Column(Numeric(8, 2), nullable=True)
#     incentive_earned = Column(Numeric(8, 2), nullable=True)
    
#     # Operational Metrics
#     total_distance_traveled = Column(Numeric(8, 2), nullable=True)
#     working_days = Column(Integer, nullable=True)
#     overtime_hours = Column(Numeric(5, 2), nullable=True)
    
#     # Comments and feedback
#     manager_comments = Column(Text, nullable=True)
#     improvement_areas = Column(Text, nullable=True)
#     achievements = Column(Text, nullable=True)
    
#     # Relationships
#     dealer = relationship("Dealer")

"""
Dealer model for managing dealer information, territory, performance metrics, and GPS tracking.
Based on SFA_GPSData and SFA_Orders datasets structure.
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, ForeignKey, Enum, Index
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func
import enum
import json

from .base import BaseModel


class DealerType(enum.Enum):
    """Dealer type enumeration."""
    SALES_REP = "sales_rep"
    FIELD_AGENT = "field_agent"
    TERRITORY_MANAGER = "territory_manager"
    DISTRIBUTOR_REP = "distributor_rep"
    EXTERNAL_AGENT = "external_agent"


class DealerStatus(enum.Enum):
    """Dealer status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    ON_LEAVE = "on_leave"
    TERMINATED = "terminated"
    PROBATION = "probation"


class PerformanceGrade(enum.Enum):
    """Performance grade enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"


class Dealer(BaseModel):
    """
    Dealer model representing sales representatives and field agents.
    Based on SFA_GPSData and SFA_Orders datasets.
    """
    __tablename__ = 'dealers'
    
    # Basic Information (from SFA_GPSData)
    user_code = Column(String(50), unique=True, nullable=False, index=True)  # UserCode from dataset
    user_name = Column(String(255), nullable=False, index=True)  # UserName from dataset
    division_code = Column(String(20), nullable=False, index=True)  # DivisionCode from dataset
    
    # Extended Basic Information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    dealer_type = Column(Enum(DealerType), default=DealerType.SALES_REP, nullable=False)
    status = Column(Enum(DealerStatus), default=DealerStatus.ACTIVE, nullable=False)
    
    # Employment Information
    employee_id = Column(String(50), nullable=True, index=True)
    joining_date = Column(DateTime(timezone=True), nullable=True)
    termination_date = Column(DateTime(timezone=True), nullable=True)
    contract_type = Column(String(50), default='permanent', nullable=False)  # permanent, contract, temporary
    
    # Contact Information
    phone = Column(String(20), nullable=True, index=True)
    mobile = Column(String(20), nullable=False, index=True)
    email = Column(String(255), nullable=True, index=True)
    emergency_contact = Column(String(20), nullable=True)
    emergency_contact_name = Column(String(255), nullable=True)
    
    # Address Information
    address_line_1 = Column(String(255), nullable=True)
    address_line_2 = Column(String(255), nullable=True)
    city = Column(String(100), nullable=True, index=True)
    state_province = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)
    country = Column(String(100), default='Sri Lanka', nullable=False)
    
    # Territory Assignment
    territory_code = Column(String(20), nullable=True, index=True)
    region_code = Column(String(20), nullable=True, index=True)
    area_code = Column(String(20), nullable=True, index=True)
    beat_assignment = Column(Text, nullable=True)  # JSON array of assigned beats/routes
    territory_assigned_date = Column(DateTime(timezone=True), nullable=True)
    
    # Management Hierarchy
    manager_user_id = Column(Integer, ForeignKey('users.id'), nullable=True, index=True)
    supervisor_dealer_id = Column(Integer, ForeignKey('dealers.id'), nullable=True, index=True)
    created_by_user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    
    # Performance Metrics
    monthly_target = Column(Float, default=0.0, nullable=False)
    quarterly_target = Column(Float, default=0.0, nullable=False)
    annual_target = Column(Float, default=0.0, nullable=False)
    current_month_achievement = Column(Float, default=0.0, nullable=False)
    current_quarter_achievement = Column(Float, default=0.0, nullable=False)
    current_year_achievement = Column(Float, default=0.0, nullable=False)
    
    # Performance Grades
    overall_performance_grade = Column(Enum(PerformanceGrade), nullable=True)
    sales_performance_grade = Column(Enum(PerformanceGrade), nullable=True)
    customer_service_grade = Column(Enum(PerformanceGrade), nullable=True)
    last_performance_review = Column(DateTime(timezone=True), nullable=True)
    
    # Activity Metrics
    total_customers_assigned = Column(Integer, default=0, nullable=False)
    active_customers = Column(Integer, default=0, nullable=False)
    total_orders = Column(Integer, default=0, nullable=False)
    total_sales_value = Column(Float, default=0.0, nullable=False)
    average_order_value = Column(Float, default=0.0, nullable=False)
    last_order_date = Column(DateTime(timezone=True), nullable=True)
    
    # Visit Metrics
    total_visits = Column(Integer, default=0, nullable=False)
    visits_this_month = Column(Integer, default=0, nullable=False)
    average_visits_per_day = Column(Float, default=0.0, nullable=False)
    last_visit_date = Column(DateTime(timezone=True), nullable=True)
    
    # GPS and Mobility
    gps_tracking_enabled = Column(Boolean, default=True, nullable=False)
    last_gps_update = Column(DateTime(timezone=True), nullable=True)
    current_latitude = Column(Float, nullable=True, index=True)
    current_longitude = Column(Float, nullable=True, index=True)
    location_accuracy = Column(Float, nullable=True)
    total_distance_traveled = Column(Float, default=0.0, nullable=False)  # in kilometers
    
    # Work Schedule
    work_start_time = Column(String(10), default='09:00', nullable=False)
    work_end_time = Column(String(10), default='17:00', nullable=False)
    working_days = Column(String(20), default='1,2,3,4,5', nullable=False)  # 1=Monday, 7=Sunday
    timezone = Column(String(50), default='Asia/Colombo', nullable=False)
    
    # Device and Technology
    device_id = Column(String(100), nullable=True, index=True)
    device_type = Column(String(50), nullable=True)  # android, ios, tablet
    app_version = Column(String(20), nullable=True)
    last_app_update = Column(DateTime(timezone=True), nullable=True)
    device_registered_date = Column(DateTime(timezone=True), nullable=True)
    
    # Financial Information
    base_salary = Column(Float, nullable=True)
    commission_rate = Column(Float, default=0.0, nullable=False)
    incentive_eligibility = Column(Boolean, default=True, nullable=False)
    bank_account_no = Column(String(50), nullable=True)
    bank_name = Column(String(100), nullable=True)
    
    # Training and Certification
    training_completed = Column(Text, nullable=True)  # JSON array of completed trainings
    certifications = Column(Text, nullable=True)  # JSON array of certifications
    last_training_date = Column(DateTime(timezone=True), nullable=True)
    next_training_due = Column(DateTime(timezone=True), nullable=True)
    
    # Preferences and Settings
    language_preference = Column(String(10), default='en', nullable=False)
    notification_preferences = Column(Text, nullable=True)  # JSON object
    
    # Relationships
    manager = relationship("User", foreign_keys=[manager_user_id], back_populates="managed_dealers")
    creator = relationship("User", foreign_keys=[created_by_user_id], back_populates="created_dealers")
    supervisor = relationship("Dealer", remote_side=[id], backref="subordinates")
    
    # GPS tracking data
    gps_tracks = relationship("GPSData", back_populates="dealer", lazy="dynamic")
    
    # Sales orders
    orders = relationship("Sales", back_populates="dealer", lazy="dynamic")
    
    # Customer assignments
    assigned_customers = relationship("Customer", foreign_keys="Customer.assigned_sales_rep_id", backref="sales_rep")
    
    # Dealer visits
    visits = relationship("DealerVisit", back_populates="dealer", lazy="dynamic")
    
    # Performance records
    performance_records = relationship("DealerPerformance", back_populates="dealer", lazy="dynamic")
    
    @hybrid_property
    def full_name(self):
        """Get full name."""
        return f"{self.first_name} {self.last_name}"
    
    @hybrid_property
    def is_active_dealer(self):
        """Check if dealer is active."""
        return self.status == DealerStatus.ACTIVE and self.is_active
    
    @hybrid_property
    def has_current_location(self):
        """Check if dealer has current GPS location."""
        return self.current_latitude is not None and self.current_longitude is not None
    
    @hybrid_property
    def monthly_target_achievement_percentage(self):
        """Calculate monthly target achievement percentage."""
        if self.monthly_target > 0:
            return (self.current_month_achievement / self.monthly_target) * 100
        return 0
    
    @hybrid_property
    def is_gps_tracking_active(self):
        """Check if GPS tracking is recent (within last hour)."""
        if self.last_gps_update:
            return (datetime.utcnow() - self.last_gps_update) <= timedelta(hours=1)
        return False
    
    @hybrid_property
    def days_since_last_order(self):
        """Calculate days since last order."""
        if self.last_order_date:
            return (datetime.utcnow() - self.last_order_date).days
        return None
    
    def update_current_location(self, latitude: float, longitude: float, accuracy: float = None):
        """Update current GPS location."""
        self.current_latitude = latitude
        self.current_longitude = longitude
        self.location_accuracy = accuracy
        self.last_gps_update = datetime.utcnow()
    
    def calculate_distance_from(self, lat: float, lon: float) -> float:
        """Calculate distance from given coordinates."""
        if not self.has_current_location:
            return None
        
        import math
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [self.current_latitude, self.current_longitude, lat, lon])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        return c * r
    
    def update_performance_metrics(self):
        """Update performance metrics based on orders."""
        current_month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        current_year_start = datetime.utcnow().replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Calculate current month achievement
        month_orders = self.orders.filter(Sales.created_at >= current_month_start)
        self.current_month_achievement = sum([order.final_value for order in month_orders if order.final_value])
        
        # Calculate current year achievement
        year_orders = self.orders.filter(Sales.created_at >= current_year_start)
        self.current_year_achievement = sum([order.final_value for order in year_orders if order.final_value])
        
        # Update total metrics
        self.total_orders = self.orders.count()
        self.total_sales_value = sum([order.final_value for order in self.orders if order.final_value])
        self.average_order_value = self.total_sales_value / self.total_orders if self.total_orders > 0 else 0
        
        if self.orders.count() > 0:
            self.last_order_date = self.orders.order_by('created_at desc').first().created_at
    
    def assign_beat(self, beat_info: Dict[str, Any]):
        """Assign a beat/route to dealer."""
        beats = json.loads(self.beat_assignment) if self.beat_assignment else []
        beats.append({
            'beat_id': beat_info.get('beat_id'),
            'beat_name': beat_info.get('beat_name'),
            'assigned_date': datetime.utcnow().isoformat(),
            'customers': beat_info.get('customers', []),
            'is_active': True
        })
        self.beat_assignment = json.dumps(beats)
    
    def get_assigned_beats(self) -> List[Dict]:
        """Get all assigned beats."""
        return json.loads(self.beat_assignment) if self.beat_assignment else []
    
    def update_targets(self, monthly: float = None, quarterly: float = None, annual: float = None):
        """Update sales targets."""
        if monthly is not None:
            self.monthly_target = monthly
        if quarterly is not None:
            self.quarterly_target = quarterly
        if annual is not None:
            self.annual_target = annual
    
    def add_training(self, training_info: Dict[str, Any]):
        """Add completed training."""
        trainings = json.loads(self.training_completed) if self.training_completed else []
        trainings.append({
            'training_name': training_info.get('name'),
            'completion_date': datetime.utcnow().isoformat(),
            'certificate_id': training_info.get('certificate_id'),
            'score': training_info.get('score'),
            'validity_period': training_info.get('validity_period')
        })
        self.training_completed = json.dumps(trainings)
        self.last_training_date = datetime.utcnow()
    
    def get_trainings(self) -> List[Dict]:
        """Get all completed trainings."""
        return json.loads(self.training_completed) if self.training_completed else []
    
    def set_notification_preference(self, notification_type: str, enabled: bool):
        """Set notification preference."""
        prefs = json.loads(self.notification_preferences) if self.notification_preferences else {}
        prefs[notification_type] = enabled
        self.notification_preferences = json.dumps(prefs)
    
    def get_notification_preference(self, notification_type: str) -> bool:
        """Get notification preference."""
        prefs = json.loads(self.notification_preferences) if self.notification_preferences else {}
        return prefs.get(notification_type, True)  # Default to True
    
    def suspend_dealer(self, reason: str = None, end_date: datetime = None):
        """Suspend dealer."""
        self.status = DealerStatus.SUSPENDED
        suspension_info = {
            'reason': reason,
            'suspended_at': datetime.utcnow().isoformat(),
            'end_date': end_date.isoformat() if end_date else None
        }
        self.update_metadata('suspension_info', suspension_info)
    
    def activate_dealer(self):
        """Activate suspended dealer."""
        self.status = DealerStatus.ACTIVE
        activation_info = {
            'activated_at': datetime.utcnow().isoformat(),
            'activated_by': 'system'  # Can be updated to track who activated
        }
        self.update_metadata('activation_info', activation_info)
    
    def terminate_dealer(self, reason: str = None, termination_date: datetime = None):
        """Terminate dealer employment."""
        self.status = DealerStatus.TERMINATED
        self.termination_date = termination_date or datetime.utcnow()
        termination_info = {
            'reason': reason,
            'terminated_at': datetime.utcnow().isoformat(),
            'final_settlement_date': None,
            'handover_completed': False
        }
        self.update_metadata('termination_info', termination_info)
    
    def get_current_performance_grade(self) -> str:
        """Get current overall performance grade."""
        if self.overall_performance_grade:
            return self.overall_performance_grade.value
        return 'not_evaluated'
    
    def is_target_achieved(self, target_type: str = 'monthly') -> bool:
        """Check if target is achieved."""
        if target_type == 'monthly':
            return self.monthly_target_achievement_percentage >= 100
        elif target_type == 'quarterly':
            return (self.current_quarter_achievement / self.quarterly_target * 100) >= 100 if self.quarterly_target > 0 else False
        elif target_type == 'annual':
            return (self.current_year_achievement / self.annual_target * 100) >= 100 if self.annual_target > 0 else False
        return False
    
    def get_working_days_list(self) -> List[int]:
        """Get list of working days."""
        return [int(day) for day in self.working_days.split(',') if day.strip()]
    
    def set_working_days(self, days: List[int]):
        """Set working days."""
        self.working_days = ','.join(map(str, days))
    
    def is_working_today(self) -> bool:
        """Check if today is a working day."""
        today = datetime.utcnow().weekday() + 1  # Monday is 1
        return today in self.get_working_days_list()
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        return {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'app_version': self.app_version,
            'last_update': self.last_app_update.isoformat() if self.last_app_update else None,
            'registered_date': self.device_registered_date.isoformat() if self.device_registered_date else None
        }
    
    def update_device_info(self, device_id: str, device_type: str, app_version: str):
        """Update device information."""
        self.device_id = device_id
        self.device_type = device_type
        self.app_version = app_version
        self.last_app_update = datetime.utcnow()
        if not self.device_registered_date:
            self.device_registered_date = datetime.utcnow()
    
    def calculate_commission(self, sales_amount: float) -> float:
        """Calculate commission based on sales amount."""
        return sales_amount * (self.commission_rate / 100)
    
    def get_territory_info(self) -> Dict[str, Any]:
        """Get territory information."""
        return {
            'territory_code': self.territory_code,
            'region_code': self.region_code,
            'area_code': self.area_code,
            'assigned_date': self.territory_assigned_date.isoformat() if self.territory_assigned_date else None,
            'beats': self.get_assigned_beats()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dealer to dictionary."""
        return {
            'id': self.id,
            'user_code': self.user_code,
            'user_name': self.user_name,
            'full_name': self.full_name,
            'division_code': self.division_code,
            'dealer_type': self.dealer_type.value if self.dealer_type else None,
            'status': self.status.value if self.status else None,
            'is_active': self.is_active_dealer,
            'contact_info': {
                'phone': self.phone,
                'mobile': self.mobile,
                'email': self.email
            },
            'current_location': {
                'latitude': self.current_latitude,
                'longitude': self.current_longitude,
                'accuracy': self.location_accuracy,
                'last_update': self.last_gps_update.isoformat() if self.last_gps_update else None
            },
            'performance': {
                'monthly_target': self.monthly_target,
                'monthly_achievement': self.current_month_achievement,
                'monthly_achievement_percentage': self.monthly_target_achievement_percentage,
                'overall_grade': self.overall_performance_grade.value if self.overall_performance_grade else None
            },
            'territory': self.get_territory_info(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    def __repr__(self):
        return f"<Dealer(id={self.id}, user_code='{self.user_code}', name='{self.full_name}', status='{self.status.value if self.status else None}')>"


# Database indexes for performance optimization
Index('idx_dealer_user_code_division', Dealer.user_code, Dealer.division_code)
Index('idx_dealer_territory', Dealer.territory_code, Dealer.region_code, Dealer.area_code)
Index('idx_dealer_location', Dealer.current_latitude, Dealer.current_longitude)
Index('idx_dealer_status_type', Dealer.status, Dealer.dealer_type)
Index('idx_dealer_performance', Dealer.overall_performance_grade, Dealer.status)
Index('idx_dealer_gps_update', Dealer.last_gps_update, Dealer.gps_tracking_enabled)