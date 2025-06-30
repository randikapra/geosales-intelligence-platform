
# # # # backend/models/customer.py
# # # """
# # # Customer model
# # # """
# # # from sqlalchemy import Column, String, Float, Integer, ForeignKey
# # # from sqlalchemy.orm import relationship
# # # from geoalchemy2 import Geometry
# # # from .base import BaseModel


# # # class Customer(BaseModel):
# # #     __tablename__ = "customers"
    
# # #     customer_code = Column(String(50), unique=True, index=True)
# # #     name = Column(String(255), nullable=False)
# # #     city = Column(String(100))
# # #     contact = Column(String(100))
# # #     telex_no = Column(String(50))
# # #     document_sending_profile = Column(String(100))
# # #     ship_to_code = Column(String(50))
# # #     our_account_no = Column(String(50))
# # #     territory_code = Column(String(50))
# # #     global_dimension_1_code = Column(String(50))
# # #     global_dimension_2_code = Column(String(50))
    
# # #     # Geospatial data
# # #     latitude = Column(Float)
# # #     longitude = Column(Float)
# # #     location = Column(Geometry('POINT'))
    
# # #     # Relationships
# # #     sales = relationship("Sale", back_populates="customer")
# # #     orders = relationship("Order", back_populates="customer")


# # from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
# # from sqlalchemy.orm import relationship
# # from geoalchemy2 import Geography
# # from .base import Base

# # class Customer(Base):
# #     __tablename__ = "customers"
    
# #     id = Column(Integer, primary_key=True, index=True)
# #     customer_code = Column(String, unique=True, index=True)
# #     city = Column(String)
# #     contact = Column(String)
# #     telex_no = Column(String)
# #     territory_code = Column(String)
# #     ship_to_code = Column(String)
# #     location = Column(Geography(geometry_type='POINT', srid=4326))
# #     latitude = Column(Float)
# #     longitude = Column(Float)
# #     is_active = Column(Boolean, default=True)
# #     created_at = Column(DateTime)
# #     updated_at = Column(DateTime)
    
# #     # Relationships
# #     orders = relationship("Order", back_populates="customer")
# #     visits = relationship("DealerVisit", back_populates="customer")



# """
# Customer model for customer details, location, contact info, and business profile.
# """
# import enum
# from decimal import Decimal
# from typing import Optional, List
# from sqlalchemy import Column, Integer, String, Numeric, DateTime, Boolean, Text, Enum, ForeignKey
# from sqlalchemy.orm import relationship
# from sqlalchemy.ext.hybrid import hybrid_property
# from geoalchemy2 import Geography, Geometry
# from .base import BaseModel


# class CustomerStatus(enum.Enum):
#     """Customer status enumeration."""
#     ACTIVE = "active"
#     INACTIVE = "inactive"
#     SUSPENDED = "suspended"
#     PROSPECT = "prospect"
#     BLOCKED = "blocked"


# class CustomerType(enum.Enum):
#     """Customer type enumeration."""
#     DISTRIBUTOR = "distributor"
#     RETAILER = "retailer"
#     WHOLESALER = "wholesaler"
#     DIRECT = "direct"
#     ONLINE = "online"


# class CustomerCategory(enum.Enum):
#     """Customer category enumeration."""
#     A = "A"  # High value
#     B = "B"  # Medium value
#     C = "C"  # Low value
#     NEW = "NEW"  # New customer


# class Customer(BaseModel):
#     """
#     Customer model based on Customer.xlsx dataset.
#     Stores customer details, location, and business information.
#     """
    
#     __tablename__ = "customer"
    
#     # Primary customer identifier (from dataset: "No." column)
#     customer_code = Column(String(20), unique=True, nullable=False, index=True)
    
#     # Basic Information
#     name = Column(String(200), nullable=False)
#     display_name = Column(String(200), nullable=True)
    
#     # Location Information (from dataset: "City" column)
#     city = Column(String(100), nullable=False, index=True)
#     address_line_1 = Column(String(200), nullable=True)
#     address_line_2 = Column(String(200), nullable=True)
#     postal_code = Column(String(20), nullable=True)
#     state_province = Column(String(100), nullable=True)
#     country = Column(String(100), default="Sri Lanka", nullable=False)
    
#     # Geographic coordinates (from customer.csv dataset)
#     latitude = Column(Numeric(10, 8), nullable=True, index=True)
#     longitude = Column(Numeric(11, 8), nullable=True, index=True)
#     location = Column(Geography('POINT', srid=4326), nullable=True)  # PostGIS point
#     location_accuracy = Column(Numeric(8, 2), nullable=True)  # GPS accuracy in meters
#     location_updated_at = Column(DateTime(timezone=True), nullable=True)
    
#     # Contact Information (from dataset: "Contact" column)
#     primary_contact = Column(String(100), nullable=True)
#     phone_primary = Column(String(20), nullable=True)
#     phone_secondary = Column(String(20), nullable=True)
#     mobile = Column(String(20), nullable=True)
#     email = Column(String(100), nullable=True)
#     website = Column(String(200), nullable=True)
    
#     # Communication Details (from dataset: "Telex No." column)
#     telex_no = Column(String(50), nullable=True)
#     fax = Column(String(20), nullable=True)
    
#     # Business Classification
#     customer_type = Column(Enum(CustomerType), default=CustomerType.DISTRIBUTOR, nullable=False)
#     customer_category = Column(Enum(CustomerCategory), default=CustomerCategory.NEW, nullable=False)
#     status = Column(Enum(CustomerStatus), default=CustomerStatus.ACTIVE, nullable=False)
    
#     # Business Profile
#     business_registration_no = Column(String(50), nullable=True)
#     tax_id = Column(String(50), nullable=True)
#     vat_no = Column(String(50), nullable=True)
#     credit_limit = Column(Numeric(15, 2), default=0, nullable=False)
#     credit_days = Column(Integer, default=30, nullable=False)
#     discount_percentage = Column(Numeric(5, 2), default=0, nullable=False)
    
#     # Document and Shipping (from dataset)
#     document_sending_profile = Column(String(100), nullable=True)  # From "Document Sending Profile"
#     ship_to_code = Column(String(50), nullable=True)  # From "Ship-to Code"
#     our_account_no = Column(String(50), nullable=True)  # From "Our Account No."
    
#     # Territory and Dimensions (from dataset)
#     territory_code = Column(String(20), nullable=True, index=True)  # From "Territory Code"
#     global_dimension_1_code = Column(String(20), nullable=True)  # From "Global Dimension 1 Code"
#     global_dimension_2_code = Column(String(20), nullable=True)  # From "Global Dimension 2 Code"
#     division_code = Column(String(20), nullable=True, index=True)
#     region_code = Column(String(20), nullable=True, index=True)
    
#     # Sales Metrics
#     total_sales_ytd = Column(Numeric(15, 2), default=0, nullable=False)
#     total_sales_last_year = Column(Numeric(15, 2), default=0, nullable=False)
#     average_order_value = Column(Numeric(12, 2), default=0, nullable=False)
#     last_order_date = Column(DateTime(timezone=True), nullable=True)
#     first_order_date = Column(DateTime(timezone=True), nullable=True)
#     total_orders_count = Column(Integer, default=0, nullable=False)
    
#     # Customer Insights
#     customer_lifetime_value = Column(Numeric(15, 2), default=0, nullable=False)
#     acquisition_cost = Column(Numeric(10, 2), default=0, nullable=False)
#     satisfaction_score = Column(Numeric(3, 2), nullable=True)  # 1-5 scale
#     risk_score = Column(Numeric(3, 2), nullable=True)  # 1-5 scale
#     churn_probability = Column(Numeric(3, 2), nullable=True)  # 0-1 probability
    
#     # Operational Details
#     preferred_delivery_time = Column(String(50), nullable=True)
#     payment_terms = Column(String(100), nullable=True)
#     currency_code = Column(String(3), default="LKR", nullable=False)
#     price_list_code = Column(String(20), nullable=True)
#     sales_rep_code = Column(String(20), nullable=True)
    
#     # Timestamps
#     registration_date = Column(DateTime(timezone=True), nullable=True)
#     last_contact_date = Column(DateTime(timezone=True), nullable=True)
#     last_visit_date = Column(DateTime(timezone=True), nullable=True)
    
#     # Additional Fields
#     notes = Column(Text, nullable=True)
#     special_instructions = Column(Text, nullable=True)
#     is_key_account = Column(Boolean, default=False, nullable=False)
#     is_export_customer = Column(Boolean, default=False, nullable=False)
    
#     # Relationships
#     sales_orders = relationship("Sales", back_populates="customer", lazy="dynamic")
#     visits = relationship("CustomerVisit", back_populates="customer", cascade="all, delete-orphan")
#     contact_persons = relationship("CustomerContact", back_populates="customer", cascade="all, delete-orphan")
    
#     @hybrid_property
#     def full_address(self) -> str:
#         """Get formatted full address."""
#         parts = [self.address_line_1, self.address_line_2, self.city, self.postal_code]
#         return ", ".join(filter(None, parts))
    
#     @hybrid_property
#     def is_active_customer(self) -> bool:
#         """Check if customer is active."""
#         return self.is_active and self.status == CustomerStatus.ACTIVE
    
#     @hybrid_property
#     def days_since_last_order(self) -> Optional[int]:
#         """Calculate days since last order."""
#         if self.last_order_date:
#             from datetime import datetime
#             delta = datetime.utcnow() - self.last_order_date
#             return delta.days
#         return None
    
#     def update_location(self, latitude: float, longitude: float, accuracy: float = None) -> None:
#         """Update customer location coordinates."""
#         from datetime import datetime
#         from geoalchemy2.elements import WKTElement
        
#         self.latitude = Decimal(str(latitude))
#         self.longitude = Decimal(str(longitude))
#         self.location = WKTElement(f'POINT({longitude} {latitude})', srid=4326)
#         self.location_accuracy = Decimal(str(accuracy)) if accuracy else None
#         self.location_updated_at = datetime.utcnow()
    
#     def calculate_distance_to(self, lat: float, lon: float) -> Optional[float]:
#         """Calculate distance to given coordinates in kilometers."""
#         if not self.latitude or not self.longitude:
#             return None
        
#         from math import radians, cos, sin, asin, sqrt
        
#         # Haversine formula
#         lat1, lon1 = float(self.latitude), float(self.longitude)
#         lat2, lon2 = lat, lon
        
#         lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
#         dlat = lat2 - lat1
#         dlon = lon2 - lon1
#         a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
#         c = 2 * asin(sqrt(a))
#         r = 6371  # Radius of earth in kilometers
        
#         return c * r
    
#     def update_sales_metrics(self, order_value: Decimal, order_date: DateTime) -> None:
#         """Update sales metrics when new order is placed."""
#         from datetime import datetime
        
#         self.total_sales_ytd += order_value
#         self.total_orders_count += 1
#         self.last_order_date = order_date
        
#         if not self.first_order_date:
#             self.first_order_date = order_date
        
#         # Recalculate average order value
#         if self.total_orders_count > 0:
#             self.average_order_value = self.total_sales_ytd / self.total_orders_count
        
#         self.updated_at = datetime.utcnow()
    
#     def get_customer_segment(self) -> str:
#         """Determine customer segment based on sales metrics."""
#         if self.total_sales_ytd >= 1000000:  # 1M+
#             return "Premium"
#         elif self.total_sales_ytd >= 500000:  # 500K+
#             return "Gold"
#         elif self.total_sales_ytd >= 100000:  # 100K+
#             return "Silver"
#         else:
#             return "Bronze"


# class CustomerContact(BaseModel):
#     """Customer contact persons."""
    
#     __tablename__ = "customer_contact"
    
#     customer_id = Column(Integer, ForeignKey('customer.id'), nullable=False)
#     contact_type = Column(String(50), nullable=False)  # primary, billing, technical, etc.
#     name = Column(String(100), nullable=False)
#     designation = Column(String(100), nullable=True)
#     phone = Column(String(20), nullable=True)
#     mobile = Column(String(20), nullable=True)
#     email = Column(String(100), nullable=True)
#     is_primary = Column(Boolean, default=False, nullable=False)
    
#     # Relationships
#     customer = relationship("Customer", back_populates="contact_persons")


# class CustomerVisit(BaseModel):
#     """Customer visit tracking."""
    
#     __tablename__ = "customer_visit"
    
#     customer_id = Column(Integer, ForeignKey('customer.id'), nullable=False)
#     dealer_id = Column(Integer, ForeignKey('dealer.id'), nullable=True)
#     visit_date = Column(DateTime(timezone=True), nullable=False)
#     visit_type = Column(String(50), nullable=False)  # sales, service, collection, etc.
#     visit_purpose = Column(Text, nullable=True)
#     visit_outcome = Column(Text, nullable=True)
#     next_visit_date = Column(DateTime(timezone=True), nullable=True)
    
#     # Visit location
#     visit_latitude = Column(Numeric(10, 8), nullable=True)
#     visit_longitude = Column(Numeric(11, 8), nullable=True)
#     visit_location = Column(Geography('POINT', srid=4326), nullable=True)
    
#     # Visit metrics
#     duration_minutes = Column(Integer, nullable=True)
#     orders_generated = Column(Integer, default=0, nullable=False)
#     order_value = Column(Numeric(12, 2), default=0, nullable=False)
    
#     # Relationships
#     customer = relationship("Customer", back_populates="visits")
#     dealer = relationship("Dealer")

"""
Customer model for managing customer/distributor information, locations, and business profiles.
Based on Customer.xlsx dataset structure.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float, ForeignKey, Enum, Index
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func
import enum
import json

from .base import BaseModel


class CustomerType(enum.Enum):
    """Customer type enumeration."""
    DISTRIBUTOR = "distributor"
    RETAILER = "retailer"
    WHOLESALER = "wholesaler"
    DEALER = "dealer"
    AGENT = "agent"
    DIRECT_CUSTOMER = "direct_customer"


class CustomerStatus(enum.Enum):
    """Customer status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_APPROVAL = "pending_approval"
    BLOCKED = "blocked"
    PROSPECTIVE = "prospective"


class CustomerSegment(enum.Enum):
    """Customer segment enumeration."""
    PREMIUM = "premium"
    STANDARD = "standard"
    BASIC = "basic"
    VIP = "vip"
    ENTERPRISE = "enterprise"
    SME = "sme"


class PaymentTerm(enum.Enum):
    """Payment terms enumeration."""
    CASH = "cash"
    NET_15 = "net_15"
    NET_30 = "net_30"
    NET_45 = "net_45"
    NET_60 = "net_60"
    COD = "cod"
    ADVANCE = "advance"


class Customer(BaseModel):
    """
    Customer model representing distributors and other customers.
    Based on Customer.xlsx dataset with additional fields for comprehensive customer management.
    """
    __tablename__ = 'customers'
    
    # Basic Information (from Customer.xlsx)
    customer_code = Column(String(50), unique=True, nullable=False, index=True)  # Customer ID from dataset
    city = Column(String(100), nullable=False, index=True)  # City from dataset
    contact = Column(String(100), nullable=True)  # Contact from dataset
    telex_no = Column(String(50), nullable=True)  # Telex No. from dataset
    
    # Extended Basic Information
    name = Column(String(255), nullable=False, index=True)
    legal_name = Column(String(255), nullable=True)
    customer_type = Column(Enum(CustomerType), default=CustomerType.DISTRIBUTOR, nullable=False)
    status = Column(Enum(CustomerStatus), default=CustomerStatus.ACTIVE, nullable=False)
    segment = Column(Enum(CustomerSegment), default=CustomerSegment.STANDARD, nullable=False)
    
    # Business Information
    business_registration_no = Column(String(100), nullable=True, index=True)
    tax_id = Column(String(50), nullable=True, index=True)
    vat_no = Column(String(50), nullable=True)
    industry = Column(String(100), nullable=True)
    business_type = Column(String(100), nullable=True)
    establishment_date = Column(DateTime(timezone=True), nullable=True)
    
    # Contact Information
    primary_phone = Column(String(20), nullable=True, index=True)
    secondary_phone = Column(String(20), nullable=True)
    mobile = Column(String(20), nullable=True, index=True)
    email = Column(String(255), nullable=True, index=True)
    website = Column(String(255), nullable=True)
    fax = Column(String(50), nullable=True)
    
    # Address Information
    address_line_1 = Column(String(255), nullable=True)
    address_line_2 = Column(String(255), nullable=True)
    state_province = Column(String(100), nullable=True, index=True)
    postal_code = Column(String(20), nullable=True, index=True)
    country = Column(String(100), default='Sri Lanka', nullable=False)
    
    # Location Data (from customer.csv dataset)
    latitude = Column(Float, nullable=True, index=True)  # For geo-location
    longitude = Column(Float, nullable=True, index=True)  # For geo-location
    location_accuracy = Column(Float, nullable=True)  # GPS accuracy in meters
    location_updated_at = Column(DateTime(timezone=True), nullable=True)
    delivery_zone = Column(String(50), nullable=True, index=True)
    
    # Territory and Hierarchy (from Customer.xlsx)
    territory_code = Column(String(20), nullable=True, index=True)  # Territory Code from dataset
    global_dimension_1_code = Column(String(50), nullable=True)  # Global Dimension 1 Code from dataset
    global_dimension_2_code = Column(String(50), nullable=True)  # Global Dimension 2 Code from dataset
    ship_to_code = Column(String(50), nullable=True)  # Ship-to Code from dataset
    document_sending_profile = Column(String(100), nullable=True)  # Document Sending Profile from dataset
    our_account_no = Column(String(100), nullable=True)  # Our Account No. from dataset
    
    # Business Metrics
    credit_limit = Column(Float, default=0.0, nullable=False)
    available_credit = Column(Float, default=0.0, nullable=False)
    payment_terms = Column(Enum(PaymentTerm), default=PaymentTerm.NET_30, nullable=False)
    discount_percentage = Column(Float, default=0.0, nullable=False)
    commission_rate = Column(Float, default=0.0, nullable=False)
    
    # Performance Metrics
    total_orders = Column(Integer, default=0, nullable=False)
    total_order_value = Column(Float, default=0.0, nullable=False)
    last_order_date = Column(DateTime(timezone=True), nullable=True)
    last_payment_date = Column(DateTime(timezone=True), nullable=True)
    average_order_value = Column(Float, default=0.0, nullable=False)
    outstanding_amount = Column(Float, default=0.0, nullable=False)
    
    # Customer Lifecycle
    acquisition_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    first_order_date = Column(DateTime(timezone=True), nullable=True)
    last_activity_date = Column(DateTime(timezone=True), nullable=True)
    lifecycle_stage = Column(String(50), default='new', nullable=False)  # new, active, dormant, churned
    
    # Preferences and Settings
    preferred_delivery_time = Column(String(100), nullable=True)
    preferred_contact_method = Column(String(50), default='phone', nullable=True)
    language_preference = Column(String(10), default='en', nullable=False)
    currency_preference = Column(String(10), default='LKR', nullable=False)
    
    # Risk and Compliance
    risk_level = Column(String(20), default='low', nullable=False)  # low, medium, high
    kyc_status = Column(String(50), default='pending', nullable=False)
    kyc_verified_date = Column(DateTime(timezone=True), nullable=True)
    compliance_score = Column(Float, nullable=True)
    
    # Sales Team Assignment
    assigned_sales_rep_id = Column(Integer, ForeignKey('users.id'), nullable=True, index=True)
    assigned_territory_manager_id = Column(Integer, ForeignKey('users.id'), nullable=True, index=True)
    assignment_date = Column(DateTime(timezone=True), nullable=True)
    
    # Marketing and Segmentation
    marketing_consent = Column(Boolean, default=False, nullable=False)
    email_marketing_consent = Column(Boolean, default=False, nullable=False)
    sms_marketing_consent = Column(Boolean, default=False, nullable=False)
    customer_source = Column(String(100), nullable=True)  # referral, advertisement, cold_call, etc.
    referral_code = Column(String(50), nullable=True)
    
    # Additional Metadata
    notes = Column(Text, nullable=True)
    tags = Column(Text, nullable=True)  # JSON array of tags
    custom_fields = Column(Text, nullable=True)  # JSON object for custom fields
    
    # Integration Fields
    external_id = Column(String(100), nullable=True, index=True)  # For external system integration
    sync_status = Column(String(50), default='synced', nullable=False)
    last_sync_date = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    sales_rep = relationship("User", foreign_keys=[assigned_sales_rep_id], backref="assigned_customers")
    territory_manager = relationship("User", foreign_keys=[assigned_territory_manager_id], backref="managed_customers")
    
    # Customer-related orders (from sales model)
    orders = relationship("Sales", back_populates="customer", lazy="dynamic")
    
    # Customer contacts
    contacts = relationship("CustomerContact", back_populates="customer", cascade="all, delete-orphan")
    
    # Customer visits
    visits = relationship("CustomerVisit", back_populates="customer", lazy="dynamic")
    
    # Customer feedback
    feedback = relationship("CustomerFeedback", back_populates="customer", lazy="dynamic")
    
    @hybrid_property
    def full_address(self):
        """Get formatted full address."""
        parts = [self.address_line_1, self.address_line_2, self.city, self.state_province, self.postal_code]
        return ", ".join([part for part in parts if part])
    
    @hybrid_property
    def is_active_customer(self):
        """Check if customer is active."""
        return self.status == CustomerStatus.ACTIVE and self.is_active
    
    @hybrid_property
    def has_valid_location(self):
        """Check if customer has valid GPS coordinates."""
        return self.latitude is not None and self.longitude is not None
    
    @hybrid_property
    def days_since_last_order(self):
        """Calculate days since last order."""
        if self.last_order_date:
            return (datetime.utcnow() - self.last_order_date).days
        return None
    
    @hybrid_property
    def credit_utilization_percentage(self):
        """Calculate credit utilization percentage."""
        if self.credit_limit > 0:
            used_credit = self.credit_limit - self.available_credit
            return (used_credit / self.credit_limit) * 100
        return 0
    
    def update_location(self, latitude: float, longitude: float, accuracy: float = None):
        """Update customer location coordinates."""
        self.latitude = latitude
        self.longitude = longitude
        self.location_accuracy = accuracy
        self.location_updated_at = datetime.utcnow()
    
    def calculate_distance_from(self, lat: float, lon: float) -> float:
        """Calculate distance from given coordinates using Haversine formula."""
        if not self.has_valid_location:
            return None
        
        import math
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [self.latitude, self.longitude, lat, lon])
        
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
        # This would typically be called after order updates
        if self.orders.count() > 0:
            self.total_orders = self.orders.count()
            self.total_order_value = sum([order.final_value for order in self.orders if order.final_value])
            self.average_order_value = self.total_order_value / self.total_orders if self.total_orders > 0 else 0
            self.last_order_date = self.orders.order_by('created_at desc').first().created_at
            self.first_order_date = self.orders.order_by('created_at asc').first().created_at
    
    def update_lifecycle_stage(self):
        """Update customer lifecycle stage."""
        if not self.first_order_date:
            self.lifecycle_stage = 'prospect'
        elif self.days_since_last_order is None:
            self.lifecycle_stage = 'new'
        elif self.days_since_last_order <= 30:
            self.lifecycle_stage = 'active'
        elif self.days_since_last_order <= 90:
            self.lifecycle_stage = 'dormant'
        else:
            self.lifecycle_stage = 'churned'
    
    def add_tag(self, tag: str):
        """Add a tag to customer."""
        tags = json.loads(self.tags) if self.tags else []
        if tag not in tags:
            tags.append(tag)
            self.tags = json.dumps(tags)
    
    def remove_tag(self, tag: str):
        """Remove a tag from customer."""
        tags = json.loads(self.tags) if self.tags else []
        if tag in tags:
            tags.remove(tag)
            self.tags = json.dumps(tags)
    
    def get_tags(self) -> List[str]:
        """Get all customer tags."""
        return json.loads(self.tags) if self.tags else []
    
    def set_custom_field(self, key: str, value: Any):
        """Set a custom field value."""
        custom_fields = json.loads(self.custom_fields) if self.custom_fields else {}
        custom_fields[key] = value
        self.custom_fields = json.dumps(custom_fields)
    
    def get_custom_field(self, key: str):
        """Get a custom field value."""
        custom_fields = json.loads(self.custom_fields) if self.custom_fields else {}
        return custom_fields.get(key)
    
    def update_credit_limit(self, new_limit: float, reason: str = None):
        """Update credit limit and log the change."""
        old_limit = self.credit_limit
        self.credit_limit = new_limit
        self.available_credit = new_limit - (old_limit - self.available_credit)
        
        # Log the change in metadata
        change_log = {
            'old_limit': old_limit,
            'new_limit': new_limit,
            'reason': reason,
            'changed_at': datetime.utcnow().isoformat()
        }
        self.update_metadata('credit_limit_changes', change_log)
    
    def block_customer(self, reason: str = None):
        """Block customer."""
        self.status = CustomerStatus.BLOCKED
        self.is_active = False
        if reason:
            self.update_metadata('block_reason', reason)
            self.update_metadata('blocked_at', datetime.utcnow().isoformat())
    
    def unblock_customer(self, reason: str = None):
        """Unblock customer."""
        self.status = CustomerStatus.ACTIVE
        self.is_active = True
        if reason:
            self.update_metadata('unblock_reason', reason)
            self.update_metadata('unblocked_at', datetime.utcnow().isoformat())
    
    def to_dict(self, include_relationships: bool = False):
        """Convert to dictionary with additional computed fields."""
        data = super().to_dict()
        
        # Add computed fields
        data['full_address'] = self.full_address
        data['has_valid_location'] = self.has_valid_location
        data['days_since_last_order'] = self.days_since_last_order
        data['credit_utilization_percentage'] = self.credit_utilization_percentage
        data['tags'] = self.get_tags()
        
        if include_relationships:
            data['sales_rep_name'] = self.sales_rep.full_name if self.sales_rep else None
            data['territory_manager_name'] = self.territory_manager.full_name if self.territory_manager else None
            data['total_contacts'] = self.contacts.count() if self.contacts else 0
            data['recent_visits'] = [visit.to_dict() for visit in self.visits.limit(5)]
        
        return data
    
    def __repr__(self):
        return f"<Customer(code={self.customer_code}, name={self.name}, city={self.city})>"


# Add indexes for better performance
Index('idx_customer_location', Customer.latitude, Customer.longitude)
Index('idx_customer_territory', Customer.territory_code, Customer.city)
Index('idx_customer_performance', Customer.status, Customer.last_order_date)


class CustomerContact(BaseModel):
    """Customer contact persons."""
    __tablename__ = 'customer_contacts'
    
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False, index=True)
    contact_type = Column(String(50), default='primary', nullable=False)  # primary, secondary, billing, technical
    
    # Contact Information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    job_title = Column(String(100), nullable=True)
    department = Column(String(100), nullable=True)
    
    # Communication
    phone = Column(String(20), nullable=True)
    mobile = Column(String(20), nullable=True)
    email = Column(String(255), nullable=True, index=True)
    whatsapp = Column(String(20), nullable=True)
    
    # Preferences
    preferred_contact_method = Column(String(50), default='phone', nullable=False)
    best_contact_time = Column(String(100), nullable=True)
    language_preference = Column(String(10), default='en', nullable=False)
    
    # Status
    is_primary = Column(Boolean, default=False, nullable=False)
    is_decision_maker = Column(Boolean, default=False, nullable=False)
    
    # Relationship
    customer = relationship("Customer", back_populates="contacts")
    
    @hybrid_property
    def full_name(self):
        """Get full name."""
        return f"{self.first_name} {self.last_name}"
    
    def __repr__(self):
        return f"<CustomerContact(name={self.full_name}, customer_id={self.customer_id})>"


class CustomerVisit(BaseModel):
    """Customer visit tracking."""
    __tablename__ = 'customer_visits'
    
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False, index=True)
    visited_by_user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    
    # Visit Details
    visit_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    visit_type = Column(String(50), nullable=False)  # sales_call, delivery, support, collection
    purpose = Column(String(255), nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    
    # Location
    visit_latitude = Column(Float, nullable=True)
    visit_longitude = Column(Float, nullable=True)
    location_accuracy = Column(Float, nullable=True)
    
    # Visit Details
    contact_person = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)
    outcome = Column(String(255), nullable=True)
    follow_up_required = Column(Boolean, default=False, nullable=False)
    follow_up_date = Column(DateTime(timezone=True), nullable=True)
    
    # Status
    status = Column(String(50), default='completed', nullable=False)  # planned, in_progress, completed, cancelled
    
    # Relationships
    customer = relationship("Customer", back_populates="visits")
    visited_by = relationship("User", backref="customer_visits")
    
    def __repr__(self):
        return f"<CustomerVisit(customer_id={self.customer_id}, date={self.visit_date}, type={self.visit_type})>"


class CustomerFeedback(BaseModel):
    """Customer feedback and ratings."""
    __tablename__ = 'customer_feedback'
    
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False, index=True)
    feedback_type = Column(String(50), nullable=False)  # service, product, delivery, support
    
    # Ratings (1-5 scale)
    overall_rating = Column(Integer, nullable=True)
    service_rating = Column(Integer, nullable=True)
    product_rating = Column(Integer, nullable=True)
    delivery_rating = Column(Integer, nullable=True)
    
    # Feedback Details
    subject = Column(String(255), nullable=True)
    feedback_text = Column(Text, nullable=True)
    suggestions = Column(Text, nullable=True)
    
    # Context
    related_order_id = Column(Integer, ForeignKey('sales.id'), nullable=True)
    feedback_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Status
    status = Column(String(50), default='new', nullable=False)  # new, reviewed, resolved, closed
    reviewed_by_user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    response = Column(Text, nullable=True)
    response_date = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    customer = relationship("Customer", back_populates="feedback")
    reviewed_by = relationship("User", backref="reviewed_feedback")
    
    def __repr__(self):
        return f"<CustomerFeedback(customer_id={self.customer_id}, type={self.feedback_type}, rating={self.overall_rating})>"