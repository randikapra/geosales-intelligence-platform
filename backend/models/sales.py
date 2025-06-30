# # # # backend/models/sales.py
# # # """
# # # Sales and orders models
# # # """
# # # from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Integer, Text
# # # from sqlalchemy.orm import relationship
# # # from .base import BaseModel


# # # class Sale(BaseModel):
# # #     __tablename__ = "sales"
    
# # #     code = Column(String(100), unique=True, index=True)
# # #     date = Column(DateTime, nullable=False)
# # #     distributor_code = Column(String(50), ForeignKey("customers.customer_code"))
# # #     user_code = Column(String(50), ForeignKey("dealers.user_code"))
# # #     user_name = Column(String(255))
# # #     final_value = Column(Float, nullable=False)
# # #     creation_date = Column(DateTime)
# # #     submitted_date = Column(DateTime)
# # #     erp_order_number = Column(String(100))
    
# # #     # Additional fields
# # #     product_category = Column(String(100))
# # #     quantity = Column(Float)
# # #     discount = Column(Float, default=0.0)
# # #     tax_amount = Column(Float, default=0.0)
# # #     status = Column(String(50), default="completed")
# # #     notes = Column(Text)
    
# # #     # Relationships
# # #     customer = relationship("Customer", back_populates="sales")
# # #     dealer = relationship("Dealer", back_populates="sales")


# # # class Order(BaseModel):
# # #     __tablename__ = "orders"
    
# # #     code = Column(String(100), unique=True, index=True)
# # #     distributor_code = Column(String(50), ForeignKey("customers.customer_code"))
# # #     user_code = Column(String(50), ForeignKey("dealers.user_code"))
# # #     latitude = Column(Float)
# # #     longitude = Column(Float)
# # #     days = Column(String(20))
# # #     total = Column(Float, nullable=False)
    
# # #     # Order details
# # #     order_date = Column(DateTime)
# # #     delivery_date = Column(DateTime)
# # #     priority = Column(String(20), default="normal")
    
# # #     # Relationships
# # #     customer = relationship("Customer", back_populates="orders")

# # from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
# # from sqlalchemy.orm import relationship
# # from geoalchemy2 import Geography
# # from .base import Base

# # class Order(Base):
# #     __tablename__ = "orders"
    
# #     id = Column(Integer, primary_key=True, index=True)
# #     code = Column(String, unique=True, index=True)
# #     date = Column(DateTime)
# #     distributor_code = Column(String)
# #     customer_id = Column(Integer, ForeignKey("customers.id"))
# #     dealer_id = Column(Integer, ForeignKey("dealers.id"))
# #     final_value = Column(Float)
# #     creation_date = Column(DateTime)
# #     submitted_date = Column(DateTime)
# #     erp_order_number = Column(String)
# #     order_latitude = Column(Float)
# #     order_longitude = Column(Float)
# #     order_location = Column(Geography(geometry_type='POINT', srid=4326))
    
# #     # Relationships
# #     customer = relationship("Customer", back_populates="orders")
# #     dealer = relationship("Dealer", back_populates="orders")
# #     order_items = relationship("OrderItem", back_populates="order")

# # class OrderItem(Base):
# #     __tablename__ = "order_items"
    
# #     id = Column(Integer, primary_key=True, index=True)
# #     order_id = Column(Integer, ForeignKey("orders.id"))
# #     product_code = Column(String)
# #     product_name = Column(String)
# #     quantity = Column(Float)
# #     unit_price = Column(Float)
# #     total_value = Column(Float)
    
# #     # Relationships
# #     order = relationship("Order", back_populates="order_items")


# """
# Sales model for sales transactions, order details, and product information.
# Based on SFA_Orders.xlsx and SFA_PO datasets.
# """
# import enum
# from decimal import Decimal
# from typing import Optional, List, Dict
# from datetime import datetime, timedelta
# from sqlalchemy import (
#     Column, Integer, String, Numeric, DateTime, Boolean, Text, 
#     Enum, ForeignKey, JSON, Index
# )
# from sqlalchemy.orm import relationship
# from sqlalchemy.ext.hybrid import hybrid_property
# from geoalchemy2 import Geography
# from .base import BaseModel


# class OrderStatus(enum.Enum):
#     """Order status enumeration."""
#     DRAFT = "draft"
#     SUBMITTED = "submitted"
#     CONFIRMED = "confirmed"
#     PROCESSING = "processing"
#     SHIPPED = "shipped"
#     DELIVERED = "delivered"
#     CANCELLED = "cancelled"
#     RETURNED = "returned"


# class PaymentStatus(enum.Enum):
#     """Payment status enumeration."""
#     PENDING = "pending"
#     PARTIAL = "partial"
#     PAID = "paid"
#     OVERDUE = "overdue"
#     CANCELLED = "cancelled"


# class OrderType(enum.Enum):
#     """Order type enumeration."""
#     SALES_ORDER = "sales_order"
#     PURCHASE_ORDER = "purchase_order"
#     RETURN_ORDER = "return_order"
#     SAMPLE_ORDER = "sample_order"


# class Sales(BaseModel):
#     """
#     Sales/Orders model based on SFA_Orders.xlsx dataset.
#     Stores sales transactions and order details.
#     """
    
#     __tablename__ = "sales"
    
#     # Primary order identifier (from dataset: "Code" column)
#     order_code = Column(String(50), unique=True, nullable=False, index=True)
    
#     # Order Information (from dataset: "Date" column)
#     order_date = Column(DateTime(timezone=True), nullable=False, index=True)
    
#     # Customer Information (from dataset: "DistributorCode" column)
#     distributor_code = Column(String(50), ForeignKey('customer.customer_code'), nullable=False, index=True)
#     customer_id = Column(Integer, ForeignKey('customer.id'), nullable=True)
    
#     # Dealer Information (from dataset: "UserCode" and "UserName" columns)
#     user_code = Column(String(20), ForeignKey('dealer.user_code'), nullable=False, index=True)
#     dealer_id = Column(Integer, ForeignKey('dealer.id'), nullable=True)
#     user_name = Column(String(100), nullable=False)  # Redundant but from dataset
    
#     # Financial Information (from dataset: "FinalValue" column)
#     final_value = Column(Numeric(15, 2), nullable=False)
#     gross_amount = Column(Numeric(15, 2), nullable=True)
#     discount_amount = Column(Numeric(10, 2), default=0, nullable=False)
#     tax_amount = Column(Numeric(10, 2), default=0, nullable=False)
#     net_amount = Column(Numeric(15, 2), nullable=True)
    
#     # Timestamps (from dataset: "CreationDate" and "SubmittedDate" columns)
#     creation_date = Column(DateTime(timezone=True), nullable=False)
#     submitted_date = Column(DateTime(timezone=True), nullable=True)
    
#     # ERP Integration (from dataset: "ERPOrderNumber" column)
#     erp_order_number = Column(String(50), nullable=True, index=True)
    
#     # Order Status and Type
#     order_status = Column(Enum(OrderStatus), default=OrderStatus.DRAFT, nullable=False)
#     order_type = Column(Enum(OrderType), default=OrderType.SALES_ORDER, nullable=False)
#     payment_status = Column(Enum(PaymentStatus), default=PaymentStatus.PENDING, nullable=False)
    
#     # Additional Order Details
#     reference_number = Column(String(50), nullable=True)
#     po_number = Column(String(50), nullable=True)  # Customer PO number
#     quote_number = Column(String(50), nullable=True)
    
#     # Delivery Information
#     delivery_date = Column(DateTime(timezone=True), nullable=True)
#     delivery_address = Column(Text, nullable=True)
#     delivery_notes = Column(Text, nullable=True)
#     shipping_method = Column(String(50), nullable=True)
#     tracking_number = Column(String(100), nullable=True)
    
#     # Location Information (when order was placed)
#     order_latitude = Column(Numeric(10, 8), nullable=True)
#     order_longitude = Column(Numeric(11, 8), nullable=True)
#     order_location = Column(Geography('POINT', srid=4326), nullable=True)
#     location_accuracy = Column(Numeric(8, 2), nullable=True)
    
#     # Territory and Business Context
#     territory_code = Column(String(20), nullable=True, index=True)
#     division_code = Column(String(20), nullable=True, index=True)
#     region_code = Column(String(20), nullable=True, index=True)
#     tour_code = Column(String(50), nullable=True, index=True)
    
#     # Payment Information
#     payment_terms = Column(String(100), nullable=True)
#     payment_method = Column(String(50), nullable=True)
#     credit_days = Column(Integer, default=30, nullable=False)
#     due_date = Column(DateTime(timezone=True), nullable=True)
    
#     # Additional Metrics
#     total_items = Column(Integer, default=0, nullable=False)
#     total_quantity = Column(Numeric(10, 3), default=0, nullable=False)
#     average_item_value = Column(Numeric(10, 2), default=0, nullable=False)
    
#     # Approval and Workflow
#     approved_by = Column(String(100), nullable=True)
#     approved_date = Column(DateTime(timezone=True), nullable=True)
#     approval_notes = Column(Text, nullable=True)
    
#     # Additional Fields
#     notes = Column(Text, nullable=True)
#     internal_notes = Column(Text, nullable=True)
#     special_instructions = Column(Text, nullable=True)
    
#     # Audit Fields
#     created_by_id = Column(Integer, ForeignKey('user.id'), nullable=True)
#     modified_by_id = Column(Integer, ForeignKey('user.id'), nullable=True)
    
#     # Relationships
#     customer = relationship("Customer", back_populates="sales_orders")
#     dealer = relationship("Dealer", back_populates="sales_orders")
#     order_items = relationship("SalesItem", back_populates="sales_order", cascade="all, delete-orphan")
#     payments = relationship("SalesPayment", back_populates="sales_order", cascade="all, delete-orphan")
#     created_by_user = relationship("User", foreign_keys=[created_by_id], back_populates="created_sales")
#     modified_by_user = relationship("User", foreign_keys=[modified_by_id])
    
#     # Indexes for performance
#     __table_args__ = (
#         Index('idx_sales_date_dealer', 'order_date', 'user_code'),
#         Index('idx_sales_customer_date', 'distributor_code', 'order_date'),
#         Index('idx_sales_territory_date', 'territory_code', 'order_date'),
#         Index('idx_sales_status_date', 'order_status', 'order_date'),
#     )
    
#     @hybrid_property
#     def days_since_order(self) -> int:
#         """Calculate days since order was placed."""
#         delta = datetime.utcnow() - self.order_date
#         return delta.days
    
#     @hybrid_property
#     def is_overdue(self) -> bool:
#         """Check if order is overdue for payment."""
#         if self.due_date and self.payment_status != PaymentStatus.PAID:
#             return datetime.utcnow() > self.due_date
#         return False
    
#     @hybrid_property
#     def days_overdue(self) -> Optional[int]:
#         """Calculate days overdue for payment."""
#         if self.is_overdue:
#             delta = datetime.utcnow() - self.due_date
#             return delta.days
#         return None
    
#     @hybrid_property
#     def profit_margin(self) -> Optional[Decimal]:
#         """Calculate profit margin if cost is available."""
#         total_cost = sum(item.total_cost or 0 for item in self.order_items)
#         if total_cost > 0 and self.final_value > 0:
#             profit = self.final_value - total_cost
#             return (profit / self.final_value) * 100
#         return None
    
#     def calculate_totals(self) -> None:
#         """Calculate order totals from line items."""
#         if not self.order_items:
#             return
        
#         self.gross_amount = sum(item.line_total for item in self.order_items)
#         self.total_items = len(self.order_items)
#         self.total_quantity = sum(item.quantity for item in self.order_items)
        
#         # Calculate net amount
#         self.net_amount = self.gross_amount - self.discount_amount + self.tax_amount
#         self.final_value = self.net_amount
        
#         # Calculate average item value
#         if self.total_items > 0:
#             self.average_item_value = self.final_value / self.total_items
    
#     def set_order_location(self, latitude: float, longitude: float, accuracy: float = None) -> None:
#         """Set the location where order was placed."""
#         from geoalchemy2.elements import WKTElement
        
#         self.order_latitude = Decimal(str(latitude))
#         self.order_longitude = Decimal(str(longitude))
#         self.order_location = WKTElement(f'POINT({longitude} {latitude})', srid=4326)
#         self.location_accuracy = Decimal(str(accuracy)) if accuracy else None
    
#     def calculate_due_date(self) -> None:
#         """Calculate due date based on credit days."""
#         if self.order_date and self.credit_days:
#             self.due_date = self.order_date + timedelta(days=self.credit_days)
    
#     def submit_order(self) -> None:
#         """Submit the order."""
#         self.order_status = OrderStatus.SUBMITTED
#         self.submitted_date = datetime.utcnow()
#         self.calculate_due_date()
    
#     def confirm_order(self, approved_by: str = None) -> None:
#         """Confirm the order."""
#         self.order_status = OrderStatus.CONFIRMED
#         self.approved_by = approved_by
#         self.approved_date = datetime.utcnow()
    
#     def cancel_order(self, reason: str = None) -> None:
#         """Cancel the order."""
#         self.order_status = OrderStatus.CANCELLED
#         if reason:
#             self.notes = f"Cancelled: {reason}"
    
#     def get_order_summary(self) -> Dict:
#         """Get order summary for reporting."""
#         return {
#             'order_code': self.order_code,
#             'order_date': self.order_date,
#             'customer_code': self.distributor_code,
#             'dealer_code': self.user_code,
#             'final_value': float(self.final_value),
#             'status': self.order_status.value,
#             'payment_status': self.payment_status.value,
#             'total_items': self.total_items,
#             'days_since_order': self.days_since_order,
#             'is_overdue': self.is_overdue
#         }


# class SalesItem(BaseModel):
#     """
#     Sales order line items.
#     """
    
#     __tablename__ = "sales_item"
    
#     sales_order_id = Column(Integer, ForeignKey('sales.id'), nullable=False)
#     line_number = Column(Integer, nullable=False)
    
#     # Product Information
#     product_code = Column(String(50), nullable=False, index=True)
#     product_name = Column(String(200), nullable=False)
#     product_description = Column(Text, nullable=True)
#     product_category = Column(String(100), nullable=True)
    
#     # Quantity and Units
#     quantity = Column(Numeric(10, 3), nullable=False)
#     unit_of_measure = Column(String(20), nullable=False)
    
#     # Pricing
#     unit_price = Column(Numeric(10, 2), nullable=False)
#     line_total = Column(Numeric(12, 2), nullable=False)
#     discount_percentage = Column(Numeric(5, 2), default=0, nullable=False)
#     discount_amount = Column(Numeric(8, 2), default=0, nullable=False)
#     tax_percentage = Column(Numeric(5, 2), default=0, nullable=False)
#     tax_amount = Column(Numeric(8, 2), default=0, nullable=False)
#     net_amount = Column(Numeric(12, 2), nullable=False)
    
#     # Cost Information (for profit calculation)
#     unit_cost = Column(Numeric(10, 2), nullable=True)
#     total_cost = Column(Numeric(12, 2), nullable=True)
    
#     # Additional Details
#     brand = Column(String(100), nullable=True)
#     model = Column(String(100), nullable=True)
#     size = Column(String(50), nullable=True)
#     color = Column(String(50), nullable=True)
    
#     # Inventory
#     warehouse_code = Column(String(20), nullable=True)
#     batch_number = Column(String(50), nullable=True)
#     expiry_date = Column(DateTime(timezone=True), nullable=True)
    
#     # Delivery
#     delivery_quantity = Column(Numeric(10, 3), default=0, nullable=False)
#     remaining_quantity = Column(Numeric(10, 3), nullable=True)
    
#     # Relationships
#     sales_order = relationship("Sales", back_populates="order_items")
    
#     def calculate_line_total(self) -> None:
#         """Calculate line total including discounts and taxes."""
#         gross_amount = self.quantity * self.unit_price
#         self.line_total = gross_amount
        
#         # Apply discount
#         if self.discount_percentage > 0:
#             self.discount_amount = gross_amount * (self.discount_percentage / 100)
#         elif self.discount_amount > 0:
#             self.discount_percentage = (self.discount_amount / gross_amount) * 100
        
#         # Apply tax
#         taxable_amount = gross_amount - self.discount_amount
#         if self.tax_percentage > 0:
#             self.tax_amount = taxable_amount * (self.tax_percentage / 100)
        
#         self.net_amount = taxable_amount + self.tax_amount
        
#         # Calculate cost
#         if self.unit_cost:
#             self.total_cost = self.quantity * self.unit_cost


# class SalesPayment(BaseModel):
#     """
#     Sales order payment tracking.
#     """
    
#     __tablename__ = "sales_payment"
    
#     sales_order_id = Column(Integer, ForeignKey('sales.id'), nullable=False)
#     payment_reference = Column(String(100), unique=True, nullable=False, index=True)
    
#     # Payment Details
#     payment_date = Column(DateTime(timezone=True), nullable=False)
#     payment_amount = Column(Numeric(12, 2), nullable=False)
#     payment_method = Column(String(50), nullable=False)  # cash, bank_transfer, cheque, card
    
#     # Bank Details
#     bank_name = Column(String(100), nullable=True)
#     account_number = Column(String(50), nullable=True)
#     cheque_number = Column(String(50), nullable=True)
#     cheque_date = Column(DateTime(timezone=True), nullable=True)
    
#     # Transaction Details
#     transaction_id = Column(String(100), nullable=True)
#     gateway_reference = Column(String(100), nullable=True)
    
#     # Status
#     payment_status = Column(Enum(PaymentStatus), default=PaymentStatus.PENDING, nullable=False)
#     cleared_date = Column(DateTime(timezone=True), nullable=True)
    
#     # Additional Information
#     notes = Column(Text, nullable=True)
#     receipt_number = Column(String(50), nullable=True)
    
#     # Audit
#     recorded_by_id = Column(Integer, ForeignKey('user.id'), nullable=True)
#     verified_by_id = Column(Integer, ForeignKey('user.id'), nullable=True)
#     verified_date = Column(DateTime(timezone=True), nullable=True)
    
#     # Relationships
#     sales_order = relationship("Sales", back_populates="payments")
#     recorded_by = relationship("User", foreign_keys=[recorded_by_id])
#     verified_by = relationship("User", foreign_keys=[verified_by_id])
    
#     def mark_as_cleared(self, verified_by_id: int = None) -> None:
#         """Mark payment as cleared."""
#         self.payment_status = PaymentStatus.PAID
#         self.cleared_date = datetime.utcnow()
#         self.verified_by_id = verified_by_id
#         self.verified_date = datetime.utcnow()


# class PurchaseOrder(BaseModel):
#     """
#     Purchase Orders model based on SFA_PO dataset.
#     """
    
#     __tablename__ = "purchase_order"
    
#     # Primary identifier (from dataset: "Code" column)
#     po_code = Column(String(50), unique=True, nullable=False, index=True)
    
#     # Customer/Distributor Information (from dataset: "DistributorCode" column)
#     distributor_code = Column(String(50), ForeignKey('customer.customer_code'), nullable=False, index=True)
#     customer_id = Column(Integer, ForeignKey('customer.id'), nullable=True)
    
#     # Dealer Information (from dataset: "UserCode" column)
#     user_code = Column(String(20), ForeignKey('dealer.user_code'), nullable=False, index=True)
#     dealer_id = Column(Integer, ForeignKey('dealer.id'), nullable=True)
    
#     # Location Information (from dataset: "Latitude", "Longitude" columns)
#     po_latitude = Column(Numeric(10, 8), nullable=True)
#     po_longitude = Column(Numeric(11, 8), nullable=True)
#     po_location = Column(Geography('POINT', srid=4326), nullable=True)
    
#     # Timeline Information (from dataset: "Days" column)
#     delivery_days = Column(String(20), nullable=True)  # e.g., "3-Apr", "7-May"
#     expected_delivery_date = Column(DateTime(timezone=True), nullable=True)
    
#     # Financial Information (from dataset: "Total" column)
#     total_amount = Column(Numeric(12, 2), nullable=False)
    
#     # PO Status
#     po_status = Column(Enum(OrderStatus), default=OrderStatus.DRAFT, nullable=False)
    
#     # Timestamps
#     po_date = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
#     created_date = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
#     # Additional Information
#     supplier_code = Column(String(50), nullable=True)
#     supplier_name = Column(String(200), nullable=True)
#     delivery_address = Column(Text, nullable=True)
#     special_instructions = Column(Text, nullable=True)
    
#     # Relationships
#     customer = relationship("Customer")
#     dealer = relationship("Dealer")
#     po_items = relationship("PurchaseOrderItem", back_populates="purchase_order", cascade="all, delete-orphan")
    
#     # Indexes
#     __table_args__ = (
#         Index('idx_po_distributor_date', 'distributor_code', 'po_date'),
#         Index('idx_po_dealer_date', 'user_code', 'po_date'),
#         Index('idx_po_status_date', 'po_status', 'po_date'),
#     )
    
#     def set_po_location(self, latitude: float, longitude: float) -> None:
#         """Set the location for the purchase order."""
#         from geoalchemy2.elements import WKTElement
        
#         self.po_latitude = Decimal(str(latitude))
#         self.po_longitude = Decimal(str(longitude))
#         self.po_location = WKTElement(f'POINT({longitude} {latitude})', srid=4326)
    
#     def parse_delivery_days(self, days_str: str) -> None:
#         """Parse delivery days string and set expected delivery date."""
#         # Parse strings like "3-Apr", "7-May" etc.
#         try:
#             import re
#             from datetime import datetime
            
#             match = re.match(r'(\d+)-(\w+)', days_str)
#             if match:
#                 day = int(match.group(1))
#                 month_str = match.group(2)
                
#                 month_mapping = {
#                     'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
#                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
#                 }
                
#                 if month_str in month_mapping:
#                     current_year = datetime.now().year
#                     self.expected_delivery_date = datetime(current_year, month_mapping[month_str], day)
                    
#         except Exception:
#             # If parsing fails, keep the original string
#             pass


# class PurchaseOrderItem(BaseModel):
#     """
#     Purchase Order line items.
#     """
    
#     __tablename__ = "purchase_order_item"
    
#     purchase_order_id = Column(Integer, ForeignKey('purchase_order.id'), nullable=False)
#     line_number = Column(Integer, nullable=False)
    
#     # Product Information
#     product_code = Column(String(50), nullable=False, index=True)
#     product_name = Column(String(200), nullable=False)
#     product_description = Column(Text, nullable=True)
    
#     # Quantity and Pricing
#     quantity = Column(Numeric(10, 3), nullable=False)
#     unit_price = Column(Numeric(10, 2), nullable=False)
#     line_total = Column(Numeric(12, 2), nullable=False)
#     unit_of_measure = Column(String(20), nullable=False)
    
#     # Delivery Status
#     delivered_quantity = Column(Numeric(10, 3), default=0, nullable=False)
#     remaining_quantity = Column(Numeric(10, 3), nullable=True)
    
#     # Relationships
#     purchase_order = relationship("PurchaseOrder", back_populates="po_items")
    
#     def calculate_remaining_quantity(self) -> None:
#         """Calculate remaining quantity to be delivered."""
#         self.remaining_quantity = self.quantity - self.delivered_quantity


"""
Sales model for managing sales transactions, orders, and product information.
Based on SFA_Orders.xlsx and SFA_PO datasets structure.
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


class OrderStatus(enum.Enum):
    """Order status enumeration."""
    DRAFT = "draft"
    PENDING = "pending"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"


class PaymentStatus(enum.Enum):
    """Payment status enumeration."""
    PENDING = "pending"
    PARTIAL = "partial"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class OrderType(enum.Enum):
    """Order type enumeration."""
    REGULAR = "regular"
    EMERGENCY = "emergency"
    SAMPLE = "sample"
    RETURN = "return"
    EXCHANGE = "exchange"


class Sales(BaseModel):
    """
    Sales model representing sales transactions and orders.
    Based on SFA_Orders.xlsx dataset structure.
    """
    __tablename__ = 'sales'
    
    # Basic Order Information (from SFA_Orders)
    code = Column(String(50), unique=True, nullable=False, index=True)  # Code from dataset
    date = Column(DateTime(timezone=True), nullable=False, index=True)  # Date from dataset
    distributor_code = Column(String(50), nullable=False, index=True)  # DistributorCode from dataset
    user_code = Column(String(50), nullable=False, index=True)  # UserCode from dataset
    user_name = Column(String(255), nullable=False, index=True)  # UserName from dataset
    final_value = Column(Float, nullable=False, default=0.0)  # FinalValue from dataset
    creation_date = Column(DateTime(timezone=True), nullable=False)  # CreationDate from dataset
    submitted_date = Column(DateTime(timezone=True), nullable=True)  # SubmittedDate from dataset
    erp_order_number = Column(String(100), nullable=True, index=True)  # ERPOrderNumber from dataset
    
    # Extended Order Information
    order_type = Column(Enum(OrderType), default=OrderType.REGULAR, nullable=False)
    status = Column(Enum(OrderStatus), default=OrderStatus.DRAFT, nullable=False)
    payment_status = Column(Enum(PaymentStatus), default=PaymentStatus.PENDING, nullable=False)
    
    # Foreign Keys
    dealer_id = Column(Integer, ForeignKey('dealers.id'), nullable=False, index=True)
    customer_id = Column(Integer, ForeignKey('customers.id'), nullable=False, index=True)
    
    # Financial Information
    subtotal = Column(Float, default=0.0, nullable=False)
    tax_amount = Column(Float, default=0.0, nullable=False)
    discount_amount = Column(Float, default=0.0, nullable=False)
    shipping_cost = Column(Float, default=0.0, nullable=False)
    total_amount = Column(Float, default=0.0, nullable=False)  # Same as final_value
    
    # Payment Information
    payment_terms = Column(String(50), nullable=True)  # net_30, cod, advance, etc.
    payment_due_date = Column(DateTime(timezone=True), nullable=True)
    paid_amount = Column(Float, default=0.0, nullable=False)
    outstanding_amount = Column(Float, default=0.0, nullable=False)
    
    # Location Information (from SFA_PO)
    order_latitude = Column(Float, nullable=True, index=True)  # Latitude from SFA_PO
    order_longitude = Column(Float, nullable=True, index=True)  # Longitude from SFA_PO
    location_accuracy = Column(Float, nullable=True)
    
    # Delivery Information
    delivery_address = Column(Text, nullable=True)
    delivery_date = Column(DateTime(timezone=True), nullable=True)
    delivery_time_slot = Column(String(50), nullable=True)
    delivery_instructions = Column(Text, nullable=True)
    delivered_date = Column(DateTime(timezone=True), nullable=True)
    delivered_by = Column(String(255), nullable=True)
    
    # Order Processing
    approved_by = Column(String(255), nullable=True)
    approval_date = Column(DateTime(timezone=True), nullable=True)
    processed_by = Column(String(255), nullable=True)
    processing_date = Column(DateTime(timezone=True), nullable=True)
    
    # Additional Information
    reference_number = Column(String(100), nullable=True)
    customer_po_number = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    internal_notes = Column(Text, nullable=True)
    
    # Tracking
    tracking_number = Column(String(100), nullable=True)
    carrier = Column(String(100), nullable=True)
    
    # Audit Fields
    cancelled_by = Column(String(255), nullable=True)
    cancellation_date = Column(DateTime(timezone=True), nullable=True)
    cancellation_reason = Column(Text, nullable=True)
    
    # Relationships
    dealer = relationship("Dealer", back_populates="orders")
    customer = relationship("Customer", back_populates="orders")
    order_items = relationship("SalesItem", back_populates="order", cascade="all, delete-orphan")
    payments = relationship("Payment", back_populates="order", lazy="dynamic")
    
    @hybrid_property
    def is_paid(self):
        """Check if order is fully paid."""
        return self.payment_status == PaymentStatus.PAID
    
    @hybrid_property
    def is_overdue(self):
        """Check if payment is overdue."""
        if self.payment_due_date and self.payment_status != PaymentStatus.PAID:
            return datetime.utcnow() > self.payment_due_date
        return False
    
    @hybrid_property
    def days_overdue(self):
        """Calculate days overdue."""
        if self.is_overdue:
            return (datetime.utcnow() - self.payment_due_date).days
        return 0
    
    @hybrid_property
    def total_quantity(self):
        """Calculate total quantity of items."""
        return sum([item.quantity for item in self.order_items])
    
    @hybrid_property
    def total_items(self):
        """Get total number of different items."""
        return len(self.order_items)
    
    @hybrid_property
    def profit_margin(self):
        """Calculate profit margin percentage."""
        total_cost = sum([item.cost_price * item.quantity for item in self.order_items])
        if total_cost > 0:
            return ((self.total_amount - total_cost) / self.total_amount) * 100
        return 0
    
    @hybrid_property
    def is_location_captured(self):
        """Check if order location is captured."""
        return self.order_latitude is not None and self.order_longitude is not None
    
    def update_location(self, latitude: float, longitude: float, accuracy: float = None):
        """Update order location."""
        self.order_latitude = latitude
        self.order_longitude = longitude
        self.location_accuracy = accuracy
    
    def calculate_totals(self):
        """Calculate order totals from items."""
        self.subtotal = sum([item.line_total for item in self.order_items])
        self.total_amount = self.subtotal + self.tax_amount + self.shipping_cost - self.discount_amount
        self.final_value = self.total_amount
        self.outstanding_amount = self.total_amount - self.paid_amount
    
    def add_item(self, product_code: str, product_name: str, quantity: int, 
                 unit_price: float, cost_price: float = None, discount: float = 0):
        """Add item to order."""
        item = SalesItem(
            product_code=product_code,
            product_name=product_name,
            quantity=quantity,
            unit_price=unit_price,
            cost_price=cost_price or unit_price * 0.7,  # Default cost price
            discount_amount=discount
        )
        self.order_items.append(item)
        self.calculate_totals()
    
    def submit_order(self):
        """Submit order for processing."""
        if self.status == OrderStatus.DRAFT:
            self.status = OrderStatus.SUBMITTED
            self.submitted_date = datetime.utcnow()
            self.calculate_totals()
    
    def confirm_order(self, confirmed_by: str = None):
        """Confirm order."""
        if self.status == OrderStatus.SUBMITTED:
            self.status = OrderStatus.CONFIRMED
            self.approved_by = confirmed_by
            self.approval_date = datetime.utcnow()
    
    def cancel_order(self, reason: str = None, cancelled_by: str = None):
        """Cancel order."""
        if self.status not in [OrderStatus.DELIVERED, OrderStatus.CANCELLED]:
            self.status = OrderStatus.CANCELLED
            self.payment_status = PaymentStatus.CANCELLED
            self.cancellation_reason = reason
            self.cancelled_by = cancelled_by
            self.cancellation_date = datetime.utcnow()
    
    def add_payment(self, amount: float, payment_method: str, reference: str = None):
        """Add payment to order."""
        self.paid_amount += amount
        if self.paid_amount >= self.total_amount:
            self.payment_status = PaymentStatus.PAID
            self.outstanding_amount = 0
        elif self.paid_amount > 0:
            self.payment_status = PaymentStatus.PARTIAL
            self.outstanding_amount = self.total_amount - self.paid_amount
    
    def mark_delivered(self, delivered_by: str = None, delivery_date: datetime = None):
        """Mark order as delivered."""
        self.status = OrderStatus.DELIVERED
        self.delivered_date = delivery_date or datetime.utcnow()
        self.delivered_by = delivered_by
    
    def calculate_distance_from_customer(self) -> float:
        """Calculate distance from customer location."""
        if self.is_location_captured and self.customer.latitude and self.customer.longitude:
            import math
            
            # Convert to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [
                self.order_latitude, self.order_longitude,
                self.customer.latitude, self.customer.longitude
            ])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            # Radius of earth in kilometers
            r = 6371
            return c * r
        return None
    
    def get_order_summary(self) -> Dict[str, Any]:
        """Get order summary."""
        return {
            'code': self.code,
            'total_amount': self.total_amount,
            'total_items': self.total_items,
            'total_quantity': self.total_quantity,
            'status': self.status.value,
            'payment_status': self.payment_status.value,
            'is_overdue': self.is_overdue,
            'days_overdue': self.days_overdue,
            'profit_margin': self.profit_margin
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sales order to dictionary."""
        return {
            'id': self.id,
            'code': self.code,
            'date': self.date.isoformat() if self.date else None,
            'distributor_code': self.distributor_code,
            'user_code': self.user_code,
            'user_name': self.user_name,
            'dealer_id': self.dealer_id,
            'customer_id': self.customer_id,
            'order_type': self.order_type.value if self.order_type else None,
            'status': self.status.value if self.status else None,
            'payment_status': self.payment_status.value if self.payment_status else None,
            'final_value': self.final_value,
            'total_amount': self.total_amount,
            'paid_amount': self.paid_amount,
            'outstanding_amount': self.outstanding_amount,
            'location': {
                'latitude': self.order_latitude,
                'longitude': self.order_longitude,
                'accuracy': self.location_accuracy
            },
            'dates': {
                'creation_date': self.creation_date.isoformat() if self.creation_date else None,
                'submitted_date': self.submitted_date.isoformat() if self.submitted_date else None,
                'delivery_date': self.delivery_date.isoformat() if self.delivery_date else None,
                'delivered_date': self.delivered_date.isoformat() if self.delivered_date else None
            },
            'items': [item.to_dict() for item in self.order_items],
            'summary': self.get_order_summary(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    def __repr__(self):
        return f"<Sales(id={self.id}, code='{self.code}', amount={self.total_amount}, status='{self.status.value if self.status else None}')>"


class SalesItem(BaseModel):
    """Sales order item model."""
    __tablename__ = 'sales_items'
    
    order_id = Column(Integer, ForeignKey('sales.id'), nullable=False, index=True)
    product_code = Column(String(50), nullable=False, index=True)
    product_name = Column(String(255), nullable=False)
    product_category = Column(String(100), nullable=True)
    
    quantity = Column(Integer, nullable=False, default=1)
    unit_price = Column(Float, nullable=False, default=0.0)
    cost_price = Column(Float, nullable=True, default=0.0)
    discount_amount = Column(Float, default=0.0, nullable=False)
    discount_percentage = Column(Float, default=0.0, nullable=False)
    
    line_total = Column(Float, nullable=False, default=0.0)
    
    # Product specifications
    unit_of_measure = Column(String(20), nullable=True)
    weight = Column(Float, nullable=True)
    dimensions = Column(String(100), nullable=True)
    
    # Additional information
    notes = Column(Text, nullable=True)
    
    # Relationships
    order = relationship("Sales", back_populates="order_items")
    
    @hybrid_property
    def profit_amount(self):
        """Calculate profit amount for this item."""
        return (self.unit_price - self.cost_price) * self.quantity
    
    @hybrid_property
    def profit_percentage(self):
        """Calculate profit percentage for this item."""
        if self.cost_price > 0:
            return ((self.unit_price - self.cost_price) / self.cost_price) * 100
        return 0
    
    def calculate_line_total(self):
        """Calculate line total."""
        subtotal = self.quantity * self.unit_price
        self.line_total = subtotal - self.discount_amount
    
    def apply_discount(self, discount_percentage: float = None, discount_amount: float = None):
        """Apply discount to item."""
        if discount_percentage:
            self.discount_percentage = discount_percentage
            self.discount_amount = (self.quantity * self.unit_price) * (discount_percentage / 100)
        elif discount_amount:
            self.discount_amount = discount_amount
            self.discount_percentage = (discount_amount / (self.quantity * self.unit_price)) * 100
        
        self.calculate_line_total()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sales item to dictionary."""
        return {
            'id': self.id,
            'product_code': self.product_code,
            'product_name': self.product_name,
            'product_category': self.product_category,
            'quantity': self.quantity,
            'unit_price': self.unit_price,
            'cost_price': self.cost_price,
            'discount_amount': self.discount_amount,
            'discount_percentage': self.discount_percentage,
            'line_total': self.line_total,
            'profit_amount': self.profit_amount,
            'profit_percentage': self.profit_percentage,
            'unit_of_measure': self.unit_of_measure,
            'notes': self.notes
        }

    def __repr__(self):
        return f"<SalesItem(id={self.id}, product='{self.product_code}', quantity={self.quantity}, total={self.line_total})>"


class Payment(BaseModel):
    """Payment model for tracking order payments."""
    __tablename__ = 'payments'
    
    order_id = Column(Integer, ForeignKey('sales.id'), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    payment_method = Column(String(50), nullable=False)  # cash, card, bank_transfer, cheque
    reference_number = Column(String(100), nullable=True)
    payment_date = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Bank details for bank transfers
    bank_name = Column(String(100), nullable=True)
    account_number = Column(String(50), nullable=True)
    
    # Cheque details
    cheque_number = Column(String(50), nullable=True)
    cheque_date = Column(DateTime(timezone=True), nullable=True)
    
    notes = Column(Text, nullable=True)
    
    # Relationships
    order = relationship("Sales", back_populates="payments")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert payment to dictionary."""
        return {
            'id': self.id,
            'order_id': self.order_id,
            'amount': self.amount,
            'payment_method': self.payment_method,
            'reference_number': self.reference_number,
            'payment_date': self.payment_date.isoformat() if self.payment_date else None,
            'bank_name': self.bank_name,
            'cheque_number': self.cheque_number,
            'notes': self.notes
        }

    def __repr__(self):
        return f"<Payment(id={self.id}, order_id={self.order_id}, amount={self.amount}, method='{self.payment_method}')>"


# Database indexes for performance optimization
Index('idx_sales_date_status', Sales.date, Sales.status)
Index('idx_sales_dealer_customer', Sales.dealer_id, Sales.customer_id)
Index('idx_sales_location', Sales.order_latitude, Sales.order_longitude)
Index('idx_sales_payment_status', Sales.payment_status, Sales.payment_due_date)
Index('idx_sales_erp_number', Sales.erp_order_number)
Index('idx_sales_user_date', Sales.user_code, Sales.date)
Index('idx_sales_items_product', SalesItem.product_code, SalesItem.order_id)
Index('idx_payments_order_date', Payment.order_id, Payment.payment_date)