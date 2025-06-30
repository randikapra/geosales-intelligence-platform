from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, text
from datetime import datetime, date, timedelta
from decimal import Decimal
import math

from models.customer import Customer
from models.sales import SalesOrder
from schemas.customer import (
    CustomerCreate, CustomerUpdate, CustomerSearchFilter, 
    CustomerWithSales, CustomerLocationResponse, CustomerStats
)
from utils.geo_utils import calculate_distance, get_nearby_locations
from repositories.customer_repo import CustomerRepository
from repositories.sales_repo import SalesRepository

class CustomerService:
    def __init__(self):
        self.customer_repo = CustomerRepository()
        self.sales_repo = SalesRepository()
    
    def create_customer(self, db: Session, customer_data: CustomerCreate) -> Customer:
        """Create a new customer"""
        # Generate customer number
        last_customer = db.query(Customer).order_by(Customer.id.desc()).first()
        if last_customer:
            last_no = int(last_customer.no) if last_customer.no.isdigit() else 100000
            new_no = str(last_no + 1)
        else:
            new_no = "100001"
        
        db_customer = Customer(
            no=new_no,
            **customer_data.dict()
        )
        
        return self.customer_repo.create(db, db_customer)
    
    def get_customer(self, db: Session, customer_id: int) -> Optional[Customer]:
        """Get customer by ID"""
        return self.customer_repo.get(db, customer_id)
    
    def get_customer_by_no(self, db: Session, customer_no: str) -> Optional[Customer]:
        """Get customer by customer number"""
        return db.query(Customer).filter(Customer.no == customer_no).first()
    
    def get_customers(
        self, 
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[CustomerSearchFilter] = None
    ) -> List[Customer]:
        """Get customers with optional filtering"""
        query = db.query(Customer)
        
        if filters:
            if filters.city:
                query = query.filter(Customer.city.ilike(f"%{filters.city}%"))
            
            if filters.territory_code:
                query = query.filter(Customer.territory_code == filters.territory_code)
            
            if filters.has_coordinates is not None:
                if filters.has_coordinates:
                    query = query.filter(
                        and_(Customer.latitude.isnot(None), Customer.longitude.isnot(None))
                    )
                else:
                    query = query.filter(
                        or_(Customer.latitude.is_(None), Customer.longitude.is_(None))
                    )
        
        return query.offset(skip).limit(limit).all()
    
    def update_customer(self, db: Session, customer_id: int, customer_data: CustomerUpdate) -> Optional[Customer]:
        """Update customer"""
        customer = self.get_customer(db, customer_id)
        if not customer:
            return None
        
        update_data = customer_data.dict(exclude_unset=True)
        return self.customer_repo.update(db, customer, update_data)
    
    def delete_customer(self, db: Session, customer_id: int) -> bool:
        """Delete customer"""
        customer = self.get_customer(db, customer_id)
        if not customer:
            return False
        
        return self.customer_repo.delete(db, customer)
    
    def get_customers_near_location(
        self, 
        db: Session, 
        latitude: float, 
        longitude: float, 
        radius_km: float = 10.0,
        limit: int = 50
    ) -> List[CustomerLocationResponse]:
        """Get customers within specified radius of a location"""
        # First get all customers with coordinates
        customers_with_coords = db.query(Customer).filter(
            and_(Customer.latitude.isnot(None), Customer.longitude.isnot(None))
        ).all()
        
        nearby_customers = []
        for customer in customers_with_coords:
            distance = calculate_distance(
                latitude, longitude, 
                customer.latitude, customer.longitude
            )
            
            if distance <= radius_km:
                nearby_customers.append(
                    CustomerLocationResponse(
                        id=customer.id,
                        no=customer.no,
                        city=customer.city,
                        latitude=customer.latitude,
                        longitude=customer.longitude,
                        distance_km=round(distance, 2)
                    )
                )
        
        # Sort by distance and limit results
        nearby_customers.sort(key=lambda x: x.distance_km)
        return nearby_customers[:limit]
    
    def get_customer_with_sales_data(self, db: Session, customer_id: int) -> Optional[CustomerWithSales]:
        """Get customer with aggregated sales data"""
        customer = self.get_customer(db, customer_id)
        if not customer:
            return None
        
        # Get sales statistics
        sales_stats = self.sales_repo.get_customer_sales_stats(db, customer.no)
        
        return CustomerWithSales(
            **customer.__dict__,
            total_sales=sales_stats.get('total_sales', 0),
            order_count=sales_stats.get('order_count', 0),
            last_order_date=sales_stats.get('last_order_date'),
            avg_order_value=sales_stats.get('avg_order_value', 0)
        )
    
    def get_customer_sales_history(
        self, 
        db: Session, 
        customer_id: int, 
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[SalesOrder]:
        """Get customer's sales history"""
        customer = self.get_customer(db, customer_id)
        if not customer:
            return []
        
        return self.sales_repo.get_orders_by_customer(
            db, customer.no, start_date, end_date
        )
    
    def get_customer_stats(self, db: Session) -> CustomerStats:
        """Get overall customer statistics"""
        total_customers = db.query(func.count(Customer.id)).scalar()
        
        customers_with_coords = db.query(func.count(Customer.id)).filter(
            and_(Customer.latitude.isnot(None), Customer.longitude.isnot(None))
        ).scalar()
        
        # Customers by city
        city_stats = db.query(
            Customer.city, func.count(Customer.id)
        ).group_by(Customer.city).all()
        customers_by_city = {city: count for city, count in city_stats}
        
        # Customers by territory
        territory_stats = db.query(
            Customer.territory_code, func.count(Customer.id)
        ).filter(Customer.territory_code.isnot(None)).group_by(Customer.territory_code).all()
        customers_by_territory = {territory: count for territory, count in territory_stats}
        
        # Geographic coverage (cities with customers)
        unique_cities = db.query(func.count(func.distinct(Customer.city))).scalar()
        
        return CustomerStats(
            total_customers=total_customers,
            customers_with_coordinates=customers_with_coords,
            customers_by_city=customers_by_city,
            customers_by_territory=customers_by_territory,
            geographic_coverage={"unique_cities": unique_cities}
        )
    
    def search_customers(
        self, 
        db: Session, 
        search_term: str, 
        limit: int = 20
    ) -> List[Customer]:
        """Search customers by name, city, or customer number"""
        search_pattern = f"%{search_term}%"
        
        return db.query(Customer).filter(
            or_(
                Customer.no.ilike(search_pattern),
                Customer.city.ilike(search_pattern),
                Customer.contact.ilike(search_pattern)
            )
        ).limit(limit).all()
    
    def get_customer_segments(self, db: Session) -> Dict[str, List[Customer]]:
        """Segment customers based on sales performance"""
        # Get all customers with sales data
        customers_with_sales = []
        customers = db.query(Customer).all()
        
        for customer in customers:
            sales_stats = self.sales_repo.get_customer_sales_stats(db, customer.no)
            if sales_stats.get('total_sales', 0) > 0:
                customers_with_sales.append({
                    'customer': customer,
                    'total_sales': sales_stats['total_sales'],
                    'order_count': sales_stats['order_count']
                })
        
        # Sort by total sales
        customers_with_sales.sort(key=lambda x: x['total_sales'], reverse=True)
        
        # Segment into quartiles
        total_count = len(customers_with_sales)
        quartile_size = total_count // 4
        
        segments = {
            'high_value': [c['customer'] for c in customers_with_sales[:quartile_size]],
            'medium_high': [c['customer'] for c in customers_with_sales[quartile_size:2*quartile_size]],
            'medium_low': [c['customer'] for c in customers_with_sales[2*quartile_size:3*quartile_size]],
            'low_value': [c['customer'] for c in customers_with_sales[3*quartile_size:]],
            'no_sales': [c for c in customers if not any(cs['customer'].id == c.id for cs in customers_with_sales)]
        }
        
        return segments
    
    def get_customers_by_territory(self, db: Session, territory_code: str) -> List[Customer]:
        """Get all customers in a specific territory"""
        return db.query(Customer).filter(Customer.territory_code == territory_code).all()
    
    def bulk_update_coordinates(self, db: Session, coordinate_updates: List[Dict[str, Any]]) -> int:
        """Bulk update customer coordinates"""
        updated_count = 0
        
        for update in coordinate_updates:
            customer = self.get_customer_by_no(db, update['customer_no'])
            if customer:
                customer.latitude = update.get('latitude')
                customer.longitude = update.get('longitude')
                updated_count += 1
        
        db.commit()
        return updated_count
    
    def get_customer_coverage_analysis(self, db: Session) -> Dict[str, Any]:
        """Analyze customer geographic coverage"""
        customers_with_coords = db.query(Customer).filter(
            and_(Customer.latitude.isnot(None), Customer.longitude.isnot(None))
        ).all()
        
        if not customers_with_coords:
            return {
                'total_customers': 0,
                'coverage_area': 0,
                'center_point': None,
                'spread_analysis': None
            }
        
        # Calculate center point
        avg_lat = sum(c.latitude for c in customers_with_coords) / len(customers_with_coords)
        avg_lng = sum(c.longitude for c in customers_with_coords) / len(customers_with_coords)
        
        # Calculate spread (standard deviation)
        lat_variance = sum((c.latitude - avg_lat) ** 2 for c in customers_with_coords) / len(customers_with_coords)
        lng_variance = sum((c.longitude - avg_lng) ** 2 for c in customers_with_coords) / len(customers_with_coords)
        
        # Estimate coverage area (rough approximation)
        max_distance = 0
        for customer in customers_with_coords:
            distance = calculate_distance(avg_lat, avg_lng, customer.latitude, customer.longitude)
            max_distance = max(max_distance, distance)
        
        coverage_area = math.pi * (max_distance ** 2)  # Rough circular area
        
        return {
            'total_customers': len(customers_with_coords),
            'coverage_area_km2': round(coverage_area, 2),
            'center_point': {'latitude': avg_lat, 'longitude': avg_lng},
            'spread_analysis': {
                'max_distance_from_center': round(max_distance, 2),
                'latitude_std': round(math.sqrt(lat_variance), 4),
                'longitude_std': round(math.sqrt(lng_variance), 4)
            }
        }
    
    def get_customer_activity_timeline(
        self, 
        db: Session, 
        customer_id: int, 
        days: int = 365
    ) -> List[Dict[str, Any]]:
        """Get customer activity timeline"""
        customer = self.get_customer(db, customer_id)
        if not customer:
            return []
        
        start_date = datetime.now() - timedelta(days=days)
        
        # Get sales orders
        orders = self.sales_repo.get_orders_by_customer(
            db, customer.no, start_date.date(), datetime.now().date()
        )
        
        timeline = []
        for order in orders:
            timeline.append({
                'date': order.date,
                'type': 'order',
                'description': f"Order {order.code} - ${order.final_value}",
                'value': float(order.final_value),
                'details': {
                    'order_id': order.id,
                    'dealer': order.user_name,
                    'status': order.status if hasattr(order, 'status') else 'completed'
                }
            })
        
        # Sort by date
        timeline.sort(key=lambda x: x['date'], reverse=True)
        
        return timeline

# Create singleton instance
customer_service = CustomerService()