from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, text
from models.customer import Customer
from models.base import BaseModel
from .base import CRUDBase, BaseRepository
from schemas.customer import CustomerCreate, CustomerUpdate
import math


class CustomerRepository(CRUDBase[Customer, CustomerCreate, CustomerUpdate]):
    def __init__(self):
        super().__init__(Customer)

    def get_by_city(self, db: Session, city: str, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """Get customers by city with pagination"""
        query = db.query(self.model).filter(self.model.city.ilike(f"%{city}%"))
        total = query.count()
        items = query.offset(skip).limit(limit).all()
        
        return {
            "items": items,
            "total": total,
            "page": (skip // limit) + 1,
            "pages": math.ceil(total / limit),
            "per_page": limit
        }

    def get_by_territory(self, db: Session, territory_code: str) -> List[Customer]:
        """Get customers by territory code"""
        return db.query(self.model).filter(self.model.territory_code == territory_code).all()

    def get_by_contact_pattern(self, db: Session, contact_pattern: str) -> List[Customer]:
        """Get customers by contact pattern matching"""
        return db.query(self.model).filter(self.model.contact.like(f"%{contact_pattern}%")).all()

    def search_customers(
        self,
        db: Session,
        *,
        search_term: str = None,
        city: str = None,
        territory_code: str = None,
        skip: int = 0,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Advanced customer search with multiple filters"""
        query = db.query(self.model)
        
        # Apply filters
        if search_term:
            search_conditions = [
                self.model.no.ilike(f"%{search_term}%"),
                self.model.city.ilike(f"%{search_term}%"),
                self.model.contact.ilike(f"%{search_term}%"),
                self.model.ship_to_code.ilike(f"%{search_term}%")
            ]
            query = query.filter(or_(*search_conditions))
        
        if city:
            query = query.filter(self.model.city.ilike(f"%{city}%"))
        
        if territory_code:
            query = query.filter(self.model.territory_code == territory_code)
        
        total = query.count()
        items = query.offset(skip).limit(limit).all()
        
        return {
            "items": items,
            "total": total,
            "page": (skip // limit) + 1,
            "pages": math.ceil(total / limit),
            "per_page": limit
        }

    def get_customers_by_location_radius(
        self,
        db: Session,
        latitude: float,
        longitude: float,
        radius_km: float = 10.0
    ) -> List[Customer]:
        """
        Get customers within a specific radius from given coordinates
        Uses Haversine formula for distance calculation
        """
        # Haversine formula in SQL
        distance_query = func.acos(
            func.cos(func.radians(latitude)) *
            func.cos(func.radians(self.model.latitude)) *
            func.cos(func.radians(self.model.longitude) - func.radians(longitude)) +
            func.sin(func.radians(latitude)) *
            func.sin(func.radians(self.model.latitude))
        ) * 6371  # Earth radius in km
        
        return (
            db.query(self.model)
            .filter(distance_query <= radius_km)
            .filter(and_(
                self.model.latitude.isnot(None),
                self.model.longitude.isnot(None)
            ))
            .all()
        )

    def get_customers_with_coordinates(self, db: Session) -> List[Customer]:
        """Get customers that have latitude and longitude data"""
        return (
            db.query(self.model)
            .filter(and_(
                self.model.latitude.isnot(None),
                self.model.longitude.isnot(None)
            ))
            .all()
        )

    def get_customers_without_coordinates(self, db: Session) -> List[Customer]:
        """Get customers that don't have coordinate data"""
        return (
            db.query(self.model)
            .filter(or_(
                self.model.latitude.is_(None),
                self.model.longitude.is_(None)
            ))
            .all()
        )

    def get_customer_statistics_by_city(self, db: Session) -> List[Dict[str, Any]]:
        """Get customer count statistics grouped by city"""
        results = (
            db.query(
                self.model.city,
                func.count(self.model.id).label('customer_count')
            )
            .group_by(self.model.city)
            .order_by(func.count(self.model.id).desc())
            .all()
        )
        
        return [
            {"city": result.city, "customer_count": result.customer_count}
            for result in results
        ]

    def get_customer_statistics_by_territory(self, db: Session) -> List[Dict[str, Any]]:
        """Get customer count statistics grouped by territory"""
        results = (
            db.query(
                self.model.territory_code,
                func.count(self.model.id).label('customer_count')
            )
            .group_by(self.model.territory_code)
            .order_by(func.count(self.model.id).desc())
            .all()
        )
        
        return [
            {"territory_code": result.territory_code, "customer_count": result.customer_count}
            for result in results
        ]

    def find_nearest_customers(
        self,
        db: Session,
        latitude: float,
        longitude: float,
        limit: int = 10
    ) -> List[Tuple[Customer, float]]:
        """
        Find nearest customers to given coordinates with distance
        Returns list of tuples (customer, distance_km)
        """
        # Haversine formula for distance calculation
        distance_formula = func.acos(
            func.cos(func.radians(latitude)) *
            func.cos(func.radians(self.model.latitude)) *
            func.cos(func.radians(self.model.longitude) - func.radians(longitude)) +
            func.sin(func.radians(latitude)) *
            func.sin(func.radians(self.model.latitude))
        ) * 6371  # Earth radius in km
        
        results = (
            db.query(self.model, distance_formula.label('distance'))
            .filter(and_(
                self.model.latitude.isnot(None),
                self.model.longitude.isnot(None)
            ))
            .order_by(distance_formula)
            .limit(limit)
            .all()
        )
        
        return [(result[0], float(result[1])) for result in results]

    def get_customers_by_global_dimension(
        self,
        db: Session,
        dimension1_code: str = None,
        dimension2_code: str = None
    ) -> List[Customer]:
        """Get customers by global dimension codes"""
        query = db.query(self.model)
        
        if dimension1_code:
            query = query.filter(self.model.global_dimension_1_code == dimension1_code)
        
        if dimension2_code:
            query = query.filter(self.model.global_dimension_2_code == dimension2_code)
        
        return query.all()

    def bulk_update_coordinates(
        self,
        db: Session,
        coordinate_updates: List[Dict[str, Any]]
    ) -> int:
        """
        Bulk update customer coordinates
        coordinate_updates: [{"customer_id": 1, "latitude": 6.9271, "longitude": 79.8612}, ...]
        """
        updated_count = 0
        
        for update_data in coordinate_updates:
            customer_id = update_data.get("customer_id")
            latitude = update_data.get("latitude")
            longitude = update_data.get("longitude")
            
            if customer_id and latitude is not None and longitude is not None:
                result = (
                    db.query(self.model)
                    .filter(self.model.id == customer_id)
                    .update({
                        "latitude": latitude,
                        "longitude": longitude
                    })
                )
                updated_count += result
        
        db.commit()
        return updated_count

    def get_customer_distribution_report(self, db: Session) -> Dict[str, Any]:
        """Generate comprehensive customer distribution report"""
        total_customers = db.query(self.model).count()
        
        # City distribution
        city_stats = self.get_customer_statistics_by_city(db)
        
        # Territory distribution
        territory_stats = self.get_customer_statistics_by_territory(db)
        
        # Coordinate availability
        with_coordinates = len(self.get_customers_with_coordinates(db))
        without_coordinates = len(self.get_customers_without_coordinates(db))
        
        # Global dimension statistics
        dimension1_stats = (
            db.query(
                self.model.global_dimension_1_code,
                func.count(self.model.id).label('count')
            )
            .group_by(self.model.global_dimension_1_code)
            .all()
        )
        
        dimension2_stats = (
            db.query(
                self.model.global_dimension_2_code,
                func.count(self.model.id).label('count')
            )
            .group_by(self.model.global_dimension_2_code)
            .all()
        )
        
        return {
            "total_customers": total_customers,
            "city_distribution": city_stats,
            "territory_distribution": territory_stats,
            "coordinate_coverage": {
                "with_coordinates": with_coordinates,
                "without_coordinates": without_coordinates,
                "coverage_percentage": (with_coordinates / total_customers * 100) if total_customers > 0 else 0
            },
            "global_dimension_1_distribution": [
                {"code": stat.global_dimension_1_code, "count": stat.count}
                for stat in dimension1_stats
            ],
            "global_dimension_2_distribution": [
                {"code": stat.global_dimension_2_code, "count": stat.count}
                for stat in dimension2_stats
            ]
        }

    def validate_customer_data_quality(self, db: Session) -> Dict[str, Any]:
        """Validate customer data quality and return issues"""
        issues = []
        
        # Check for missing required fields
        missing_city = db.query(self.model).filter(or_(
            self.model.city.is_(None),
            self.model.city == ""
        )).count()
        
        missing_contact = db.query(self.model).filter(or_(
            self.model.contact.is_(None),
            self.model.contact == ""
        )).count()
        
        # Check for duplicate contacts
        duplicate_contacts = (
            db.query(self.model.contact, func.count(self.model.id).label('count'))
            .group_by(self.model.contact)
            .having(func.count(self.model.id) > 1)
            .all()
        )
        
        # Check for invalid coordinates
        invalid_coordinates = db.query(self.model).filter(or_(
            and_(self.model.latitude.isnot(None), 
                 or_(self.model.latitude < -90, self.model.latitude > 90)),
            and_(self.model.longitude.isnot(None),
                 or_(self.model.longitude < -180, self.model.longitude > 180))
        )).count()
        
        return {
            "total_customers": db.query(self.model).count(),
            "data_quality_issues": {
                "missing_city": missing_city,
                "missing_contact": missing_contact,
                "duplicate_contacts": len(duplicate_contacts),
                "invalid_coordinates": invalid_coordinates
            },
            "duplicate_contact_details": [
                {"contact": dup.contact, "count": dup.count}
                for dup in duplicate_contacts
            ]
        }


# Create instance for dependency injection
def get_customer_repository() -> CustomerRepository:
    return CustomerRepository()