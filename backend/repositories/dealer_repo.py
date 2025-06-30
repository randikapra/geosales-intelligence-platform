from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, text, desc, asc
from datetime import datetime, timedelta
from models.dealer import Dealer
from models.sales import Sales
from models.gps_data import GPSData
from .base import CRUDBase, BaseRepository
from schemas.dealer import DealerCreate, DealerUpdate
import math


class DealerRepository(CRUDBase[Dealer, DealerCreate, DealerUpdate]):
    def __init__(self):
        super().__init__(Dealer)

    def get_by_user_code(self, db: Session, user_code: str) -> Optional[Dealer]:
        """Get dealer by user code"""
        return db.query(self.model).filter(self.model.user_code == user_code).first()

    def get_by_division(self, db: Session, division_code: str) -> List[Dealer]:
        """Get dealers by division code"""
        return db.query(self.model).filter(self.model.division_code == division_code).all()

    def get_by_territory(self, db: Session, territory_code: str) -> List[Dealer]:
        """Get dealers by territory code"""
        return db.query(self.model).filter(self.model.territory_code == territory_code).all()

    def search_dealers(
        self,
        db: Session,
        *,
        search_term: str = None,
        division_code: str = None,
        territory_code: str = None,
        active_only: bool = True,
        skip: int = 0,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Advanced dealer search with multiple filters"""
        query = db.query(self.model)
        
        # Apply filters
        if search_term:
            search_conditions = [
                self.model.user_code.ilike(f"%{search_term}%"),
                self.model.user_name.ilike(f"%{search_term}%"),
                self.model.division_code.ilike(f"%{search_term}%"),
                self.model.territory_code.ilike(f"%{search_term}%")
            ]
            query = query.filter(or_(*search_conditions))
        
        if division_code:
            query = query.filter(self.model.division_code == division_code)
        
        if territory_code:
            query = query.filter(self.model.territory_code == territory_code)
        
        if active_only:
            query = query.filter(self.model.is_active == True)
        
        total = query.count()
        items = query.offset(skip).limit(limit).all()
        
        return {
            "items": items,
            "total": total,
            "page": (skip // limit) + 1,
            "pages": math.ceil(total / limit),
            "per_page": limit
        }

    def get_dealer_performance_metrics(
        self,
        db: Session,
        dealer_id: int,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get comprehensive performance metrics for a dealer"""
        
        # Sales performance
        sales_query = (
            db.query(
                func.count(Sales.id).label('total_orders'),
                func.sum(Sales.final_value).label('total_sales'),
                func.avg(Sales.final_value).label('avg_order_value'),
                func.min(Sales.final_value).label('min_order'),
                func.max(Sales.final_value).label('max_order')
            )
            .filter(and_(
                Sales.user_code == self.get(db, dealer_id).user_code,
                Sales.date >= start_date,
                Sales.date <= end_date
            ))
            .first()
        )
        
        # Customer reach (unique customers served)
        unique_customers = (
            db.query(func.count(func.distinct(Sales.distributor_code)))
            .filter(and_(
                Sales.user_code == self.get(db, dealer_id).user_code,
                Sales.date >= start_date,
                Sales.date <= end_date
            ))
            .scalar()
        )
        
        # GPS tracking metrics
        gps_metrics = self.get_dealer_gps_metrics(db, dealer_id, start_date, end_date)
        
        # Daily sales trend
        daily_sales = (
            db.query(
                func.date(Sales.date).label('date'),
                func.sum(Sales.final_value).label('daily_total'),
                func.count(Sales.id).label('daily_orders')
            )
            .filter(and_(
                Sales.user_code == self.get(db, dealer_id).user_code,
                Sales.date >= start_date,
                Sales.date <= end_date
            ))
            .group_by(func.date(Sales.date))
            .order_by(func.date(Sales.date))
            .all()
        )
        
        return {
            "dealer_id": dealer_id,
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "sales_metrics": {
                "total_orders": sales_query.total_orders or 0,
                "total_sales": float(sales_query.total_sales or 0),
                "avg_order_value": float(sales_query.avg_order_value or 0),
                "min_order": float(sales_query.min_order or 0),
                "max_order": float(sales_query.max_order or 0),
                "unique_customers": unique_customers or 0
            },
            "gps_metrics": gps_metrics,
            "daily_sales_trend": [
                {
                    "date": day.date.isoformat(),
                    "total_sales": float(day.daily_total),
                    "order_count": day.daily_orders
                }
                for day in daily_sales
            ]
        }

    def get_dealer_gps_metrics(
        self,
        db: Session,
        dealer_id: int,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get GPS tracking metrics for a dealer"""
        dealer = self.get(db, dealer_id)
        if not dealer:
            return {}
        
        # Total GPS points recorded
        total_gps_points = (
            db.query(func.count(GPSData.id))
            .filter(and_(
                GPSData.user_code == dealer.user_code,
                GPSData.received_date >= start_date,
                GPSData.received_date <= end_date
            ))
            .scalar()
        )
        
        # Unique tour codes (working days)
        unique_tours = (
            db.query(func.count(func.distinct(GPSData.tour_code)))
            .filter(and_(
                GPSData.user_code == dealer.user_code,
                GPSData.received_date >= start_date,
                GPSData.received_date <= end_date
            ))
            .scalar()
        )
        
        # Calculate total distance traveled (simplified calculation)
        gps_points = (
            db.query(GPSData.latitude, GPSData.longitude, GPSData.received_date)
            .filter(and_(
                GPSData.user_code == dealer.user_code,
                GPSData.received_date >= start_date,
                GPSData.received_date <= end_date
            ))
            .order_by(GPSData.received_date)
            .all()
        )
        
        total_distance = self._calculate_total_distance(gps_points)
        
        # Working hours calculation (first to last GPS point per day)
        daily_hours = (
            db.query(
                func.date(GPSData.received_date).label('date'),
                func.min(GPSData.received_date).label('first_point'),
                func.max(GPSData.received_date).label('last_point')
            )
            .filter(and_(
                GPSData.user_code == dealer.user_code,
                GPSData.received_date >= start_date,
                GPSData.received_date <= end_date
            ))
            .group_by(func.date(GPSData.received_date))
            .all()
        )
        
        total_working_hours = sum([
            (day.last_point - day.first_point).total_seconds() / 3600
            for day in daily_hours
        ])
        
        return {
            "total_gps_points": total_gps_points or 0,
            "unique_tours": unique_tours or 0,
            "total_distance_km": round(total_distance, 2),
            "total_working_hours": round(total_working_hours, 2),
            "avg_hours_per_day": round(total_working_hours / max(len(daily_hours), 1), 2)
        }

    def _calculate_total_distance(self, gps_points: List[Any]) -> float:
        """Calculate total distance from GPS points using Haversine formula"""
        if len(gps_points) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(gps_points)):
            prev_point = gps_points[i-1]
            curr_point = gps_points[i]
            
            # Haversine formula
            lat1, lon1 = math.radians(prev_point.latitude), math.radians(prev_point.longitude)
            lat2, lon2 = math.radians(curr_point.latitude), math.radians(curr_point.longitude)
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            # Earth radius in kilometers
            r = 6371
            distance = c * r
            
            # Only add if distance is reasonable (< 50km between points)
            if distance < 50:
                total_distance += distance
        
        return total_distance

    def get_top_performers(
        self,
        db: Session,
        *,
        start_date: datetime,
        end_date: datetime,
        limit: int = 10,
        metric: str = "total_sales"
    ) -> List[Dict[str, Any]]:
        """Get top performing dealers based on specified metric"""
        
        # Join dealers with their sales data
        query = (
            db.query(
                self.model.id,
                self.model.user_code,
                self.model.user_name,
                self.model.division_code,
                self.model.territory_code,
                func.count(Sales.id).label('total_orders'),
                func.sum(Sales.final_value).label('total_sales'),
                func.avg(Sales.final_value).label('avg_order_value'),
                func.count(func.distinct(Sales.distributor_code)).label('unique_customers')
            )
            .join(Sales, self.model.user_code == Sales.user_code)
            .filter(and_(
                Sales.date >= start_date,
                Sales.date <= end_date
            ))
            .group_by(
                self.model.id,
                self.model.user_code,
                self.model.user_name,
                self.model.division_code,
                self.model.territory_code
            )
        )
        
        # Order by specified metric
        if metric == "total_sales":
            query = query.order_by(desc(func.sum(Sales.final_value)))
        elif metric == "total_orders":
            query = query.order_by(desc(func.count(Sales.id)))
        elif metric == "avg_order_value":
            query = query.order_by(desc(func.avg(Sales.final_value)))
        elif metric == "unique_customers":
            query = query.order_by(desc(func.count(func.distinct(Sales.distributor_code))))
        
        results = query.limit(limit).all()
        
        return [
            {
                "dealer_id": result.id,
                "user_code": result.user_code,
                "user_name": result.user_name,
                "division_code": result.division_code,
                "territory_code": result.territory_code,
                "total_orders": result.total_orders,
                "total_sales": float(result.total_sales),
                "avg_order_value": float(result.avg_order_value),
                "unique_customers": result.unique_customers
            }
            for result in results
        ]

    def get_dealers_by_location_radius(
        self,
        db: Session,
        latitude: float,
        longitude: float,
        radius_km: float = 50.0
    ) -> List[Dict[str, Any]]:
        """Get dealers within specified radius from given coordinates"""
        
        # Get recent GPS positions for dealers
        recent_positions = (
            db.query(
                GPSData.user_code,
                GPSData.latitude,
                GPSData.longitude,
                func.max(GPSData.received_date).label('latest_date')
            )
            .group_by(GPSData.user_code, GPSData.latitude, GPSData.longitude)
            .subquery()
        )
        
        # Calculate distance using Haversine formula
        distance_query = func.acos(
            func.cos(func.radians(latitude)) *
            func.cos(func.radians(recent_positions.c.latitude)) *
            func.cos(func.radians(recent_positions.c.longitude) - func.radians(longitude)) +
            func.sin(func.radians(latitude)) *
            func.sin(func.radians(recent_positions.c.latitude))
        ) * 6371
        
        results = (
            db.query(
                self.model,
                recent_positions.c.latitude,
                recent_positions.c.longitude,
                recent_positions.c.latest_date,
                distance_query.label('distance')
            )
            .join(recent_positions, self.model.user_code == recent_positions.c.user_code)
            .filter(distance_query <= radius_km)
            .order_by(distance_query)
            .all()
        )
        
        return [
            {
                "dealer": result[0],
                "last_known_position": {
                    "latitude": float(result[1]),
                    "longitude": float(result[2]),
                    "timestamp": result[3].isoformat()
                },
                "distance_km": round(float(result[4]), 2)
            }
            for result in results
        ]

    def get_dealer_territory_optimization_data(self, db: Session) -> List[Dict[str, Any]]:
        """Get data for territory optimization analysis"""
        
        # Get dealer performance by territory
        territory_performance = (
            db.query(
                self.model.territory_code,
                func.count(self.model.id).label('dealer_count'),
                func.count(Sales.id).label('total_orders'),
                func.sum(Sales.final_value).label('total_sales'),
                func.avg(Sales.final_value).label('avg_order_value')
            )
            .outerjoin(Sales, self.model.user_code == Sales.user_code)
            .group_by(self.model.territory_code)
            .all()
        )
        
        # Get geographic distribution
        territory_locations = (
            db.query(
                self.model.territory_code,
                func.avg(GPSData.latitude).label('avg_latitude'),
                func.avg(GPSData.longitude).label('avg_longitude'),
                func.count(func.distinct(GPSData.user_code)).label('active_dealers')
            )
            .join(GPSData, self.model.user_code == GPSData.user_code)
            .group_by(self.model.territory_code)
            .all()
        )
        
        # Combine data
        territory_data = {}
        for perf in territory_performance:
            territory_data[perf.territory_code] = {
                "territory_code": perf.territory_code,
                "dealer_count": perf.dealer_count,
                "total_orders": perf.total_orders or 0,
                "total_sales": float(perf.total_sales or 0),
                "avg_order_value": float(perf.avg_order_value or 0)
            }
        
        for loc in territory_locations:
            if loc.territory_code in territory_data:
                territory_data[loc.territory_code].update({
                    "avg_latitude": float(loc.avg_latitude) if loc.avg_latitude else None,
                    "avg_longitude": float(loc.avg_longitude) if loc.avg_longitude else None,
                    "active_dealers": loc.active_dealers
                })
        
        return list(territory_data.values())

    def get_dealer_workload_analysis(
        self,
        db: Session,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Analyze dealer workload and capacity"""
        
        results = (
            db.query(
                self.model.id,
                self.model.user_code,
                self.model.user_name,
                self.model.territory_code,
                func.count(Sales.id).label('total_orders'),
                func.sum(Sales.final_value).label('total_sales'),
                func.count(func.distinct(Sales.distributor_code)).label('unique_customers'),
                func.count(func.distinct(func.date(Sales.date))).label('working_days')
            )
            .outerjoin(Sales, and_(
                self.model.user_code == Sales.user_code,
                Sales.date >= start_date,
                Sales.date <= end_date
            ))
            .group_by(
                self.model.id,
                self.model.user_code,
                self.model.user_name,
                self.model.territory_code
            )
            .all()
        )
        
        workload_data = []
        for result in results:
            # Calculate workload metrics
            orders_per_day = result.total_orders / max(result.working_days, 1) if result.working_days else 0
            sales_per_day = float(result.total_sales or 0) / max(result.working_days, 1) if result.working_days else 0
            customers_per_day = result.unique_customers / max(result.working_days, 1) if result.working_days else 0
            
            # Determine workload status
            workload_status = "Low"
            if orders_per_day > 10:
                workload_status = "High"
            elif orders_per_day > 5:
                workload_status = "Medium"
            
            workload_data.append({
                "dealer_id": result.id,
                "user_code": result.user_code,
                "user_name": result.user_name,
                "territory_code": result.territory_code,
                "total_orders": result.total_orders,
                "total_sales": float(result.total_sales or 0),
                "unique_customers": result.unique_customers,
                "working_days": result.working_days,
                "orders_per_day": round(orders_per_day, 2),
                "sales_per_day": round(sales_per_day, 2),
                "customers_per_day": round(customers_per_day, 2),
                "workload_status": workload_status
            })
        
        return workload_data

    def get_inactive_dealers(
        self,
        db: Session,
        days_threshold: int = 30
    ) -> List[Dict[str, Any]]:
        """Get dealers who haven't been active for specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        # Get dealers with their last activity
        last_activity = (
            db.query(
                Sales.user_code,
                func.max(Sales.date).label('last_sale_date')
            )
            .group_by(Sales.user_code)
            .subquery()
        )
        
        last_gps = (
            db.query(
                GPSData.user_code,
                func.max(GPSData.received_date).label('last_gps_date')
            )
            .group_by(GPSData.user_code)
            .subquery()
        )
        
        inactive_dealers = (
            db.query(
                self.model,
                last_activity.c.last_sale_date,
                last_gps.c.last_gps_date
            )
            .outerjoin(last_activity, self.model.user_code == last_activity.c.user_code)
            .outerjoin(last_gps, self.model.user_code == last_gps.c.user_code)
            .filter(or_(
                last_activity.c.last_sale_date < cutoff_date,
                last_activity.c.last_sale_date.is_(None)
            ))
            .all()
        )
        
        return [
            {
                "dealer": result[0],
                "last_sale_date": result[1].isoformat() if result[1] else None,
                "last_gps_date": result[2].isoformat() if result[2] else None,
                "days_since_last_sale": (datetime.now() - result[1]).days if result[1] else None
            }
            for result in inactive_dealers
        ]


# Create instance for dependency injection
def get_dealer_repository() -> DealerRepository:
    return DealerRepository()