from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, text
from datetime import datetime, timedelta
import math
from geopy.distance import geodesic

from models.dealer import Dealer
from models.gps_data import GPSData
from models.sales import Sales
from models.customer import Customer
from repositories.dealer_repo import DealerRepository
from repositories.gps_repo import GPSRepository
from utils.geo_utils import calculate_distance, get_territory_bounds
from utils.date_utils import get_date_range, format_date
from core.exceptions import DealerNotFoundException, TerritoryOptimizationError


class DealerService:
    def __init__(self, db: Session):
        self.db = db
        self.dealer_repo = DealerRepository(db)
        self.gps_repo = GPSRepository(db)

    async def get_dealer_performance_metrics(
        self, 
        dealer_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calculate comprehensive dealer performance metrics"""
        try:
            dealer = await self.dealer_repo.get_by_id(dealer_id)
            if not dealer:
                raise DealerNotFoundException(f"Dealer {dealer_id} not found")

            # Sales metrics
            sales_data = await self._get_dealer_sales_metrics(dealer_id, start_date, end_date)
            
            # GPS tracking metrics
            gps_metrics = await self._get_dealer_gps_metrics(dealer_id, start_date, end_date)
            
            # Customer visit metrics
            visit_metrics = await self._get_dealer_visit_metrics(dealer_id, start_date, end_date)
            
            # Territory coverage metrics
            territory_metrics = await self._get_territory_coverage_metrics(dealer_id, start_date, end_date)

            return {
                "dealer_id": dealer_id,
                "dealer_name": dealer.name,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "sales_metrics": sales_data,
                "gps_metrics": gps_metrics,
                "visit_metrics": visit_metrics,
                "territory_metrics": territory_metrics,
                "overall_score": self._calculate_overall_performance_score(
                    sales_data, gps_metrics, visit_metrics, territory_metrics
                )
            }
        except Exception as e:
            raise Exception(f"Error calculating dealer performance: {str(e)}")

    async def _get_dealer_sales_metrics(
        self, 
        dealer_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calculate sales-related metrics for dealer"""
        query = self.db.query(
            func.count(Sales.id).label('total_orders'),
            func.sum(Sales.final_value).label('total_revenue'),
            func.avg(Sales.final_value).label('avg_order_value'),
            func.count(func.distinct(Sales.distributor_code)).label('unique_customers')
        ).filter(
            and_(
                Sales.user_code == dealer_id,
                Sales.date >= start_date,
                Sales.date <= end_date
            )
        )
        
        result = query.first()
        
        # Calculate daily averages
        days_in_period = (end_date - start_date).days + 1
        
        return {
            "total_orders": result.total_orders or 0,
            "total_revenue": float(result.total_revenue or 0),
            "avg_order_value": float(result.avg_order_value or 0),
            "unique_customers": result.unique_customers or 0,
            "daily_avg_orders": round((result.total_orders or 0) / days_in_period, 2),
            "daily_avg_revenue": round(float(result.total_revenue or 0) / days_in_period, 2)
        }

    async def _get_dealer_gps_metrics(
        self, 
        dealer_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calculate GPS tracking and movement metrics"""
        gps_points = self.db.query(GPSData).filter(
            and_(
                GPSData.user_code == dealer_id,
                GPSData.received_date >= start_date,
                GPSData.received_date <= end_date
            )
        ).order_by(GPSData.received_date).all()

        if not gps_points:
            return {
                "total_distance_km": 0,
                "active_hours": 0,
                "avg_speed_kmh": 0,
                "stops_count": 0,
                "tracking_points": 0
            }

        # Calculate total distance traveled
        total_distance = 0
        stops = []
        previous_point = None
        
        for point in gps_points:
            if previous_point:
                distance = geodesic(
                    (previous_point.latitude, previous_point.longitude),
                    (point.latitude, point.longitude)
                ).kilometers
                total_distance += distance
                
                # Detect stops (same location for more than 10 minutes)
                time_diff = (point.received_date - previous_point.received_date).total_seconds() / 60
                if distance < 0.1 and time_diff > 10:  # Less than 100m movement in 10+ minutes
                    stops.append({
                        "latitude": point.latitude,
                        "longitude": point.longitude,
                        "duration_minutes": time_diff
                    })
            
            previous_point = point

        # Calculate active hours
        if len(gps_points) > 1:
            active_duration = (gps_points[-1].received_date - gps_points[0].received_date).total_seconds() / 3600
            avg_speed = total_distance / active_duration if active_duration > 0 else 0
        else:
            active_duration = 0
            avg_speed = 0

        return {
            "total_distance_km": round(total_distance, 2),
            "active_hours": round(active_duration, 2),
            "avg_speed_kmh": round(avg_speed, 2),
            "stops_count": len(stops),
            "tracking_points": len(gps_points),
            "detailed_stops": stops[:10]  # Return first 10 stops for analysis
        }

    async def _get_dealer_visit_metrics(
        self, 
        dealer_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calculate customer visit metrics"""
        # Get sales with customer locations
        sales_with_customers = self.db.query(
            Sales.distributor_code,
            Sales.date,
            Customer.latitude,
            Customer.longitude,
            Customer.city
        ).join(
            Customer, Sales.distributor_code == Customer.no
        ).filter(
            and_(
                Sales.user_code == dealer_id,
                Sales.date >= start_date,
                Sales.date <= end_date
            )
        ).all()

        # Group visits by customer and date
        customer_visits = {}
        for sale in sales_with_customers:
            customer_id = sale.distributor_code
            visit_date = sale.date.date()
            
            if customer_id not in customer_visits:
                customer_visits[customer_id] = {
                    "visits": set(),
                    "latitude": sale.latitude,
                    "longitude": sale.longitude,
                    "city": sale.city
                }
            customer_visits[customer_id]["visits"].add(visit_date)

        # Calculate metrics
        total_customers_visited = len(customer_visits)
        total_visits = sum(len(visits["visits"]) for visits in customer_visits.values())
        
        # Customer frequency analysis
        visit_frequency = {}
        for customer_data in customer_visits.values():
            freq = len(customer_data["visits"])
            visit_frequency[freq] = visit_frequency.get(freq, 0) + 1

        return {
            "total_customers_visited": total_customers_visited,
            "total_visits": total_visits,
            "avg_visits_per_customer": round(total_visits / total_customers_visited, 2) if total_customers_visited > 0 else 0,
            "visit_frequency_distribution": visit_frequency,
            "customer_details": [
                {
                    "customer_id": cust_id,
                    "visit_count": len(data["visits"]),
                    "city": data["city"],
                    "coordinates": [data["latitude"], data["longitude"]]
                }
                for cust_id, data in customer_visits.items()
            ]
        }

    async def _get_territory_coverage_metrics(
        self, 
        dealer_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calculate territory coverage and optimization metrics"""
        # Get dealer's assigned territory
        dealer = await self.dealer_repo.get_by_id(dealer_id)
        territory_code = dealer.territory_code if dealer else None

        # Get all customers in territory
        territory_customers = self.db.query(Customer).filter(
            Customer.territory_code == territory_code
        ).all() if territory_code else []

        # Get visited customers
        visited_customers = self.db.query(
            func.distinct(Sales.distributor_code)
        ).filter(
            and_(
                Sales.user_code == dealer_id,
                Sales.date >= start_date,
                Sales.date <= end_date
            )
        ).all()

        visited_customer_ids = [vc[0] for vc in visited_customers]
        
        # Calculate coverage percentage
        coverage_percentage = (
            len(visited_customer_ids) / len(territory_customers) * 100
        ) if territory_customers else 0

        # Identify unvisited customers
        unvisited_customers = [
            {
                "customer_id": customer.no,
                "city": customer.city,
                "latitude": customer.latitude,
                "longitude": customer.longitude
            }
            for customer in territory_customers
            if customer.no not in visited_customer_ids
        ]

        return {
            "territory_code": territory_code,
            "total_customers_in_territory": len(territory_customers),
            "visited_customers_count": len(visited_customer_ids),
            "coverage_percentage": round(coverage_percentage, 2),
            "unvisited_customers": unvisited_customers,
            "territory_performance_grade": self._get_territory_grade(coverage_percentage)
        }

    def _calculate_overall_performance_score(
        self, 
        sales_metrics: Dict, 
        gps_metrics: Dict, 
        visit_metrics: Dict, 
        territory_metrics: Dict
    ) -> Dict[str, Any]:
        """Calculate overall performance score based on multiple metrics"""
        
        # Define weights for different metrics
        weights = {
            "sales": 0.4,
            "activity": 0.3,
            "coverage": 0.3
        }

        # Sales score (0-100)
        sales_score = min(100, (sales_metrics["total_orders"] * 5) + 
                         (sales_metrics["total_revenue"] / 10000))

        # Activity score (0-100)
        activity_score = min(100, (gps_metrics["active_hours"] * 10) + 
                           (visit_metrics["total_visits"] * 5))

        # Coverage score (0-100)
        coverage_score = territory_metrics["coverage_percentage"]

        # Weighted overall score
        overall_score = (
            sales_score * weights["sales"] +
            activity_score * weights["activity"] +
            coverage_score * weights["coverage"]
        )

        return {
            "overall_score": round(overall_score, 2),
            "component_scores": {
                "sales_score": round(sales_score, 2),
                "activity_score": round(activity_score, 2),
                "coverage_score": round(coverage_score, 2)
            },
            "performance_grade": self._get_performance_grade(overall_score),
            "recommendations": self._generate_recommendations(
                sales_score, activity_score, coverage_score
            )
        }

    def _get_performance_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B+"
        elif score >= 60:
            return "B"
        elif score >= 50:
            return "C"
        else:
            return "D"

    def _get_territory_grade(self, coverage_percentage: float) -> str:
        """Get territory coverage grade"""
        if coverage_percentage >= 90:
            return "Excellent"
        elif coverage_percentage >= 75:
            return "Good"
        elif coverage_percentage >= 50:
            return "Average"
        else:
            return "Poor"

    def _generate_recommendations(
        self, 
        sales_score: float, 
        activity_score: float, 
        coverage_score: float
    ) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []

        if sales_score < 70:
            recommendations.append("Focus on increasing sales volume and order values")
            recommendations.append("Implement upselling and cross-selling strategies")

        if activity_score < 70:
            recommendations.append("Increase field activity and customer visit frequency")
            recommendations.append("Optimize daily route planning for better time management")

        if coverage_score < 70:
            recommendations.append("Improve territory coverage by visiting unvisited customers")
            recommendations.append("Develop systematic customer visit schedule")

        if not recommendations:
            recommendations.append("Maintain excellent performance standards")
            recommendations.append("Consider mentoring underperforming team members")

        return recommendations

    async def optimize_dealer_territory(
        self, 
        dealer_id: str, 
        optimization_criteria: str = "distance"
    ) -> Dict[str, Any]:
        """Optimize dealer territory assignment based on various criteria"""
        try:
            dealer = await self.dealer_repo.get_by_id(dealer_id)
            if not dealer:
                raise DealerNotFoundException(f"Dealer {dealer_id} not found")

            # Get dealer's current location (latest GPS point)
            latest_gps = self.db.query(GPSData).filter(
                GPSData.user_code == dealer_id
            ).order_by(GPSData.received_date.desc()).first()

            if not latest_gps:
                raise TerritoryOptimizationError("No GPS data available for dealer")

            dealer_location = (latest_gps.latitude, latest_gps.longitude)

            # Get all customers
            customers = self.db.query(Customer).filter(
                and_(
                    Customer.latitude.isnot(None),
                    Customer.longitude.isnot(None)
                )
            ).all()

            # Calculate optimization based on criteria
            if optimization_criteria == "distance":
                optimized_customers = self._optimize_by_distance(dealer_location, customers)
            elif optimization_criteria == "revenue":
                optimized_customers = await self._optimize_by_revenue(dealer_id, customers)
            elif optimization_criteria == "hybrid":
                optimized_customers = await self._optimize_hybrid(dealer_id, dealer_location, customers)
            else:
                raise TerritoryOptimizationError(f"Unknown optimization criteria: {optimization_criteria}")

            return {
                "dealer_id": dealer_id,
                "dealer_location": dealer_location,
                "optimization_criteria": optimization_criteria,
                "recommended_customers": optimized_customers[:50],  # Top 50 recommendations
                "total_customers_analyzed": len(customers),
                "optimization_score": self._calculate_optimization_score(optimized_customers)
            }

        except Exception as e:
            raise TerritoryOptimizationError(f"Territory optimization failed: {str(e)}")

    def _optimize_by_distance(
        self, 
        dealer_location: tuple, 
        customers: List[Customer]
    ) -> List[Dict[str, Any]]:
        """Optimize territory based on distance from dealer"""
        customer_distances = []
        
        for customer in customers:
            if customer.latitude and customer.longitude:
                distance = geodesic(
                    dealer_location, 
                    (customer.latitude, customer.longitude)
                ).kilometers
                
                customer_distances.append({
                    "customer_id": customer.no,
                    "city": customer.city,
                    "distance_km": distance,
                    "coordinates": [customer.latitude, customer.longitude],
                    "priority_score": 100 - min(distance * 2, 100)  # Closer = higher priority
                })
        
        return sorted(customer_distances, key=lambda x: x["distance_km"])

    async def _optimize_by_revenue(
        self, 
        dealer_id: str, 
        customers: List[Customer]
    ) -> List[Dict[str, Any]]:
        """Optimize territory based on historical revenue potential"""
        customer_revenues = []
        
        # Get historical sales data for each customer
        for customer in customers:
            revenue_query = self.db.query(
                func.sum(Sales.final_value).label('total_revenue'),
                func.count(Sales.id).label('order_count')
            ).filter(
                Sales.distributor_code == customer.no
            ).first()
            
            total_revenue = float(revenue_query.total_revenue or 0)
            order_count = revenue_query.order_count or 0
            
            customer_revenues.append({
                "customer_id": customer.no,
                "city": customer.city,
                "total_revenue": total_revenue,
                "order_count": order_count,
                "avg_order_value": total_revenue / order_count if order_count > 0 else 0,
                "coordinates": [customer.latitude, customer.longitude],
                "priority_score": min(total_revenue / 1000, 100)  # Higher revenue = higher priority
            })
        
        return sorted(customer_revenues, key=lambda x: x["total_revenue"], reverse=True)

    async def _optimize_hybrid(
        self, 
        dealer_id: str, 
        dealer_location: tuple, 
        customers: List[Customer]
    ) -> List[Dict[str, Any]]:
        """Optimize territory using hybrid approach (distance + revenue + other factors)"""
        optimized_customers = []
        
        for customer in customers:
            if customer.latitude and customer.longitude:
                # Distance factor
                distance = geodesic(
                    dealer_location, 
                    (customer.latitude, customer.longitude)
                ).kilometers
                distance_score = max(0, 100 - distance * 2)  # Closer = better
                
                # Revenue factor
                revenue_query = self.db.query(
                    func.sum(Sales.final_value).label('total_revenue'),
                    func.count(Sales.id).label('order_count')
                ).filter(
                    Sales.distributor_code == customer.no
                ).first()
                
                total_revenue = float(revenue_query.total_revenue or 0)
                revenue_score = min(total_revenue / 1000, 100)
                
                # Visit frequency factor
                visit_count = revenue_query.order_count or 0
                frequency_score = min(visit_count * 10, 100)
                
                # Hybrid score calculation
                hybrid_score = (
                    distance_score * 0.4 +
                    revenue_score * 0.4 +
                    frequency_score * 0.2
                )
                
                optimized_customers.append({
                    "customer_id": customer.no,
                    "city": customer.city,
                    "distance_km": distance,
                    "total_revenue": total_revenue,
                    "visit_count": visit_count,
                    "coordinates": [customer.latitude, customer.longitude],
                    "distance_score": round(distance_score, 2),
                    "revenue_score": round(revenue_score, 2),
                    "frequency_score": round(frequency_score, 2),
                    "priority_score": round(hybrid_score, 2)
                })
        
        return sorted(optimized_customers, key=lambda x: x["priority_score"], reverse=True)

    def _calculate_optimization_score(self, optimized_customers: List[Dict]) -> float:
        """Calculate overall optimization score"""
        if not optimized_customers:
            return 0.0
        
        avg_priority = sum(customer["priority_score"] for customer in optimized_customers) / len(optimized_customers)
        return round(avg_priority, 2)

    async def get_dealer_route_suggestions(
        self, 
        dealer_id: str, 
        date: datetime
    ) -> Dict[str, Any]:
        """Generate optimal route suggestions for dealer"""
        try:
            # Get dealer's planned customers for the date
            planned_customers = self.db.query(
                Sales.distributor_code,
                Customer.latitude,
                Customer.longitude,
                Customer.city
            ).join(
                Customer, Sales.distributor_code == Customer.no
            ).filter(
                and_(
                    Sales.user_code == dealer_id,
                    func.date(Sales.date) == date.date()
                )
            ).all()

            if not planned_customers:
                return {
                    "dealer_id": dealer_id,
                    "date": date.isoformat(),
                    "message": "No planned customers for this date",
                    "suggestions": []
                }

            # Get dealer's starting location
            dealer_gps = self.db.query(GPSData).filter(
                GPSData.user_code == dealer_id
            ).order_by(GPSData.received_date.desc()).first()

            if not dealer_gps:
                return {
                    "error": "No GPS data available for dealer"
                }

            start_location = (dealer_gps.latitude, dealer_gps.longitude)

            # Generate optimal route using nearest neighbor algorithm
            route = self._generate_optimal_route(start_location, planned_customers)

            return {
                "dealer_id": dealer_id,
                "date": date.isoformat(),
                "start_location": start_location,
                "optimized_route": route,
                "total_distance_km": sum(leg["distance_km"] for leg in route),
                "estimated_travel_time_hours": sum(leg["estimated_time_hours"] for leg in route),
                "route_efficiency_score": self._calculate_route_efficiency(route)
            }

        except Exception as e:
            raise Exception(f"Route suggestion generation failed: {str(e)}")

    def _generate_optimal_route(
        self, 
        start_location: tuple, 
        customers: List
    ) -> List[Dict[str, Any]]:
        """Generate optimal route using nearest neighbor algorithm"""
        route = []
        current_location = start_location
        remaining_customers = list(customers)
        
        while remaining_customers:
            # Find nearest customer
            nearest_customer = min(
                remaining_customers,
                key=lambda c: geodesic(
                    current_location, 
                    (c.latitude, c.longitude)
                ).kilometers
            )
            
            distance = geodesic(
                current_location, 
                (nearest_customer.latitude, nearest_customer.longitude)
            ).kilometers
            
            route.append({
                "customer_id": nearest_customer.distributor_code,
                "city": nearest_customer.city,
                "coordinates": [nearest_customer.latitude, nearest_customer.longitude],
                "distance_km": round(distance, 2),
                "estimated_time_hours": round(distance / 30, 2)  # Assuming 30 km/h average speed
            })
            
            current_location = (nearest_customer.latitude, nearest_customer.longitude)
            remaining_customers.remove(nearest_customer)
        
        return route

    def _calculate_route_efficiency(self, route: List[Dict]) -> float:
        """Calculate route efficiency score"""
        if not route:
            return 0.0
        
        total_distance = sum(leg["distance_km"] for leg in route)
        num_stops = len(route)
        
        # Efficiency score based on average distance per stop
        avg_distance_per_stop = total_distance / num_stops
        efficiency_score = max(0, 100 - avg_distance_per_stop * 5)
        
        return round(efficiency_score, 2)