
# Route Optimization Service for Sales Force Automation
# Provides route optimization algorithms and GPS processing capabilities


from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import math
import heapq
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from models.gps_data import GPSData
from models.customer import Customer
from models.sales import Sales
from models.dealer import Dealer
from utils.geo_utils import GeoUtils
from utils.date_utils import DateUtils


@dataclass
class RoutePoint:
    """Represents a point in a route"""
    customer_code: str
    latitude: float
    longitude: float
    priority: int = 1
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None
    service_time: int = 30  # minutes
    is_depot: bool = False


@dataclass
class RouteSegment:
    """Represents a segment between two points"""
    from_point: RoutePoint
    to_point: RoutePoint
    distance: float
    travel_time: int  # minutes
    cost: float


@dataclass
class OptimizedRoute:
    """Represents an optimized route"""
    dealer_code: str
    route_points: List[RoutePoint]
    total_distance: float
    total_time: int  # minutes
    total_cost: float
    estimated_fuel_cost: float
    route_efficiency: float


class RouteService:
    """Comprehensive route optimization and GPS processing service"""
    
    def __init__(self, db: Session):
        self.db = db
        self.geo_utils = GeoUtils()
        self.date_utils = DateUtils()
        
        # Configuration
        self.avg_speed_kmh = 40  # Average speed in km/h
        self.fuel_cost_per_km = 15.0  # LKR per km
        self.time_cost_per_hour = 500.0  # LKR per hour
        self.max_route_duration = 480  # 8 hours in minutes
        self.max_route_distance = 200  # 200 km max per route
    
    def optimize_dealer_routes(self, dealer_code: str, target_date: datetime, 
                              customer_priorities: Optional[Dict[str, int]] = None) -> OptimizedRoute:
        """Optimize routes for a specific dealer"""
        
        # Get dealer's customers for the target date
        customers_to_visit = self._get_customers_to_visit(dealer_code, target_date)
        
        if not customers_to_visit:
            return self._empty_route(dealer_code)
        
        # Get dealer's base location (depot)
        depot = self._get_dealer_depot(dealer_code)
        
        # Create route points
        route_points = [depot]  # Start with depot
        
        for customer in customers_to_visit:
            priority = customer_priorities.get(customer.customer_id, 1) if customer_priorities else 1
            
            route_point = RoutePoint(
                customer_code=customer.customer_id,
                latitude=customer.latitude,
                longitude=customer.longitude,
                priority=priority,
                service_time=self._estimate_service_time(customer.customer_id, target_date)
            )
            route_points.append(route_point)
        
        # Apply optimization algorithm
        optimized_route = self._apply_optimization_algorithm(dealer_code, route_points)
        
        # Calculate route metrics
        optimized_route = self._calculate_route_metrics(optimized_route)
        
        return optimized_route
    
    def optimize_multi_dealer_routes(self, territory_code: str, target_date: datetime) -> List[OptimizedRoute]:
        """Optimize routes for multiple dealers in a territory"""
        
        # Get all dealers in territory
        dealers = self.db.query(Dealer).filter(
            Dealer.territory_code == territory_code
        ).all()
        
        if not dealers:
            return []
        
        # Get all customers in territory
        all_customers = self._get_territory_customers(territory_code)
        
        # Assign customers to dealers based on proximity and workload
        dealer_assignments = self._assign_customers_to_dealers(dealers, all_customers)
        
        # Optimize route for each dealer
        optimized_routes = []
        for dealer_code, assigned_customers in dealer_assignments.items():
            if assigned_customers:
                # Create customer priorities based on assignment algorithm
                customer_priorities = {c.customer_id: c.priority for c in assigned_customers}
                
                route = self.optimize_dealer_routes(dealer_code, target_date, customer_priorities)
                optimized_routes.append(route)
        
        return optimized_routes
    
    def analyze_route_performance(self, dealer_code: str, start_date: datetime, 
                                end_date: datetime) -> Dict[str, Any]:
        """Analyze actual route performance vs optimal routes"""
        
        # Get actual GPS data
        actual_gps_data = self.db.query(GPSData).filter(
            and_(
                GPSData.user_code == dealer_code,
                GPSData.received_date.between(start_date, end_date)
            )
        ).order_by(GPSData.received_date).all()
        
        if not actual_gps_data:
            return {"error": "No GPS data found for analysis"}
        
        # Analyze actual routes by day
        daily_routes = self._group_gps_by_day(actual_gps_data)
        
        performance_analysis = {
            "dealer_code": dealer_code,
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "daily_performance": [],
            "overall_metrics": {},
            "improvement_opportunities": []
        }
        
        total_actual_distance = 0
        total_optimal_distance = 0
        total_actual_time = 0
        total_optimal_time = 0
        
        for date, gps_points in daily_routes.items():
            # Calculate actual route metrics
            actual_metrics = self._calculate_actual_route_metrics(gps_points)
            
            # Generate optimal route for comparison
            optimal_route = self.optimize_dealer_routes(dealer_code, date)
            
            # Calculate efficiency
            efficiency_score = self._calculate_route_efficiency(actual_metrics, optimal_route)
            
            daily_performance = {
                "date": date.isoformat(),
                "actual_distance": actual_metrics["total_distance"],
                "optimal_distance": optimal_route.total_distance,
                "actual_time": actual_metrics["total_time"],
                "optimal_time": optimal_route.total_time,
                "efficiency_score": efficiency_score,
                "potential_savings": {
                    "distance": actual_metrics["total_distance"] - optimal_route.total_distance,
                    "time": actual_metrics["total_time"] - optimal_route.total_time,
                    "cost": actual_metrics["total_cost"] - optimal_route.total_cost
                },
                "route_quality": self._assess_route_quality(actual_metrics, optimal_route)
            }
            
            performance_analysis["daily_performance"].append(daily_performance)
            
            # Accumulate totals
            total_actual_distance += actual_metrics["total_distance"]
            total_optimal_distance += optimal_route.total_distance
            total_actual_time += actual_metrics["total_time"]
            total_optimal_time += optimal_route.total_time
        
        # Calculate overall metrics
        performance_analysis["overall_metrics"] = {
            "total_actual_distance": round(total_actual_distance, 2),
            "total_optimal_distance": round(total_optimal_distance, 2),
            "distance_efficiency": round((total_optimal_distance / total_actual_distance) * 100, 2) if total_actual_distance > 0 else 0,
            "time_efficiency": round((total_optimal_time / total_actual_time) * 100, 2) if total_actual_time > 0 else 0,
            "potential_monthly_savings": {
                "distance": round(total_actual_distance - total_optimal_distance, 2),
                "fuel_cost": round((total_actual_distance - total_optimal_distance) * self.fuel_cost_per_km, 2),
                "time_hours": round((total_actual_time - total_optimal_time) / 60, 2)
            }
        }
        
        # Generate improvement opportunities
        performance_analysis["improvement_opportunities"] = self._generate_route_improvements(
            performance_analysis["daily_performance"]
        )
        
        return performance_analysis
    
    def get_real_time_route_guidance(self, dealer_code: str, current_location: Tuple[float, float]) -> Dict[str, Any]:
        """Provide real-time route guidance"""
        
        current_lat, current_lon = current_location
        
        # Get today's optimized route
        today = datetime.now().date()
        optimized_route = self.optimize_dealer_routes(dealer_code, today)
        
        if not optimized_route.route_points:
            return {"error": "No route found for today"}
        
        # Find current position in route
        current_position = self._find_current_position_in_route(
            optimized_route, current_lat, current_lon
        )
        
        # Get next destination
        next_destination = self._get_next_destination(optimized_route, current_position)
        
        # Calculate navigation details
        navigation = self._calculate_navigation_details(
            current_lat, current_lon, next_destination
        )
        
        # Estimate arrival times for remaining stops
        remaining_schedule = self._calculate_remaining_schedule(
            optimized_route, current_position
        )
        
        return {
            "current_location": {
                "latitude": current_lat,
                "longitude": current_lon
            },
            "next_destination": {
                "customer_code": next_destination.customer_code,
                "latitude": next_destination.latitude,
                "longitude": next_destination.longitude,
                "distance_km": navigation["distance"],
                "estimated_time_minutes": navigation["travel_time"]
            },
            "route_progress": {
                "completed_stops": current_position,
                "total_stops": len(optimized_route.route_points) - 1,  # Exclude depot
                "progress_percentage": round((current_position / (len(optimized_route.route_points) - 1)) * 100, 2)
            },
            "remaining_schedule": remaining_schedule,
            "route_adjustments": self._suggest_route_adjustments(optimized_route, current_position)
        }
    
    def detect_route_deviations(self, dealer_code: str, target_date: datetime) -> Dict[str, Any]:
        """Detect deviations from planned routes"""
        
        # Get planned route
        planned_route = self.optimize_dealer_routes(dealer_code, target_date)
        
        # Get actual GPS data for the day
        actual_gps = self.db.query(GPSData).filter(
            and_(
                GPSData.user_code == dealer_code,
                func.date(GPSData.received_date) == target_date.date()
            )
        ).order_by(GPSData.received_date).all()
        
        if not actual_gps:
            return {"error": "No GPS data found for deviation analysis"}
        
        # Analyze deviations
        deviations = []
        
        for i, gps_point in enumerate(actual_gps):
            # Find closest planned point
            closest_planned = self._find_closest_planned_point(
                planned_route, gps_point.latitude, gps_point.longitude
            )
            
            if closest_planned:
                deviation_distance = self.geo_utils.calculate_distance(
                    gps_point.latitude, gps_point.longitude,
                    closest_planned.latitude, closest_planned.longitude
                )
                
                # Flag significant deviations (> 1km)
                if deviation_distance > 1.0:
                    deviations.append({
                        "timestamp": gps_point.received_date.isoformat(),
                        "actual_location": {
                            "latitude": gps_point.latitude,
                            "longitude": gps_point.longitude
                        },
                        "planned_location": {
                            "latitude": closest_planned.latitude,
                            "longitude": closest_planned.longitude
                        },
                        "deviation_distance": round(deviation_distance, 2),
                        "severity": self._classify_deviation_severity(deviation_distance)
                    })
        
        # Calculate deviation statistics
        deviation_stats = self._calculate_deviation_statistics(deviations)
        
        return {
            "dealer_code": dealer_code,
            "date": target_date.date().isoformat(),
            "total_deviations": len(deviations),
            "deviations": deviations,
            "statistics": deviation_stats,
            "recommendations": self._generate_deviation_recommendations(deviations)
        }
    
    def calculate_territory_coverage(self, territory_code: str) -> Dict[str, Any]:
        """Calculate territory coverage metrics"""
        
        # Get all customers in territory
        territory_customers = self.db.query(Customer).filter(
            Customer.territory_code == territory_code
        ).all()
        
        # Get active dealers in territory
        territory_dealers = self.db.query(Dealer).filter(
            Dealer.territory_code == territory_code
        ).all()
        
        if not territory_customers or not territory_dealers:
            return {"error": "No customers or dealers found in territory"}
        
        # Calculate coverage metrics
        coverage_analysis = {
            "territory_code": territory_code,
            "total_customers": len(territory_customers),
            "total_dealers": len(territory_dealers),
            "coverage_matrix": [],
            "optimization_suggestions": []
        }
        
        # Analyze dealer-customer distance matrix
        for dealer in territory_dealers:
            dealer_coverage = {
                "dealer_code": dealer.user_code,
                "dealer_name": dealer.user_name,
                "assigned_customers": [],
                "coverage_radius": 0,
                "workload_score": 0
            }
            
            customer_distances = []
            for customer in territory_customers:
                distance = self.geo_utils.calculate_distance(
                    dealer.latitude, dealer.longitude,
                    customer.latitude, customer.longitude
                )
                
                customer_distances.append({
                    "customer_code": customer.customer_id,
                    "distance": round(distance, 2),
                    "is_optimal": distance <= 25  # 25km optimal radius
                })
            
            # Sort by distance and assign customers
            customer_distances.sort(key=lambda x: x["distance"])
            
            assigned_customers = []
            total_distance = 0
            
            for customer_dist in customer_distances:
                if len(assigned_customers) < 15 and customer_dist["distance"] <= 50:  # Max 15 customers, 50km max
                    assigned_customers.append(customer_dist)
                    total_distance += customer_dist["distance"]
            
            dealer_coverage["assigned_customers"] = assigned_customers
            dealer_coverage["coverage_radius"] = max([c["distance"] for c in assigned_customers]) if assigned_customers else 0
            dealer_coverage["workload_score"] = self._calculate_workload_score(assigned_customers)
            
            coverage_analysis["coverage_matrix"].append(dealer_coverage)
        
        # Generate optimization suggestions
        coverage_analysis["optimization_suggestions"] = self._generate_coverage_optimizations(
            coverage_analysis["coverage_matrix"]
        )
        
        return coverage_analysis
    
    def generate_route_reports(self, dealer_code: str, start_date: datetime, 
                             end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive route reports"""
        
        # Get GPS data for the period
        gps_data = self.db.query(GPSData).filter(
            and_(
                GPSData.user_code == dealer_code,
                GPSData.received_date.between(start_date, end_date)
            )
        ).order_by(GPSData.received_date).all()
        
        # Get sales data for the same period
        sales_data = self.db.query(Sales).filter(
            and_(
                Sales.user_code == dealer_code,
                Sales.date.between(start_date, end_date)
            )
        ).all()
        
        if not gps_data:
            return {"error": "No GPS data found for report generation"}
        
        # Generate comprehensive report
        report = {
            "dealer_code": dealer_code,
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "executive_summary": self._generate_executive_summary(gps_data, sales_data),
            "daily_summaries": self._generate_daily_summaries(gps_data, sales_data),
            "route_efficiency": self._analyze_route_efficiency(gps_data),
            "customer_visit_analysis": self._analyze_customer_visits(gps_data, sales_data),
            "territory_coverage": self._analyze_territory_coverage(gps_data),
            "performance_trends": self._analyze_performance_trends(gps_data, sales_data),
            "recommendations": self._generate_route_recommendations(gps_data, sales_data)
        }
        
        return report
    
    # Helper methods
    def _get_customers_to_visit(self, dealer_code: str, target_date: datetime) -> List[Customer]:
        """Get customers that need to be visited on target date"""
        
        # Get customers in dealer's territory
        dealer = self.db.query(Dealer).filter(Dealer.user_code == dealer_code).first()
        if not dealer:
            return []
        
        # Get customers based on visit frequency and last visit
        customers = self.db.query(Customer).filter(
            Customer.territory_code == dealer.territory_code
        ).all()
        
        customers_to_visit = []
        for customer in customers:
            # Check if customer needs visit based on frequency
            last_visit = self._get_last_visit_date(dealer_code, customer.customer_id)
            if self._should_visit_customer(customer, last_visit, target_date):
                customers_to_visit.append(customer)
        
        return customers_to_visit
    
    def _get_dealer_depot(self, dealer_code: str) -> RoutePoint:
        """Get dealer's depot (base location)"""
        
        dealer = self.db.query(Dealer).filter(Dealer.user_code == dealer_code).first()
        
        if dealer and dealer.latitude and dealer.longitude:
            return RoutePoint(
                customer_code="DEPOT",
                latitude=dealer.latitude,
                longitude=dealer.longitude,
                is_depot=True,
                service_time=0
            )
        
        # Default depot location if dealer location not available
        return RoutePoint(
            customer_code="DEPOT",
            latitude=6.9271,  # Colombo default
            longitude=79.8612,
            is_depot=True,
            service_time=0
        )
    
    def _estimate_service_time(self, customer_id: str, target_date: datetime) -> int:
        """Estimate service time at customer location"""
        
        # Get historical sales data to estimate service time
        historical_orders = self.db.query(Sales).filter(
            and_(
                Sales.distributor_code == customer_id,
                Sales.date >= target_date - timedelta(days=90)
            )
        ).all()
        
        if historical_orders:
            avg_order_value = sum(order.final_value for order in historical_orders) / len(historical_orders)
            # More complex orders require more time
            if avg_order_value > 100000:  # High value orders
                return 45
            elif avg_order_value > 50000:  # Medium value orders
                return 35
            else:  # Small orders
                return 25
        
        return 30  # Default service time
    
    def _apply_optimization_algorithm(self, dealer_code: str, route_points: List[RoutePoint]) -> OptimizedRoute:
        """Apply route optimization algorithm (Nearest Neighbor with 2-opt improvement)"""
        
        if len(route_points) <= 2:
            return OptimizedRoute(
                dealer_code=dealer_code,
                route_points=route_points,
                total_distance=0,
                total_time=0,
                total_cost=0,
                estimated_fuel_cost=0,
                route_efficiency=100
            )
        
        # Step 1: Nearest Neighbor Algorithm
        optimized_points = self._nearest_neighbor_optimization(route_points)
        
        # Step 2: 2-opt improvement
        optimized_points = self._two_opt_optimization(optimized_points)
        
        # Step 3: Time window optimization
        optimized_points = self._optimize_time_windows(optimized_points)
        
        return OptimizedRoute(
            dealer_code=dealer_code,
            route_points=optimized_points,
            total_distance=0,  # Will be calculated later
            total_time=0,
            total_cost=0,
            estimated_fuel_cost=0,
            route_efficiency=0
        )
    
    def _nearest_neighbor_optimization(self, route_points: List[RoutePoint]) -> List[RoutePoint]:
        """Implement nearest neighbor algorithm"""
        
        if not route_points:
            return []
        
        # Start from depot
        depot = next((p for p in route_points if p.is_depot), route_points[0])
        unvisited = [p for p in route_points if not p.is_depot]
        optimized_route = [depot]
        current_point = depot
        
        while unvisited:
            # Find nearest unvisited point
            nearest_point = min(unvisited, key=lambda p: self.geo_utils.calculate_distance(
                current_point.latitude, current_point.longitude,
                p.latitude, p.longitude
            ))
            
            optimized_route.append(nearest_point)
            unvisited.remove(nearest_point)
            current_point = nearest_point
        
        # Return to depot
        optimized_route.append(depot)
        
        return optimized_route
    
    def _two_opt_optimization(self, route_points: List[RoutePoint]) -> List[RoutePoint]:
        """Apply 2-opt optimization to improve route"""
        
        if len(route_points) < 4:
            return route_points
        
        best_route = route_points[:]
        best_distance = self._calculate_total_route_distance(best_route)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(route_points) - 2):
                for j in range(i + 1, len(route_points)):
                    if j - i == 1:
                        continue  # Skip adjacent edges
                    
                    # Create new route by reversing the segment between i and j
                    new_route = route_points[:i] + route_points[i:j+1][::-1] + route_points[j+1:]
                    new_distance = self._calculate_total_route_distance(new_route)
                    
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        route_points = new_route
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_route
    
    def _optimize_time_windows(self, route_points: List[RoutePoint]) -> List[RoutePoint]:
        """Optimize route considering time windows"""
        
        optimized_points = []
        current_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)  # Start at 8 AM
        
        for point in route_points:
            if point.time_window_start and current_time < point.time_window_start:
                # Wait until time window opens
                current_time = point.time_window_start
            
            optimized_points.append(point)
            
            # Add service time and travel time to next point
            current_time += timedelta(minutes=point.service_time)
            
            if len(optimized_points) < len(route_points):
                next_point = route_points[route_points.index(point) + 1]
                travel_time = self._calculate_travel_time(point, next_point)
                current_time += timedelta(minutes=travel_time)
        
        return optimized_points
    
    def _calculate_route_metrics(self, route: OptimizedRoute) -> OptimizedRoute:
        """Calculate comprehensive route metrics"""
        
        if len(route.route_points) < 2:
            return route
        
        total_distance = 0
        total_time = 0
        total_cost = 0
        
        for i in range(len(route.route_points) - 1):
            current_point = route.route_points[i]
            next_point = route.route_points[i + 1]
            
            # Calculate distance
            distance = self.geo_utils.calculate_distance(
                current_point.latitude, current_point.longitude,
                next_point.latitude, next_point.longitude
            )
            
            # Calculate travel time
            travel_time = self._calculate_travel_time(current_point, next_point)
            
            total_distance += distance
            total_time += travel_time + next_point.service_time
        
        # Calculate costs
        fuel_cost = total_distance * self.fuel_cost_per_km
        time_cost = (total_time / 60) * self.time_cost_per_hour
        total_cost = fuel_cost + time_cost
        
        # Calculate efficiency (lower is better)
        theoretical_min_distance = self._calculate_theoretical_minimum_distance(route.route_points)
        efficiency = (theoretical_min_distance / total_distance * 100) if total_distance > 0 else 100
        
        route.total_distance = round(total_distance, 2)
        route.total_time = total_time
        route.total_cost = round(total_cost, 2)
        route.estimated_fuel_cost = round(fuel_cost, 2)
        route.route_efficiency = round(efficiency, 2)
        
        return route
    
    def _calculate_travel_time(self, from_point: RoutePoint, to_point: RoutePoint) -> int:
        """Calculate travel time between two points"""
        
        distance = self.geo_utils.calculate_distance(
            from_point.latitude, from_point.longitude,
            to_point.latitude, to_point.longitude
        )
        
        # Consider traffic factors
        traffic_factor = 1.2  # 20% extra time for traffic
        travel_time_hours = (distance / self.avg_speed_kmh) * traffic_factor
        
        return int(travel_time_hours * 60)  # Convert to minutes
    
    def _calculate_total_route_distance(self, route_points: List[RoutePoint]) -> float:
        """Calculate total distance for a route"""
        
        total_distance = 0
        for i in range(len(route_points) - 1):
            distance = self.geo_utils.calculate_distance(
                route_points[i].latitude, route_points[i].longitude,
                route_points[i + 1].latitude, route_points[i + 1].longitude
            )
            total_distance += distance
        
        return total_distance
    
    def _calculate_theoretical_minimum_distance(self, route_points: List[RoutePoint]) -> float:
        """Calculate theoretical minimum distance (straight line from depot to all points and back)"""
        
        depot = next((p for p in route_points if p.is_depot), route_points[0])
        non_depot_points = [p for p in route_points if not p.is_depot]
        
        if not non_depot_points:
            return 0
        
        # Calculate distance from depot to farthest point and back
        max_distance = 0
        for point in non_depot_points:
            distance = self.geo_utils.calculate_distance(
                depot.latitude, depot.longitude,
                point.latitude, point.longitude
            )
            max_distance = max(max_distance, distance)
        
        return max_distance * 2  # Round trip
    
    def _get_territory_customers(self, territory_code: str) -> List[Customer]:
        """Get all customers in a territory"""
        
        return self.db.query(Customer).filter(
            Customer.territory_code == territory_code
        ).all()
    
    def _assign_customers_to_dealers(self, dealers: List[Dealer], customers: List[Customer]) -> Dict[str, List[Customer]]:
        """Assign customers to dealers based on proximity and workload"""
        
        assignments = defaultdict(list)
        
        # Create priority queue for customer assignments
        customer_assignments = []
        
        for customer in customers:
            dealer_distances = []
            for dealer in dealers:
                distance = self.geo_utils.calculate_distance(
                    dealer.latitude, dealer.longitude,
                    customer.latitude, customer.longitude
                )
                dealer_distances.append((distance, dealer.user_code))
            
            # Sort by distance and add to priority queue
            dealer_distances.sort()
            customer.priority = 1  # Default priority
            
            # Assign to closest dealer with capacity
            for distance, dealer_code in dealer_distances:
                if len(assignments[dealer_code]) < 15:  # Max 15 customers per dealer
                    assignments[dealer_code].append(customer)
                    break
        
        return dict(assignments)
    
    def _group_gps_by_day(self, gps_data: List[GPSData]) -> Dict[datetime, List[GPSData]]:
        """Group GPS data by day"""
        
        daily_groups = defaultdict(list)
        
        for gps_point in gps_data:
            day = gps_point.received_date.date()
            daily_groups[day].append(gps_point)
        
        return dict(daily_groups)
    
    def _calculate_actual_route_metrics(self, gps_points: List[GPSData]) -> Dict[str, float]:
        """Calculate metrics from actual GPS data"""
        
        if len(gps_points) < 2:
            return {"total_distance": 0, "total_time": 0, "total_cost": 0}
        
        total_distance = 0
        
        for i in range(len(gps_points) - 1):
            distance = self.geo_utils.calculate_distance(
                gps_points[i].latitude, gps_points[i].longitude,
                gps_points[i + 1].latitude, gps_points[i + 1].longitude
            )
            total_distance += distance
        
        # Calculate total time
        start_time = gps_points[0].received_date
        end_time = gps_points[-1].received_date
        total_time = int((end_time - start_time).total_seconds() / 60)  # minutes
        
        # Calculate costs
        fuel_cost = total_distance * self.fuel_cost_per_km
        time_cost = (total_time / 60) * self.time_cost_per_hour
        total_cost = fuel_cost + time_cost
        
        return {
            "total_distance": round(total_distance, 2),
            "total_time": total_time,
            "total_cost": round(total_cost, 2),
            "fuel_cost": round(fuel_cost, 2)
        }
    
    def _calculate_route_efficiency(self, actual_metrics: Dict[str, float], optimal_route: OptimizedRoute) -> float:
        """Calculate route efficiency score"""
        
        if optimal_route.total_distance == 0 or actual_metrics["total_distance"] == 0:
            return 100
        
        distance_efficiency = (optimal_route.total_distance / actual_metrics["total_distance"]) * 100
        time_efficiency = (optimal_route.total_time / actual_metrics["total_time"]) * 100 if actual_metrics["total_time"] > 0 else 100
        
        # Weighted average (distance 60%, time 40%)
        overall_efficiency = (distance_efficiency * 0.6) + (time_efficiency * 0.4)
        
        return min(round(overall_efficiency, 2), 100)  # Cap at 100%
    
    def _assess_route_quality(self, actual_metrics: Dict[str, float], optimal_route: OptimizedRoute) -> str:
        """Assess overall route quality"""
        
        efficiency = self._calculate_route_efficiency(actual_metrics, optimal_route)
        
        if efficiency >= 90:
            return "Excellent"
        elif efficiency >= 80:
            return "Good"
        elif efficiency >= 70:
            return "Fair"
        elif efficiency >= 60:
            return "Poor"
        else:
            return "Very Poor"
    
    def _generate_route_improvements(self, daily_performance: List[Dict]) -> List[str]:
        """Generate route improvement recommendations"""
        
        improvements = []
        
        # Analyze patterns
        avg_efficiency = sum(day["efficiency_score"] for day in daily_performance) / len(daily_performance)
        
        if avg_efficiency < 80:
            improvements.append("Consider using route optimization suggestions more frequently")
        
        # Check for consistent inefficiencies
        poor_days = [day for day in daily_performance if day["efficiency_score"] < 70]
        if len(poor_days) > len(daily_performance) * 0.3:
            improvements.append("Review customer visit scheduling and prioritization")
        
        # Check distance vs time efficiency
        distance_savings = sum(day["potential_savings"]["distance"] for day in daily_performance)
        if distance_savings > 50:  # More than 50km potential savings
            improvements.append("Focus on reducing unnecessary travel distances")
        
        time_savings = sum(day["potential_savings"]["time"] for day in daily_performance)
        if time_savings > 300:  # More than 5 hours potential savings
            improvements.append("Optimize time management and reduce idle time")
        
        return improvements
    
    def _empty_route(self, dealer_code: str) -> OptimizedRoute:
        """Return empty route when no customers to visit"""
        
        depot = self._get_dealer_depot(dealer_code)
        
        return OptimizedRoute(
            dealer_code=dealer_code,
            route_points=[depot],
            total_distance=0,
            total_time=0,
            total_cost=0,
            estimated_fuel_cost=0,
            route_efficiency=100
        )
    
    def _get_last_visit_date(self, dealer_code: str, customer_id: str) -> Optional[datetime]:
        """Get last visit date for a customer"""
        
        last_sale = self.db.query(Sales).filter(
            and_(
                Sales.user_code == dealer_code,
                Sales.distributor_code == customer_id
            )
        ).order_by(Sales.date.desc()).first()
        
        return last_sale.date if last_sale else None
    
    def _should_visit_customer(self, customer: Customer, last_visit: Optional[datetime], target_date: datetime) -> bool:
        """Determine if customer should be visited on target date"""
        
        if not last_visit:
            return True  # First visit
        
        days_since_visit = (target_date - last_visit).days
        
        # Visit frequency based on customer tier (if available)
        if hasattr(customer, 'tier'):
            if customer.tier == 'A':
                return days_since_visit >= 7  # Weekly visits
            elif customer.tier == 'B':
                return days_since_visit >= 14  # Bi-weekly visits
            else:
                return days_since_visit >= 30  # Monthly visits
        
        # Default: visit if more than 2 weeks
        return days_since_visit >= 14
    
    def _find_current_position_in_route(self, route: OptimizedRoute, lat: float, lon: float) -> int:
        """Find current position in route based on GPS coordinates"""
        
        min_distance = float('inf')
        position = 0
        
        for i, point in enumerate(route.route_points):
            distance = self.geo_utils.calculate_distance(lat, lon, point.latitude, point.longitude)
            if distance < min_distance:
                min_distance = distance
                position = i
        
        return position
    
    def _get_next_destination(self, route: OptimizedRoute, current_position: int) -> RoutePoint:
        """Get next destination in route"""
        
        if current_position < len(route.route_points) - 1:
            return route.route_points[current_position + 1]
        
        # Return depot if at end of route
        return next((p for p in route.route_points if p.is_depot), route.route_points[0])
    
    def _calculate_navigation_details(self, current_lat: float, current_lon: float, destination: RoutePoint) -> Dict[str, Any]:
        """Calculate navigation details to destination"""
        
        distance = self.geo_utils.calculate_distance(
            current_lat, current_lon,
            destination.latitude, destination.longitude
        )
        
        travel_time = int((distance / self.avg_speed_kmh) * 60 * 1.2)  # Include traffic factor
        
        return {
            "distance": round(distance, 2),
            "travel_time": travel_time
        }
    
    def _calculate_remaining_schedule(self, route: OptimizedRoute, current_position: int) -> List[Dict]:
        """Calculate schedule for remaining stops"""
        
        remaining_schedule = []
        current_time = datetime.now()
        
        for i in range(current_position + 1, len(route.route_points)):
            point = route.route_points[i]
            
            if i > current_position + 1:
                # Add travel time from previous point
                prev_point = route.route_points[i - 1]
                travel_time = self._calculate_travel_time(prev_point, point)
                current_time += timedelta(minutes=travel_time)
            
            remaining_schedule.append({
                "customer_code": point.customer_code,
                "estimated_arrival": current_time.strftime("%H:%M"),
                "service_time": point.service_time,
                "estimated_departure": (current_time + timedelta(minutes=point.service_time)).strftime("%H:%M")
            })
            
            current_time += timedelta(minutes=point.service_time)
        
        return remaining_schedule
    
    def _suggest_route_adjustments(self, route: OptimizedRoute, current_position: int) -> List[str]:
        """Suggest real-time route adjustments"""
        
        suggestions = []
        
        # Check remaining time vs remaining stops
        remaining_stops = len(route.route_points) - current_position - 1
        current_time = datetime.now()
        end_of_day = current_time.replace(hour=17, minute=0, second=0, microsecond=0)
        remaining_time = (end_of_day - current_time).total_seconds() / 60  # minutes
        
        if remaining_stops > 0:
            time_per_stop = remaining_time / remaining_stops
            
            if time_per_stop < 45:  # Less than 45 minutes per stop
                suggestions.append("Consider postponing low-priority customers to finish on time")
            
            if remaining_stops > 8:
                suggestions.append("High number of remaining stops - consider optimizing route")
        
        return suggestions
    
    def _find_closest_planned_point(self, route: OptimizedRoute, lat: float, lon: float) -> Optional[RoutePoint]:
        """Find closest planned point to GPS coordinates"""
        
        if not route.route_points:
            return None
        
        closest_point = min(route.route_points, key=lambda p: self.geo_utils.calculate_distance(
            lat, lon, p.latitude, p.longitude
        ))
        
        return closest_point
    
    def _classify_deviation_severity(self, deviation_distance: float) -> str:
        """Classify deviation severity based on distance"""
        
        if deviation_distance < 1:
            return "Minor"
        elif deviation_distance < 3:
            return "Moderate"
        elif deviation_distance < 5:
            return "Significant"
        else:
            return "Major"
    
    def _calculate_deviation_statistics(self, deviations: List[Dict]) -> Dict[str, Any]:
        """Calculate deviation statistics"""
        
        if not deviations:
            return {"avg_deviation": 0, "max_deviation": 0, "deviation_frequency": 0}
        
        distances = [d["deviation_distance"] for d in deviations]
        
        return {
            "avg_deviation": round(sum(distances) / len(distances), 2),
            "max_deviation": round(max(distances), 2),
            "deviation_frequency": len(deviations),
            "severity_breakdown": {
                "minor": len([d for d in deviations if d["severity"] == "Minor"]),
                "moderate": len([d for d in deviations if d["severity"] == "Moderate"]),
                "significant": len([d for d in deviations if d["severity"] == "Significant"]),
                "major": len([d for d in deviations if d["severity"] == "Major"])
            }
        }
    
    def _generate_deviation_recommendations(self, deviations: List[Dict]) -> List[str]:
        """Generate recommendations based on deviations"""
        
        recommendations = []
        
        if not deviations:
            return ["Excellent route adherence - no significant deviations detected"]
        
        major_deviations = len([d for d in deviations if d["severity"] == "Major"])
        if major_deviations > 0:
            recommendations.append(f"Address {major_deviations} major route deviations - review route planning")
        
        if len(deviations) > 10:
            recommendations.append("High deviation frequency - consider route optimization training")
        
        avg_deviation = sum(d["deviation_distance"] for d in deviations) / len(deviations)
        if avg_deviation > 2:
            recommendations.append("Average deviation distance is high - review navigation systems")
        
        return recommendations
    
    def _calculate_workload_score(self, assigned_customers: List[Dict]) -> float:
        """Calculate workload score for dealer"""
        
        if not assigned_customers:
            return 0
        
        # Factor in number of customers and distances
        customer_count = len(assigned_customers)
        total_distance = sum(c["distance"] for c in assigned_customers)
        avg_distance = total_distance / customer_count if customer_count > 0 else 0
        
        # Workload score (0-100, higher means more workload)
        workload_score = min(100, (customer_count * 5) + (avg_distance * 2))
        
        return round(workload_score, 2)
    
    def _generate_coverage_optimizations(self, coverage_matrix: List[Dict]) -> List[str]:
        """Generate territory coverage optimization suggestions"""
        
        suggestions = []
        
        # Check for overloaded dealers
        overloaded_dealers = [d for d in coverage_matrix if d["workload_score"] > 80]
        if overloaded_dealers:
            suggestions.append(f"Redistribute customers from {len(overloaded_dealers)} overloaded dealers")
        
        # Check for underutilized dealers
        underutilized_dealers = [d for d in coverage_matrix if d["workload_score"] < 30]
        if underutilized_dealers:
            suggestions.append(f"Assign more customers to {len(underutilized_dealers)} underutilized dealers")
        
        # Check coverage gaps
        high_distance_assignments = []
        for dealer in coverage_matrix:
            far_customers = [c for c in dealer["assigned_customers"] if c["distance"] > 30]
            if far_customers:
                high_distance_assignments.extend(far_customers)
        
        if high_distance_assignments:
            suggestions.append(f"Review {len(high_distance_assignments)} long-distance customer assignments")
        
        return suggestions
    
    def _generate_executive_summary(self, gps_data: List[GPSData], sales_data: List[Sales]) -> Dict[str, Any]:
        """Generate executive summary for route reports"""
        
        total_days = len(set(gps.received_date.date() for gps in gps_data))
        total_sales = len(sales_data)
        total_sales_value = sum(sale.final_value for sale in sales_data)
        
        # Calculate total distance from GPS data
        total_distance = 0
        daily_gps = self._group_gps_by_day(gps_data)
        
        for day_gps in daily_gps.values():
            day_metrics = self._calculate_actual_route_metrics(day_gps)
            total_distance += day_metrics["total_distance"]
        
        return {
            "total_active_days": total_days,
            "total_distance_covered": round(total_distance, 2),
            "total_sales_orders": total_sales,
            "total_sales_value": round(total_sales_value, 2),
            "avg_daily_distance": round(total_distance / total_days, 2) if total_days > 0 else 0,
            "avg_sales_per_day": round(total_sales / total_days, 2) if total_days > 0 else 0,
            "efficiency_rating": self._calculate_overall_efficiency_rating(gps_data, sales_data)
        }
    
    def _generate_daily_summaries(self, gps_data: List[GPSData], sales_data: List[Sales]) -> List[Dict]:
        """Generate daily summaries"""
        
        daily_gps = self._group_gps_by_day(gps_data)
        daily_sales = defaultdict(list)
        
        for sale in sales_data:
            daily_sales[sale.date.date()].append(sale)
        
        summaries = []
        
        for date, day_gps in daily_gps.items():
            day_sales = daily_sales.get(date, [])
            route_metrics = self._calculate_actual_route_metrics(day_gps)
            
            summary = {
                "date": date.isoformat(),
                "distance_covered": route_metrics["total_distance"],
                "time_active": route_metrics["total_time"],
                "sales_orders": len(day_sales),
                "sales_value": sum(sale.final_value for sale in day_sales),
                "fuel_cost": route_metrics["fuel_cost"],
                "customers_visited": len(set(sale.distributor_code for sale in day_sales)),
                "efficiency_score": self._calculate_daily_efficiency(day_gps, day_sales)
            }
            
            summaries.append(summary)
        
        return sorted(summaries, key=lambda x: x["date"])
    
    def _analyze_route_efficiency(self, gps_data: List[GPSData]) -> Dict[str, Any]:
        """Analyze route efficiency metrics"""
        
        daily_gps = self._group_gps_by_day(gps_data)
        efficiency_scores = []
        
        for day_gps in daily_gps.values():
            # Calculate efficiency based on distance and time optimization
            metrics = self._calculate_actual_route_metrics(day_gps)
            
            # Simple efficiency calculation based on distance/time ratio
            if metrics["total_time"] > 0:
                efficiency = (metrics["total_distance"] / metrics["total_time"]) * 100
                efficiency_scores.append(min(efficiency, 100))
        
        if efficiency_scores:
            return {
                "avg_efficiency": round(sum(efficiency_scores) / len(efficiency_scores), 2),
                "best_day_efficiency": round(max(efficiency_scores), 2),
                "worst_day_efficiency": round(min(efficiency_scores), 2),
                "consistency_score": round(100 - (np.std(efficiency_scores) * 10), 2)
            }
        
        return {"avg_efficiency": 0, "best_day_efficiency": 0, "worst_day_efficiency": 0, "consistency_score": 0}
    
    def _analyze_customer_visits(self, gps_data: List[GPSData], sales_data: List[Sales]) -> Dict[str, Any]:
        """Analyze customer visit patterns"""
        
        unique_customers = set(sale.distributor_code for sale in sales_data)
        visit_frequency = defaultdict(int)
        
        for sale in sales_data:
            visit_frequency[sale.distributor_code] += 1
        
        return {
            "total_unique_customers": len(unique_customers),
            "avg_visits_per_customer": round(sum(visit_frequency.values()) / len(unique_customers), 2) if unique_customers else 0,
            "most_visited_customers": sorted(visit_frequency.items(), key=lambda x: x[1], reverse=True)[:5],
            "single_visit_customers": len([c for c, count in visit_frequency.items() if count == 1])
        }
    
    def _analyze_territory_coverage(self, gps_data: List[GPSData]) -> Dict[str, Any]:
        """Analyze territory coverage from GPS data"""
        
        if not gps_data:
            return {"coverage_area": 0, "geographic_spread": 0}
        
        # Calculate bounding box of GPS coordinates
        latitudes = [gps.latitude for gps in gps_data]
        longitudes = [gps.longitude for gps in gps_data]
        
        lat_range = max(latitudes) - min(latitudes)
        lon_range = max(longitudes) - min(longitudes)
        
        # Estimate coverage area (rough approximation)
        coverage_area = lat_range * lon_range * 111 * 111  # Convert to sq km approximately
        
        # Calculate geographic spread
        center_lat = sum(latitudes) / len(latitudes)
        center_lon = sum(longitudes) / len(longitudes)
        
        max_distance_from_center = 0
        for gps in gps_data:
            distance = self.geo_utils.calculate_distance(
                center_lat, center_lon, gps.latitude, gps.longitude
            )
            max_distance_from_center = max(max_distance_from_center, distance)
        
        return {
            "coverage_area_sq_km": round(coverage_area, 2),
            "geographic_spread_km": round(max_distance_from_center * 2, 2),  # Diameter
            "territory_center": {
                "latitude": round(center_lat, 6),
                "longitude": round(center_lon, 6)
            },
            "coverage_bounds": {
                "north": max(latitudes),
                "south": min(latitudes),
                "east": max(longitudes),
                "west": min(longitudes)
            }
        }
    
    def _analyze_performance_trends(self, gps_data: List[GPSData], sales_data: List[Sales]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        
        daily_gps = self._group_gps_by_day(gps_data)
        daily_sales = defaultdict(list)
        
        for sale in sales_data:
            daily_sales[sale.date.date()].append(sale)
        
        daily_metrics = []
        
        for date in sorted(daily_gps.keys()):
            day_gps = daily_gps[date]
            day_sales = daily_sales.get(date, [])
            route_metrics = self._calculate_actual_route_metrics(day_gps)
            
            daily_metrics.append({
                "date": date,
                "distance": route_metrics["total_distance"],
                "sales_count": len(day_sales),
                "sales_value": sum(sale.final_value for sale in day_sales),
                "efficiency": self._calculate_daily_efficiency(day_gps, day_sales)
            })
        
        if len(daily_metrics) < 2:
            return {"trend": "Insufficient data for trend analysis"}
        
        # Calculate trends
        distances = [d["distance"] for d in daily_metrics]
        sales_values = [d["sales_value"] for d in daily_metrics]
        efficiencies = [d["efficiency"] for d in daily_metrics]
        
        return {
            "distance_trend": self._calculate_trend(distances),
            "sales_trend": self._calculate_trend(sales_values),
            "efficiency_trend": self._calculate_trend(efficiencies),
            "performance_correlation": {
                "distance_vs_sales": self._calculate_correlation(distances, sales_values),
                "efficiency_vs_sales": self._calculate_correlation(efficiencies, sales_values)
            }
        }
    
    def _generate_route_recommendations(self, gps_data: List[GPSData], sales_data: List[Sales]) -> List[str]:
        """Generate route recommendations based on analysis"""
        
        recommendations = []
        
        # Analyze efficiency
        daily_gps = self._group_gps_by_day(gps_data)
        efficiency_scores = []
        
        for day_gps in daily_gps.values():
            day_sales = [s for s in sales_data if s.date.date() == day_gps[0].received_date.date()]
            efficiency = self._calculate_daily_efficiency(day_gps, day_sales)
            efficiency_scores.append(efficiency)
        
        if efficiency_scores:
            avg_efficiency = sum(efficiency_scores) / len(efficiency_scores)
            
            if avg_efficiency < 70:
                recommendations.append("Overall route efficiency is below optimal - consider route optimization")
            
            if max(efficiency_scores) - min(efficiency_scores) > 30:
                recommendations.append("High variability in daily performance - standardize route planning")
        
        # Analyze distance patterns
        total_distance = sum(self._calculate_actual_route_metrics(day_gps)["total_distance"] 
                           for day_gps in daily_gps.values())
        avg_daily_distance = total_distance / len(daily_gps) if daily_gps else 0
        
        if avg_daily_distance > 100:
            recommendations.append("High daily travel distance - review customer assignments")
        elif avg_daily_distance < 30:
            recommendations.append("Low daily travel distance - consider expanding territory coverage")
        
        # Analyze sales efficiency
        total_sales = len(sales_data)
        sales_per_km = total_sales / total_distance if total_distance > 0 else 0
        
        if sales_per_km < 0.5:
            recommendations.append("Low sales per kilometer - focus on high-potential customers")
        
        return recommendations
    
    def _calculate_overall_efficiency_rating(self, gps_data: List[GPSData], sales_data: List[Sales]) -> str:
        """Calculate overall efficiency rating"""
        
        daily_gps = self._group_gps_by_day(gps_data)
        efficiency_scores = []
        
        for day_gps in daily_gps.values():
            day_sales = [s for s in sales_data if s.date.date() == day_gps[0].received_date.date()]
            efficiency = self._calculate_daily_efficiency(day_gps, day_sales)
            efficiency_scores.append(efficiency)
        
        if not efficiency_scores:
            return "No Data"
        
        avg_efficiency = sum(efficiency_scores) / len(efficiency_scores)
        
        if avg_efficiency >= 85:
            return "Excellent"
        elif avg_efficiency >= 75:
            return "Good"
        elif avg_efficiency >= 65:
            return "Fair"
        elif avg_efficiency >= 50:
            return "Poor"
        else:
            return "Very Poor"
    
    def _calculate_daily_efficiency(self, day_gps: List[GPSData], day_sales: List[Sales]) -> float:
        """Calculate daily route efficiency"""
        
        if not day_gps or not day_sales:
            return 0
        
        route_metrics = self._calculate_actual_route_metrics(day_gps)
        sales_value = sum(sale.final_value for sale in day_sales)
        
        # Efficiency based on sales value per kilometer and per hour
        distance_efficiency = sales_value / route_metrics["total_distance"] if route_metrics["total_distance"] > 0 else 0
        time_efficiency = sales_value / (route_metrics["total_time"] / 60) if route_metrics["total_time"] > 0 else 0
        
        # Normalize and combine (scale to 0-100)
        normalized_efficiency = min(100, (distance_efficiency / 1000) + (time_efficiency / 2000))
        
        return round(normalized_efficiency, 2)
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        
        if len(values) < 2:
            return "No trend"
        
        # Simple linear trend
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.1:
            return "Improving"
        elif second_avg < first_avg * 0.9:
            return "Declining"
        else:
            return "Stable"
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate correlation coefficient"""
        
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0
        
        # Simple correlation calculation
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        sum_y2 = sum(y * y for y in y_values)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        if denominator == 0:
            return 0
        
        correlation = numerator / denominator
        return round(correlation, 3)


# Utility Classes
class GeoUtils:
    """Geographic utility functions"""
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r
    
    @staticmethod
    def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing between two points"""
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    @staticmethod
    def is_point_in_radius(center_lat: float, center_lon: float, point_lat: float, 
                          point_lon: float, radius_km: float) -> bool:
        """Check if point is within radius of center point"""
        
        distance = GeoUtils.calculate_distance(center_lat, center_lon, point_lat, point_lon)
        return distance <= radius_km


class DateUtils:
    """Date utility functions"""
    
    @staticmethod
    def get_business_days(start_date: datetime, end_date: datetime) -> List[datetime]:
        """Get list of business days between two dates"""
        
        business_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Monday = 0, Sunday = 6
                business_days.append(current_date)
            current_date += timedelta(days=1)
        
        return business_days
    
    @staticmethod
    def is_business_day(date: datetime) -> bool:
        """Check if date is a business day"""
        return date.weekday() < 5
    
    @staticmethod
    def get_week_start(date: datetime) -> datetime:
        """Get start of week (Monday) for given date"""
        days_since_monday = date.weekday()
        return date - timedelta(days=days_since_monday)
    
    @staticmethod
    def get_month_start(date: datetime) -> datetime:
        """Get start of month for given date"""
        return date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)