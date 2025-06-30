"""
Geospatial utility functions for SFA system.
Handles distance calculations, coordinate transformations, and location-based operations.
"""

import math
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import transform
import pyproj
from functools import lru_cache

@dataclass
class Location:
    """Represents a geographical location."""
    latitude: float
    longitude: float
    name: Optional[str] = None
    address: Optional[str] = None
    
    def __post_init__(self):
        self.validate_coordinates()
    
    def validate_coordinates(self):
        """Validate latitude and longitude ranges."""
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Invalid latitude: {self.latitude}. Must be between -90 and 90.")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Invalid longitude: {self.longitude}. Must be between -180 and 180.")
    
    def to_tuple(self) -> Tuple[float, float]:
        """Return coordinates as (lat, lon) tuple."""
        return (self.latitude, self.longitude)
    
    def to_dict(self) -> Dict[str, Union[float, str]]:
        """Return location as dictionary."""
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'name': self.name,
            'address': self.address
        }

@dataclass
class BoundingBox:
    """Represents a geographical bounding box."""
    north: float  # Max latitude
    south: float  # Min latitude
    east: float   # Max longitude
    west: float   # Min longitude
    
    def contains_point(self, location: Location) -> bool:
        """Check if a point is within the bounding box."""
        return (self.south <= location.latitude <= self.north and
                self.west <= location.longitude <= self.east)
    
    def get_center(self) -> Location:
        """Get center point of bounding box."""
        center_lat = (self.north + self.south) / 2
        center_lon = (self.east + self.west) / 2
        return Location(center_lat, center_lon, "Center")

class GeoUtils:
    """Comprehensive geospatial utility class for SFA operations."""
    
    # Sri Lankan geographical bounds
    SL_BOUNDS = BoundingBox(
        north=9.8311,   # Point Pedro
        south=5.9167,   # Dondra Head
        east=81.8812,   # Sangamankanda Point
        west=79.6959    # Kalpitiya
    )
    
    # Major Sri Lankan cities with coordinates
    SL_CITIES = {
        'COLOMBO': Location(6.9271, 79.8612, 'Colombo'),
        'KANDY': Location(7.2906, 80.6337, 'Kandy'),
        'GALLE': Location(6.0329, 80.2168, 'Galle'),
        'JAFFNA': Location(9.6615, 80.0255, 'Jaffna'),
        'NEGOMBO': Location(7.2083, 79.8358, 'Negombo'),
        'ANURADHAPURA': Location(8.3114, 80.4037, 'Anuradhapura'),
        'MATARA': Location(5.9549, 80.5550, 'Matara'),
        'BATTICALOA': Location(7.7102, 81.6924, 'Batticaloa'),
        'TRINCOMALEE': Location(8.5874, 81.2152, 'Trincomalee'),
        'KURUNEGALA': Location(7.4818, 80.3609, 'Kurunegala')
    }
    
    @staticmethod
    def haversine_distance(loc1: Location, loc2: Location) -> float:
        """
        Calculate the great circle distance between two points using Haversine formula.
        Returns distance in kilometers.
        """
        lat1, lon1 = math.radians(loc1.latitude), math.radians(loc1.longitude)
        lat2, lon2 = math.radians(loc2.latitude), math.radians(loc2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        
        return c * r
    
    @staticmethod
    def geodesic_distance(loc1: Location, loc2: Location) -> float:
        """
        Calculate geodesic distance between two points using geopy.
        More accurate than Haversine for precise calculations.
        Returns distance in kilometers.
        """
        return geodesic(loc1.to_tuple(), loc2.to_tuple()).kilometers
    
    @staticmethod
    def bearing_between_points(loc1: Location, loc2: Location) -> float:
        """
        Calculate the bearing (direction) from loc1 to loc2.
        Returns bearing in degrees (0-360).
        """
        lat1, lon1 = math.radians(loc1.latitude), math.radians(loc1.longitude)
        lat2, lon2 = math.radians(loc2.latitude), math.radians(loc2.longitude)
        
        dlon = lon2 - lon1
        
        y = math.sin(dlon) * math.cos(lat2)
        x = (math.cos(lat1) * math.sin(lat2) - 
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
    @staticmethod
    def get_direction_name(bearing: float) -> str:
        """Convert bearing to compass direction name."""
        directions = [
            "North", "North-Northeast", "Northeast", "East-Northeast",
            "East", "East-Southeast", "Southeast", "South-Southeast",
            "South", "South-Southwest", "Southwest", "West-Southwest",
            "West", "West-Northwest", "Northwest", "North-Northwest"
        ]
        
        index = round(bearing / 22.5) % 16
        return directions[index]
    
    @staticmethod
    def find_nearest_location(target: Location, locations: List[Location]) -> Tuple[Location, float]:
        """
        Find the nearest location from a list of locations.
        Returns the nearest location and distance in km.
        """
        if not locations:
            raise ValueError("Locations list cannot be empty")
        
        nearest_location = locations[0]
        min_distance = GeoUtils.geodesic_distance(target, nearest_location)
        
        for location in locations[1:]:
            distance = GeoUtils.geodesic_distance(target, location)
            if distance < min_distance:
                min_distance = distance
                nearest_location = location
        
        return nearest_location, min_distance
    
    @staticmethod
    def get_locations_within_radius(center: Location, locations: List[Location], 
                                   radius_km: float) -> List[Tuple[Location, float]]:
        """
        Get all locations within specified radius from center point.
        Returns list of (location, distance) tuples.
        """
        nearby_locations = []
        
        for location in locations:
            distance = GeoUtils.geodesic_distance(center, location)
            if distance <= radius_km:
                nearby_locations.append((location, distance))
        
        # Sort by distance
        nearby_locations.sort(key=lambda x: x[1])
        return nearby_locations
    
    @staticmethod
    def calculate_route_distance(locations: List[Location]) -> float:
        """
        Calculate total distance of a route through multiple locations.
        Returns distance in kilometers.
        """
        if len(locations) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(locations) - 1):
            total_distance += GeoUtils.geodesic_distance(locations[i], locations[i + 1])
        
        return total_distance
    
    @staticmethod
    def get_route_center(locations: List[Location]) -> Location:
        """Calculate the geographic center of a route."""
        if not locations:
            raise ValueError("Locations list cannot be empty")
        
        avg_lat = sum(loc.latitude for loc in locations) / len(locations)
        avg_lon = sum(loc.longitude for loc in locations) / len(locations)
        
        return Location(avg_lat, avg_lon, "Route Center")
    
    @staticmethod
    def get_bounding_box(locations: List[Location]) -> BoundingBox:
        """Get bounding box that contains all locations."""
        if not locations:
            raise ValueError("Locations list cannot be empty")
        
        latitudes = [loc.latitude for loc in locations]
        longitudes = [loc.longitude for loc in locations]
        
        return BoundingBox(
            north=max(latitudes),
            south=min(latitudes),
            east=max(longitudes),
            west=min(longitudes)
        )
    
    @staticmethod
    def is_location_in_sri_lanka(location: Location) -> bool:
        """Check if location is within Sri Lankan boundaries."""
        return GeoUtils.SL_BOUNDS.contains_point(location)
    
    @staticmethod
    def get_nearest_city(location: Location) -> Tuple[str, Location, float]:
        """
        Find the nearest major Sri Lankan city to given location.
        Returns city name, city location, and distance in km.
        """
        cities = list(GeoUtils.SL_CITIES.values())
        nearest_city, distance = GeoUtils.find_nearest_location(location, cities)
        
        # Find city name
        city_name = None
        for name, city_loc in GeoUtils.SL_CITIES.items():
            if (city_loc.latitude == nearest_city.latitude and 
                city_loc.longitude == nearest_city.longitude):
                city_name = name
                break
        
        return city_name, nearest_city, distance
    
    @staticmethod
    def calculate_speed(loc1: Location, loc2: Location, time_diff_seconds: float) -> float:
        """
        Calculate speed between two locations.
        Returns speed in km/h.
        """
        if time_diff_seconds <= 0:
            return 0.0
        
        distance_km = GeoUtils.geodesic_distance(loc1, loc2)
        time_hours = time_diff_seconds / 3600
        
        return distance_km / time_hours if time_hours > 0 else 0.0
    
    @staticmethod
    def detect_stationary_points(gps_points: List[Tuple[Location, float]], 
                                min_time_seconds: int = 300,
                                max_radius_meters: float = 50) -> List[Dict]:
        """
        Detect stationary points in GPS tracking data.
        Returns list of stationary periods with location and duration.
        """
        if len(gps_points) < 2:
            return []
        
        stationary_points = []
        current_cluster = []
        
        for i, (location, timestamp) in enumerate(gps_points):
            if not current_cluster:
                current_cluster = [(location, timestamp)]
                continue
            
            # Check if current point is within radius of cluster center
            cluster_center = GeoUtils.get_route_center([loc for loc, _ in current_cluster])
            distance_meters = GeoUtils.geodesic_distance(location, cluster_center) * 1000
            
            if distance_meters <= max_radius_meters:
                current_cluster.append((location, timestamp))
            else:
                # End current cluster and check if it's long enough
                if len(current_cluster) >= 2:
                    duration = current_cluster[-1][1] - current_cluster[0][1]
                    if duration >= min_time_seconds:
                        center = GeoUtils.get_route_center([loc for loc, _ in current_cluster])
                        stationary_points.append({
                            'location': center,
                            'start_time': current_cluster[0][1],
                            'end_time': current_cluster[-1][1],
                            'duration_seconds': duration,
                            'point_count': len(current_cluster)
                        })
                
                # Start new cluster
                current_cluster = [(location, timestamp)]
        
        # Check final cluster
        if len(current_cluster) >= 2:
            duration = current_cluster[-1][1] - current_cluster[0][1]
            if duration >= min_time_seconds:
                center = GeoUtils.get_route_center([loc for loc, _ in current_cluster])
                stationary_points.append({
                    'location': center,
                    'start_time': current_cluster[0][1],
                    'end_time': current_cluster[-1][1],
                    'duration_seconds': duration,
                    'point_count': len(current_cluster)
                })
        
        return stationary_points
    
    @staticmethod
    def optimize_route_simple(start: Location, destinations: List[Location], 
                             end: Optional[Location] = None) -> List[Location]:
        """
        Simple route optimization using nearest neighbor algorithm.
        Not optimal but fast for small route sets.
        """
        if not destinations:
            return [start] + ([end] if end else [])
        
        route = [start]
        remaining = destinations.copy()
        current = start
        
        while remaining:
            nearest, _ = GeoUtils.find_nearest_location(current, remaining)
            route.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        if end:
            route.append(end)
        
        return route
    
    @staticmethod
    def calculate_territory_coverage(dealer_locations: List[Location], 
                                   customer_locations: List[Location],
                                   coverage_radius_km: float = 5.0) -> Dict:
        """
        Calculate territory coverage metrics for dealers.
        """
        total_customers = len(customer_locations)
        covered_customers = set()
        dealer_coverage = {}
        
        for i, dealer_loc in enumerate(dealer_locations):
            nearby_customers = GeoUtils.get_locations_within_radius(
                dealer_loc, customer_locations, coverage_radius_km
            )
            
            dealer_coverage[f"dealer_{i}"] = {
                'location': dealer_loc,
                'customers_in_range': len(nearby_customers),
                'customers': [loc for loc, _ in nearby_customers]
            }
            
            # Track which customers are covered
            for customer_loc, _ in nearby_customers:
                covered_customers.add(id(customer_loc))
        
        coverage_percentage = (len(covered_customers) / total_customers * 100) if total_customers > 0 else 0
        
        return {
            'total_customers': total_customers,
            'covered_customers': len(covered_customers),
            'coverage_percentage': coverage_percentage,
            'dealer_coverage': dealer_coverage,
            'uncovered_customers': total_customers - len(covered_customers)
        }
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def geocode_address(address: str) -> Optional[Location]:
        """
        Geocode an address to get coordinates.
        Uses caching to avoid repeated API calls.
        """
        try:
            geolocator = Nominatim(user_agent="sfa_system")
            location = geolocator.geocode(f"{address}, Sri Lanka")
            
            if location:
                return Location(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    address=location.address
                )
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def reverse_geocode(location: Location) -> Optional[str]:
        """Get address from coordinates."""
        try:
            geolocator = Nominatim(user_agent="sfa_system")
            result = geolocator.reverse(location.to_tuple())
            return result.address if result else None
        except Exception:
            return None
    
    @staticmethod
    def validate_gps_accuracy(locations: List[Tuple[Location, float]], 
                             max_speed_kmh: float = 120.0) -> List[bool]:
        """
        Validate GPS points by checking for unrealistic speeds.
        Returns list of booleans indicating valid points.
        """
        if len(locations) < 2:
            return [True] * len(locations)
        
        valid_points = [True]  # First point is always considered valid
        
        for i in range(1, len(locations)):
            prev_loc, prev_time = locations[i-1]
            curr_loc, curr_time = locations[i]
            
            time_diff = abs(curr_time - prev_time)
            if time_diff > 0:
                speed = GeoUtils.calculate_speed(prev_loc, curr_loc, time_diff)
                valid_points.append(speed <= max_speed_kmh)
            else:
                valid_points.append(False)  # Zero time difference is invalid
        
        return valid_points
    
    @staticmethod
    def calculate_area_polygon(locations: List[Location]) -> float:
        """
        Calculate area of polygon formed by locations in square kilometers.
        Uses Shoelace formula.
        """
        if len(locations) < 3:
            return 0.0
        
        # Convert to UTM for more accurate area calculation
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32644', always_xy=True)
        
        utm_coords = []
        for loc in locations:
            x, y = transformer.transform(loc.longitude, loc.latitude)
            utm_coords.append((x, y))
        
        # Shoelace formula
        n = len(utm_coords)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += utm_coords[i][0] * utm_coords[j][1]
            area -= utm_coords[j][0] * utm_coords[i][1]
        
        area = abs(area) / 2.0
        
        # Convert from square meters to square kilometers
        return area / 1_000_000
    
    @staticmethod
    def generate_grid_points(bbox: BoundingBox, grid_size_km: float) -> List[Location]:
        """
        Generate a grid of points within a bounding box.
        Useful for territory analysis and coverage planning.
        """
        points = []
        
        # Calculate approximate degrees per kilometer at this latitude
        lat_center = (bbox.north + bbox.south) / 2
        lat_deg_per_km = 1 / 111.32  # Approximate
        lon_deg_per_km = 1 / (111.32 * math.cos(math.radians(lat_center)))
        
        lat_step = lat_deg_per_km * grid_size_km
        lon_step = lon_deg_per_km * grid_size_km
        
        lat = bbox.south
        while lat <= bbox.north:
            lon = bbox.west
            while lon <= bbox.east:
                points.append(Location(lat, lon, f"Grid_{len(points)}"))
                lon += lon_step
            lat += lat_step
        
        return points
    
    @staticmethod
    def cluster_locations(locations: List[Location], max_distance_km: float = 2.0) -> List[List[Location]]:
        """
        Cluster nearby locations using simple distance-based clustering.
        """
        if not locations:
            return []
        
        clusters = []
        remaining = locations.copy()
        
        while remaining:
            current = remaining.pop(0)
            cluster = [current]
            
            # Find all locations within max_distance of current
            i = 0
            while i < len(remaining):
                if GeoUtils.geodesic_distance(current, remaining[i]) <= max_distance_km:
                    cluster.append(remaining.pop(i))
                else:
                    i += 1
            
            clusters.append(cluster)
        
        return clusters
    
    @staticmethod
    def calculate_visit_efficiency(route: List[Location], visit_times: List[float]) -> Dict:
        """
        Calculate route efficiency metrics.
        """
        if len(route) != len(visit_times) or len(route) < 2:
            return {}
        
        total_distance = GeoUtils.calculate_route_distance(route)
        total_time = sum(visit_times)
        
        # Calculate time between visits
        travel_times = []
        for i in range(len(route) - 1):
            distance = GeoUtils.geodesic_distance(route[i], route[i + 1])
            # Assume average speed of 30 km/h for travel time estimation
            travel_time = (distance / 30) * 60  # minutes
            travel_times.append(travel_time)
        
        total_travel_time = sum(travel_times)
        
        return {
            'total_distance_km': total_distance,
            'total_visit_time_minutes': total_time,
            'total_travel_time_minutes': total_travel_time,
            'efficiency_ratio': total_time / (total_time + total_travel_time) if total_travel_time > 0 else 1.0,
            'visits_per_km': len(route) / total_distance if total_distance > 0 else 0,
            'average_distance_between_visits': total_distance / (len(route) - 1) if len(route) > 1 else 0
        }
    
    @staticmethod
    def find_optimal_depot_location(customer_locations: List[Location]) -> Location:
        """
        Find optimal depot location using geometric median.
        """
        if not customer_locations:
            raise ValueError("Customer locations list cannot be empty")
        
        if len(customer_locations) == 1:
            return customer_locations[0]
        
        # Simple centroid as starting point
        center = GeoUtils.get_route_center(customer_locations)
        
        # Weiszfeld algorithm for geometric median
        for _ in range(100):  # Max iterations
            total_weight = 0
            weighted_lat = 0
            weighted_lon = 0
            
            for loc in customer_locations:
                distance = GeoUtils.geodesic_distance(center, loc)
                if distance > 0:
                    weight = 1 / distance
                    total_weight += weight
                    weighted_lat += weight * loc.latitude
                    weighted_lon += weight * loc.longitude
            
            if total_weight > 0:
                new_lat = weighted_lat / total_weight
                new_lon = weighted_lon / total_weight
                new_center = Location(new_lat, new_lon, "Optimal Depot")
                
                # Check convergence
                if GeoUtils.geodesic_distance(center, new_center) < 0.01:  # 10m tolerance
                    break
                
                center = new_center
        
        return center

# Additional utility functions for SFA-specific operations
class SFAGeoAnalytics:
    """Advanced geospatial analytics for SFA operations."""
    
    @staticmethod
    def analyze_dealer_territory_overlap(dealers: List[Dict]) -> Dict:
        """
        Analyze territory overlaps between dealers.
        dealers: List of dicts with 'location' and 'territory_radius' keys
        """
        overlaps = []
        
        for i, dealer1 in enumerate(dealers):
            for j, dealer2 in enumerate(dealers[i+1:], i+1):
                distance = GeoUtils.geodesic_distance(
                    dealer1['location'], 
                    dealer2['location']
                )
                
                overlap_distance = (dealer1.get('territory_radius', 5) + 
                                  dealer2.get('territory_radius', 5)) - distance
                
                if overlap_distance > 0:
                    overlaps.append({
                        'dealer1_id': i,
                        'dealer2_id': j,
                        'overlap_distance_km': overlap_distance,
                        'distance_between_km': distance
                    })
        
        return {
            'total_overlaps': len(overlaps),
            'overlaps': overlaps,
            'overlap_percentage': len(overlaps) / (len(dealers) * (len(dealers) - 1) / 2) * 100 if len(dealers) > 1 else 0
        }
    
    @staticmethod
    def calculate_sales_density_heatmap(sales_data: List[Dict], grid_size_km: float = 1.0) -> Dict:
        """
        Calculate sales density for heatmap visualization.
        sales_data: List of dicts with 'location' and 'value' keys
        """
        if not sales_data:
            return {}
        
        # Get bounding box of all sales locations
        locations = [data['location'] for data in sales_data]
        bbox = GeoUtils.get_bounding_box(locations)
        
        # Generate grid
        grid_points = GeoUtils.generate_grid_points(bbox, grid_size_km)
        
        # Calculate sales density for each grid point
        heatmap_data = []
        
        for grid_point in grid_points:
            total_sales = 0
            nearby_sales = GeoUtils.get_locations_within_radius(
                grid_point, locations, grid_size_km / 2
            )
            
            for sales_loc, _ in nearby_sales:
                # Find corresponding sales value
                for sales_item in sales_data:
                    if (sales_item['location'].latitude == sales_loc.latitude and
                        sales_item['location'].longitude == sales_loc.longitude):
                        total_sales += sales_item.get('value', 0)
                        break
            
            if total_sales > 0:
                heatmap_data.append({
                    'location': grid_point,
                    'sales_density': total_sales,
                    'nearby_sales_count': len(nearby_sales)
                })
        
        return {
            'heatmap_points': heatmap_data,
            'max_density': max(point['sales_density'] for point in heatmap_data) if heatmap_data else 0,
            'total_grid_points': len(grid_points),
            'active_grid_points': len(heatmap_data)
        }
    
    @staticmethod
    def suggest_new_dealer_locations(existing_dealers: List[Location], 
                                   customers: List[Location],
                                   target_coverage_radius: float = 5.0) -> List[Location]:
        """
        Suggest optimal locations for new dealers based on uncovered customer areas.
        """
        # Find uncovered customers
        uncovered_customers = []
        
        for customer in customers:
            is_covered = False
            for dealer in existing_dealers:
                if GeoUtils.geodesic_distance(customer, dealer) <= target_coverage_radius:
                    is_covered = True
                    break
            
            if not is_covered:
                uncovered_customers.append(customer)
        
        if not uncovered_customers:
            return []
        
        # Cluster uncovered customers
        clusters = GeoUtils.cluster_locations(uncovered_customers, target_coverage_radius * 2)
        
        # Find optimal location for each cluster
        suggested_locations = []
        for cluster in clusters:
            if len(cluster) >= 3:  # Only suggest if cluster has enough customers
                optimal_location = GeoUtils.find_optimal_depot_location(cluster)
                optimal_location.name = f"Suggested_Dealer_{len(suggested_locations) + 1}"
                suggested_locations.append(optimal_location)
        
        return suggested_locations
            