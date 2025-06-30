# import numpy as np
# import pandas as pd
# from typing import List, Tuple, Dict, Any
# from scipy.spatial.distance import pdist, squareform
# from scipy.optimize import minimize
# import networkx as nx
# from geopy.distance import geodesic
# import random
# from dataclasses import dataclass

# @dataclass
# class Location:
#     lat: float
#     lng: float
#     name: str = ""
#     priority: int = 1
#     time_window: Tuple[int, int] = (0, 24)  # Hours
#     service_time: int = 30  # Minutes

# class RouteOptimizer:
#     def __init__(self):
#         self.distance_matrix = None
#         self.locations = []
        
#     def calculate_distance_matrix(self, locations: List[Location]) -> np.ndarray:
#         """Calculate distance matrix between all locations"""
#         n = len(locations)
#         distance_matrix = np.zeros((n, n))
        
#         for i in range(n):
#             for j in range(n):
#                 if i != j:
#                     coord1 = (locations[i].lat, locations[i].lng)
#                     coord2 = (locations[j].lat, locations[j].lng)
#                     distance_matrix[i][j] = geodesic(coord1, coord2).kilometers
        
#         return distance_matrix
    
#     def genetic_algorithm_tsp(self, distance_matrix: np.ndarray, 
#                             population_size: int = 100, 
#                             generations: int = 500,
#                             mutation_rate: float = 0.02) -> List[int]:
#         """Solve TSP using Genetic Algorithm"""
        
#         n_cities = len(distance_matrix)
        
#         def create_individual():
#             """Create random tour"""
#             return random.sample(range(1, n_cities), n_cities - 1)
        
#         def fitness(individual):
#             """Calculate fitness (negative total distance)"""
#             total_distance = distance_matrix[0][individual[0]]
#             for i in range(len(individual) - 1):
#                 total_distance += distance_matrix[individual[i]][individual[i + 1]]
#             total_distance += distance_matrix[individual[-1]][0]  # Return to start
#             return -total_distance
        
#         def crossover(parent1, parent2):
#             """Order crossover (OX)"""
#             size = len(parent1)
#             start, end = sorted(random.sample(range(size), 2))
            
#             child = [-1] * size
#             child[start:end] = parent1[start:end]
            
#             pointer = end
#             for city in parent2[end:] + parent2[:end]:
#                 if city not in child:
#                     child[pointer % size] = city
#                     pointer += 1
            
#             return child
        
#         def mutate(individual):
#             """Swap mutation"""
#             if random.random() < mutation_rate:
#                 i, j = random.sample(range(len(individual)), 2)
#                 individual[i], individual[j] = individual[j], individual[i]
#             return individual
        
#         # Initialize population
#         population = [create_individual() for _ in range(population_size)]
        
#         best_individual = None
#         best_fitness = float('-inf')
        
#         for generation in range(generations):
#             # Evaluate fitness
#             fitness_scores = [fitness(ind) for ind in population]
            
#             # Track best solution
#             current_best_idx = np.argmax(fitness_scores)
#             if fitness_scores[current_best_idx] > best_fitness:
#                 best_fitness = fitness_scores[current_best_idx]
#                 best_individual = population[current_best_idx].copy()
            
#             # Selection (tournament selection)
#             new_population = []
#             for _ in range(population_size):
#                 tournament = random.sample(list(zip(population, fitness_scores)), 3)
#                 winner = max(tournament, key=lambda x: x[1])[0]
#                 new_population.append(winner)
            
#             # Crossover and mutation
#             for i in range(0, population_size - 1, 2):
#                 if random.random() < 0.8:  # Crossover probability
#                     child1 = crossover(new_population[i], new_population[i + 1])
#                     child2 = crossover(new_population[i + 1], new_population[i])
#                     new_population[i] = mutate(child1)
#                     new_population[i + 1] = mutate(child2)
            
#             population = new_population
        
#         return [0] + best_individual + [0]  # Add start and end depot
    
#     def optimize_route_with_constraints(self, locations: List[Location], 
#                                       start_time: int = 8,  # 8 AM
#                                       max_working_hours: int = 10) -> Dict[str, Any]:
#         """Optimize route considering time windows and constraints"""
        
#         self.locations = locations
#         distance_matrix = self.calculate_distance_matrix(locations)
        
#         # Solve basic TSP
#         basic_route = self.genetic_algorithm_tsp(distance_matrix)
        
#         # Apply time window constraints
#         optimized_route, schedule = self._apply_time_constraints(
#             basic_route, locations, distance_matrix, start_time, max_working_hours
#         )
        
#         # Calculate route metrics
#         total_distance = self._calculate_route_distance(optimized_route, distance_matrix)
#         total_time = schedule[-1]['arrival_time'] - start_time if schedule else 0
        
#         return {
#             'route': optimized_route,
#             'schedule': schedule,
#             'total_distance': total_distance,
#             'total_time': total_time,
#             'locations': [locations[i] for i in optimized_route]
#         }
    
#     def _apply_time_constraints(self, route: List[int], locations: List[Location],
#                               distance_matrix: np.ndarray, start_time: int,
#                               max_working_hours: int) -> Tuple[List[int], List[Dict]]:
#         """Apply time window constraints to route"""
        
#         schedule = []
#         current_time = start_time
#         optimized_route = [route[0]]  # Start with depot
        
#         for i in range(1, len(route) - 1):
#             current_loc_idx = route[i-1]
#             next_loc_idx = route[i]
            
#             # Calculate travel time (assume 50 km/h average speed)
#             travel_time = distance_matrix[current_loc_idx][next_loc_idx] / 50  # hours
#             arrival_time = current_time + travel_time
            
#             location = locations[next_loc_idx]
            
#             # Check if arrival is within time window
#             if arrival_time < location.time_window[0]:
#                 # Wait until time window opens
#                 arrival_time = location.time_window[0]
#             elif arrival_time > location.time_window[1]:
#                 # Skip this location if too late
#                 continue
            
#             # Check if we exceed max working hours
#             service_time = location.service_time / 60  # Convert to hours
#             if arrival_time + service_time - start_time > max_working_hours:
#                 break
            
#             optimized_route.append(next_loc_idx)
#             schedule.append({
#                 'location_index': next_loc_idx,
#                 'location_name': location.name,
#                 'arrival_time': arrival_time,
#                 'departure_time': arrival_time + service_time,
#                 'service_time': service_time
#             })
            
#             current_time = arrival_time + service_time
        
#         optimized_route.append(route[-1])  # Add return to depot
        
#         return optimized_route, schedule
    
#     def _calculate_route_distance(self, route: List[int], distance_matrix: np.ndarray) -> float:
#         """Calculate total distance of route"""
#         total_distance = 0
#         for i in range(len(route) - 1):
#             total_distance += distance_matrix[route[i]][route[i + 1]]
#         return total_distance
    
#     def multi_objective_optimization(self, locations: List[Location],
#                                    objectives: Dict[str, float] = None) -> Dict[str, Any]:
#         """Multi-objective route optimization"""
        
#         if objectives is None:
#             objectives = {
#                 'distance': 0.4,
#                 'time': 0.3,
#                 'priority': 0.2,
#                 'fuel_cost': 0.1
#             }
        
#         distance_matrix = self.calculate_distance_matrix(locations)
        
#         def multi_objective_fitness(route):
#             """Calculate multi-objective fitness score"""
            
#             # Distance objective
#             total_distance = self._calculate_route_distance(route, distance_matrix)
#             distance_score = -total_distance * objectives['distance']
            
#             # Time objective (minimize total time)
#             total_time = self._estimate_total_time(route, distance_matrix, locations)
#             time_score = -total_time * objectives['time']
            
#             # Priority objective (visit high priority locations first)
#             priority_score = self._calculate_priority_score(route, locations) * objectives['priority']
            
#             # Fuel cost objective
#             fuel_cost = total_distance * 0.08  # Assume 0.08 cost per km
#             fuel_score = -fuel_cost * objectives['fuel_cost']
            
#             return distance_score + time_score + priority_score + fuel_score
        
#         # Use genetic algorithm with multi-objective fitness
#         best_route = self._multi_objective_ga(distance_matrix, multi_objective_fitness)
        
#         # Calculate final metrics
#         total_distance = self._calculate_route_distance(best_route, distance_matrix)
#         total_time = self._estimate_total_time(best_route, distance_matrix, locations)
#         fuel_cost = total_distance * 0.08
        
#         return {
#             'route': best_route,
#             'total_distance': total_distance,
#             'total_time': total_time,
#             'fuel_cost': fuel_cost,
#             'locations': [locations[i] for i in best_route],
#             'objectives_used': objectives
#         }
    
#     def _estimate_total_time(self, route: List[int], distance_matrix: np.ndarray,
#                            locations: List[Location]) -> float:
#         """Estimate total time for route including service times"""
        
#         travel_time = self._calculate_route_distance(route, distance_matrix) / 50  # 50 km/h
#         service_time = sum(locations[i].service_time for i in route[1:-1]) / 60  # Convert to hours
        
#         return travel_time + service_time
    
#     def _calculate_priority_score(self, route: List[int], locations: List[Location]) -> float:
#         """Calculate priority score (higher priority locations visited earlier get higher score)"""
        
#         score = 0
#         for i, loc_idx in enumerate(route[1:-1]):  # Exclude start and end depot
#             priority = locations[loc_idx].priority
#             position_weight = 1 / (i + 1)  # Earlier positions get higher weight
#             score += priority * position_weight
        
#         return score
    
#     def _multi_objective_ga(self, distance_matrix: np.ndarray, fitness_func) -> List[int]:
#         """Genetic algorithm with multi-objective fitness"""
        
#         population_size = 100
#         generations = 300
#         mutation_rate = 0.02
#         n_cities = len(distance_matrix)
        
#         def create_individual():
#             return random.sample(range(1, n_cities), n_cities - 1)
        
#         def crossover(parent1, parent2):
#             size = len(parent1)
#             start, end = sorted(random.sample(range(size), 2))
            
#             child = [-1] * size
#             child[start:end] = parent1[start:end]
            
#             pointer = end
#             for city in parent2[end:] + parent2[:end]:
#                 if city not in child:
#                     child[pointer % size] = city
#                     pointer += 1
            
#             return child
        
#         def mutate(individual):
#             if random.random() < mutation_rate:
#                 i, j = random.sample(range(len(individual)), 2)
#                 individual[i], individual[j] = individual[j], individual[i]
#             return individual
        
#         # Initialize population
#         population = [create_individual() for _ in range(population_size)]
        
#         best_individual = None
#         best_fitness = float('-inf')
        
#         for generation in range(generations):
#             # Evaluate fitness for each individual
#             fitness_scores = []
#             for individual in population:
#                 route = [0] + individual + [0]
#                 fitness_scores.append(fitness_func(route))
            
#             # Track best solution
#             current_best_idx = np.argmax(fitness_scores)
#             if fitness_scores[current_best_idx] > best_fitness:
#                 best_fitness = fitness_scores[current_best_idx]
#                 best_individual = population[current_best_idx].copy()
            
#             # Selection, crossover, and mutation
#             new_population = []
#             for _ in range(population_size):
#                 # Tournament selection
#                 tournament = random.sample(list(zip(population, fitness_scores)), 3)
#                 winner = max(tournament, key=lambda x: x[1])[0]
#                 new_population.append(winner)
            
#             # Apply crossover and mutation
#             for i in range(0, population_size - 1, 2):
#                 if random.random() < 0.8:
#                     child1 = crossover(new_population[i], new_population[i + 1])
#                     child2 = crossover(new_population[i + 1], new_population[i])
#                     new_population[i] = mutate(child1)
#                     new_population[i + 1] = mutate(child2)
            
#             population = new_population
        
#         return [0] + best_individual + [0]



"""
Advanced Route Optimization Module
Implements multiple algorithms for optimal route planning including:
- Traveling Salesman Problem (TSP) solutions
- Vehicle Routing Problem (VRP) with constraints
- Multi-objective optimization considering time, distance, and sales potential
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import networkx as nx
from sklearn.cluster import KMeans
import random
import math
import logging
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

@dataclass
class Location:
    """Represents a customer/dealer location"""
    id: str
    name: str
    latitude: float
    longitude: float
    priority: int = 1
    service_time: int = 30  # minutes
    time_window_start: Optional[int] = None  # minutes from start of day
    time_window_end: Optional[int] = None
    sales_potential: float = 0.0
    last_visit_days: int = 0

@dataclass
class RouteResult:
    """Route optimization result"""
    route: List[str]  # sequence of location IDs
    total_distance: float
    total_time: int
    estimated_sales: float
    efficiency_score: float
    route_coordinates: List[Tuple[float, float]]

class HaversineCalculator:
    """Calculate distances using Haversine formula"""
    
    @staticmethod
    def distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c

class TSPSolver:
    """Multiple TSP solving algorithms"""
    
    def __init__(self, locations: List[Location]):
        self.locations = locations
        self.distance_matrix = self._calculate_distance_matrix()
        
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate distance matrix between all locations"""
        n = len(self.locations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = HaversineCalculator.distance(
                        self.locations[i].latitude, self.locations[i].longitude,
                        self.locations[j].latitude, self.locations[j].longitude
                    )
        return matrix
    
    def nearest_neighbor(self, start_idx: int = 0) -> Tuple[List[int], float]:
        """Nearest neighbor heuristic"""
        n = len(self.locations)
        unvisited = set(range(n))
        current = start_idx
        route = [current]
        unvisited.remove(current)
        total_distance = 0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.distance_matrix[current][x])
            total_distance += self.distance_matrix[current][nearest]
            current = nearest
            route.append(current)
            unvisited.remove(current)
        
        # Return to start
        total_distance += self.distance_matrix[current][start_idx]
        route.append(start_idx)
        
        return route, total_distance
    
    def two_opt_improvement(self, route: List[int]) -> Tuple[List[int], float]:
        """2-opt local search improvement"""
        best_route = route[:]
        best_distance = self._calculate_route_distance(route)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1: continue  # Skip adjacent edges
                    
                    new_route = route[:]
                    new_route[i:j] = route[i:j][::-1]  # Reverse the segment
                    
                    new_distance = self._calculate_route_distance(new_route)
                    if new_distance < best_distance:
                        best_route = new_route[:]
                        best_distance = new_distance
                        route = new_route[:]
                        improved = True
                        break
                if improved:
                    break
        
        return best_route, best_distance
    
    def _calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance for a route"""
        total = 0
        for i in range(len(route) - 1):
            total += self.distance_matrix[route[i]][route[i + 1]]
        return total

class GeneticAlgorithmTSP:
    """Genetic Algorithm for TSP"""
    
    def __init__(self, locations: List[Location], population_size: int = 100, 
                 generations: int = 500, mutation_rate: float = 0.02):
        self.locations = locations
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.distance_matrix = self._calculate_distance_matrix()
        
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate distance matrix"""
        n = len(self.locations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = HaversineCalculator.distance(
                        self.locations[i].latitude, self.locations[i].longitude,
                        self.locations[j].latitude, self.locations[j].longitude
                    )
        return matrix
    
    def _fitness(self, route: List[int]) -> float:
        """Calculate fitness (inverse of total distance)"""
        total_distance = 0
        for i in range(len(route)):
            j = (i + 1) % len(route)
            total_distance += self.distance_matrix[route[i]][route[j]]
        return 1 / (total_distance + 1e-10)
    
    def _create_individual(self) -> List[int]:
        """Create random route"""
        route = list(range(len(self.locations)))
        random.shuffle(route)
        return route
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover (OX)"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [-1] * size
        child[start:end] = parent1[start:end]
        
        pointer = end
        for city in parent2[end:] + parent2[:end]:
            if city not in child:
                child[pointer] = city
                pointer = (pointer + 1) % size
        
        return child
    
    def _mutate(self, route: List[int]) -> List[int]:
        """Swap mutation"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route
    
    def solve(self) -> Tuple[List[int], float]:
        """Solve TSP using genetic algorithm"""
        # Initialize population
        population = [self._create_individual() for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            # Calculate fitness for all individuals
            fitness_scores = [self._fitness(individual) for individual in population]
            
            # Selection (tournament selection)
            new_population = []
            for _ in range(self.population_size):
                tournament_size = 5
                tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
                winner = max(tournament, key=lambda x: x[1])[0]
                new_population.append(winner[:])
            
            # Crossover and mutation
            for i in range(0, self.population_size - 1, 2):
                if random.random() < 0.8:  # Crossover probability
                    child1 = self._crossover(new_population[i], new_population[i + 1])
                    child2 = self._crossover(new_population[i + 1], new_population[i])
                    new_population[i] = self._mutate(child1)
                    new_population[i + 1] = self._mutate(child2)
            
            population = new_population
            
            if generation % 50 == 0:
                best_fitness = max(fitness_scores)
                best_distance = 1 / best_fitness
                logger.info(f"Generation {generation}: Best distance = {best_distance:.2f}")
        
        # Return best solution
        final_fitness = [self._fitness(individual) for individual in population]
        best_idx = np.argmax(final_fitness)
        best_route = population[best_idx]
        best_distance = 1 / final_fitness[best_idx]
        
        return best_route, best_distance

class VehicleRoutingProblem:
    """Vehicle Routing Problem solver with multiple constraints"""
    
    def __init__(self, locations: List[Location], depot_location: Location,
                 vehicle_capacity: int = 1000, max_route_time: int = 480):
        self.locations = locations
        self.depot = depot_location
        self.vehicle_capacity = vehicle_capacity
        self.max_route_time = max_route_time
        self.all_locations = [depot_location] + locations
        self.distance_matrix = self._calculate_distance_matrix()
        
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate distance matrix including depot"""
        n = len(self.all_locations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = HaversineCalculator.distance(
                        self.all_locations[i].latitude, self.all_locations[i].longitude,
                        self.all_locations[j].latitude, self.all_locations[j].longitude
                    )
        return matrix
    
    def solve_clarke_wright(self) -> List[List[int]]:
        """Clarke-Wright savings algorithm for VRP"""
        n = len(self.locations)
        
        # Calculate savings
        savings = []
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                saving = (self.distance_matrix[0][i] + self.distance_matrix[0][j] - 
                         self.distance_matrix[i][j])
                savings.append((saving, i, j))
        
        # Sort by savings (descending)
        savings.sort(reverse=True)
        
        # Initialize routes (each customer in separate route)
        routes = [[i] for i in range(1, n + 1)]
        route_demands = [self.locations[i-1].priority for i in range(1, n + 1)]
        route_times = [self.locations[i-1].service_time for i in range(1, n + 1)]
        
        # Merge routes based on savings
        for saving, i, j in savings:
            # Find routes containing i and j
            route_i = route_j = None
            idx_i = idx_j = None
            
            for idx, route in enumerate(routes):
                if i in route:
                    route_i = route
                    idx_i = idx
                if j in route:
                    route_j = route
                    idx_j = idx
            
            # Check if routes can be merged
            if (route_i != route_j and route_i is not None and route_j is not None):
                # Check capacity and time constraints
                new_demand = route_demands[idx_i] + route_demands[idx_j]
                new_time = route_times[idx_i] + route_times[idx_j]
                
                if (new_demand <= self.vehicle_capacity and new_time <= self.max_route_time):
                    # Merge routes
                    if route_i[-1] == i and route_j[0] == j:
                        new_route = route_i + route_j
                    elif route_i[0] == i and route_j[-1] == j:
                        new_route = route_j + route_i
                    elif route_i[-1] == i and route_j[-1] == j:
                        new_route = route_i + route_j[::-1]
                    elif route_i[0] == i and route_j[0] == j:
                        new_route = route_i[::-1] + route_j
                    else:
                        continue
                    
                    # Update routes
                    routes.remove(route_i)
                    routes.remove(route_j)
                    routes.append(new_route)
                    
                    route_demands = [sum(self.locations[k-1].priority for k in route) 
                                   for route in routes]
                    route_times = [sum(self.locations[k-1].service_time for k in route) + 
                                 self._calculate_route_time(route) for route in routes]
        
        return routes
    
    def _calculate_route_time(self, route: List[int]) -> int:
        """Calculate travel time for a route"""
        total_time = 0
        # Add travel time from depot to first location
        if route:
            total_time += self.distance_matrix[0][route[0]] * 2  # Assume 2 minutes per km
            
            # Add travel time between locations
            for i in range(len(route) - 1):
                total_time += self.distance_matrix[route[i]][route[i + 1]] * 2
            
            # Add travel time from last location back to depot
            total_time += self.distance_matrix[route[-1]][0] * 2
        
        return int(total_time)

class MultiObjectiveRouteOptimizer:
    """Multi-objective route optimization considering distance, time, and sales potential"""
    
    def __init__(self, locations: List[Location], depot_location: Location):
        self.locations = locations
        self.depot = depot_location
        self.tsp_solver = TSPSolver(locations)
        self.ga_solver = GeneticAlgorithmTSP(locations)
        self.vrp_solver = VehicleRoutingProblem(locations, depot_location)
        
    def optimize_single_route(self, method: str = "genetic") -> RouteResult:
        """Optimize single route using specified method"""
        if method == "nearest_neighbor":
            route_indices, distance = self.tsp_solver.nearest_neighbor()
            # Apply 2-opt improvement
            route_indices, distance = self.tsp_solver.two_opt_improvement(route_indices)
        elif method == "genetic":
            route_indices, distance = self.ga_solver.solve()
        else:
            raise ValueError("Method must be 'nearest_neighbor' or 'genetic'")
        
        # Convert indices to location IDs
        route_ids = [self.locations[i].id for i in route_indices[:-1]]  # Exclude return to start
        route_coordinates = [(self.locations[i].latitude, self.locations[i].longitude) 
                           for i in route_indices[:-1]]
        
        # Calculate additional metrics
        total_time = sum(self.locations[i].service_time for i in route_indices[:-1])
        total_time += distance * 2  # Travel time (2 minutes per km)
        
        estimated_sales = sum(self.locations[i].sales_potential for i in route_indices[:-1])
        efficiency_score = estimated_sales / (distance + 1e-10)
        
        return RouteResult(
            route=route_ids,
            total_distance=distance,
            total_time=int(total_time),
            estimated_sales=estimated_sales,
            efficiency_score=efficiency_score,
            route_coordinates=route_coordinates
        )
    
    def optimize_multiple_routes(self, max_routes: int = 5) -> List[RouteResult]:
        """Optimize multiple routes using VRP"""
        routes = self.vrp_solver.solve_clarke_wright()
        
        results = []
        for route_indices in routes[:max_routes]:
            # Convert to location objects for TSP optimization
            route_locations = [self.locations[i-1] for i in route_indices]
            
            if len(route_locations) > 1:
                # Optimize individual route
                route_tsp = TSPSolver(route_locations)
                optimized_indices, distance = route_tsp.nearest_neighbor()
                optimized_indices, distance = route_tsp.two_opt_improvement(optimized_indices)
                
                # Convert back to original indices
                route_ids = [route_locations[i].id for i in optimized_indices[:-1]]
                route_coordinates = [(route_locations[i].latitude, route_locations[i].longitude) 
                                   for i in optimized_indices[:-1]]
                
                total_time = sum(route_locations[i].service_time for i in optimized_indices[:-1])
                total_time += distance * 2
                
                estimated_sales = sum(route_locations[i].sales_potential for i in optimized_indices[:-1])
                efficiency_score = estimated_sales / (distance + 1e-10)
                
                results.append(RouteResult(
                    route=route_ids,
                    total_distance=distance,
                    total_time=int(total_time),
                    estimated_sales=estimated_sales,
                    efficiency_score=efficiency_score,
                    route_coordinates=route_coordinates
                ))
        
        return results
    
    def optimize_with_priorities(self, priority_weight: float = 0.3, 
                               distance_weight: float = 0.4, 
                               sales_weight: float = 0.3) -> RouteResult:
        """Multi-objective optimization with weighted priorities"""
        
        def objective_function(route_indices: List[int]) -> float:
            """Calculate weighted objective score"""
            distance = self.tsp_solver._calculate_route_distance(route_indices)
            
            priority_score = sum(self.locations[i].priority for i in route_indices[:-1])
            sales_score = sum(self.locations[i].sales_potential for i in route_indices[:-1])
            
            # Normalize scores (higher is better, distance is minimized)
            normalized_distance = 1 / (distance + 1e-10)
            normalized_priority = priority_score / len(route_indices)
            normalized_sales = sales_score
            
            return (distance_weight * normalized_distance + 
                   priority_weight * normalized_priority + 
                   sales_weight * normalized_sales)
        
        # Use genetic algorithm with custom fitness function
        ga = GeneticAlgorithmTSP(self.locations)
        ga._fitness = lambda route: objective_function(route)
        
        route_indices, _ = ga.solve()
        
        # Calculate final metrics
        distance = self.tsp_solver._calculate_route_distance(route_indices)
        route_ids = [self.locations[i].id for i in route_indices[:-1]]
        route_coordinates = [(self.locations[i].latitude, self.locations[i].longitude) 
                           for i in route_indices[:-1]]
        
        total_time = sum(self.locations[i].service_time for i in route_indices[:-1])
        total_time += distance * 2
        
        estimated_sales = sum(self.locations[i].sales_potential for i in route_indices[:-1])
        efficiency_score = estimated_sales / (distance + 1e-10)
        
        return RouteResult(
            route=route_ids,
            total_distance=distance,
            total_time=int(total_time),
            estimated_sales=estimated_sales,
            efficiency_score=efficiency_score,
            route_coordinates=route_coordinates
        )

class RouteOptimizer:
    """Main route optimizer class"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_locations_from_dataframe(self, df: pd.DataFrame, 
                                    lat_col: str = 'Latitude', 
                                    lon_col: str = 'Longitude',
                                    id_col: str = 'No.',
                                    name_col: str = 'City') -> List[Location]:
        """Load locations from pandas DataFrame"""
        locations = []
        
        for _, row in df.iterrows():
            location = Location(
                id=str(row[id_col]),
                name=str(row[name_col]),
                latitude=float(row[lat_col]),
                longitude=float(row[lon_col]),
                priority=1,
                service_time=30,
                sales_potential=random.uniform(1000, 10000)  # Random sales potential
            )
            locations.append(location)
        
        return locations
    
    def optimize_routes(self, locations: List[Location], 
                       depot_location: Location,
                       optimization_type: str = "single",
                       method: str = "genetic") -> List[RouteResult]:
        """Main optimization method"""
        optimizer = MultiObjectiveRouteOptimizer(locations, depot_location)
        
        if optimization_type == "single":
            result = optimizer.optimize_single_route(method=method)
            return [result]
        elif optimization_type == "multiple":
            return optimizer.optimize_multiple_routes()
        elif optimization_type == "priority":
            result = optimizer.optimize_with_priorities()
            return [result]
        else:
            raise ValueError("optimization_type must be 'single', 'multiple', or 'priority'")
    
    def export_routes_to_json(self, routes: List[RouteResult], filename: str):
        """Export routes to JSON file"""
        route_data = []
        for i, route in enumerate(routes):
            route_data.append({
                'route_id': i + 1,
                'locations': route.route,
                'coordinates': route.route_coordinates,
                'total_distance_km': round(route.total_distance, 2),
                'total_time_minutes': route.total_time,
                'estimated_sales': round(route.estimated_sales, 2),
                'efficiency_score': round(route.efficiency_score, 4)
            })
        
        with open(filename, 'w') as f:
            json.dump(route_data, f, indent=2)
        
        self.logger.info(f"Routes exported to {filename}")

# Example usage
if __name__ == "__main__":
    # Sample locations for testing
    sample_locations = [
        Location("1", "Colombo", 6.9271, 79.8612, sales_potential=5000),
        Location("2", "Kandy", 7.2906, 80.6337, sales_potential=3000),
        Location("3", "Galle", 6.0535, 80.2210, sales_potential=2000),
        Location("4", "Jaffna", 9.6615, 80.0255, sales_potential=4000),
        Location("5", "Negombo", 7.2084, 79.8366, sales_potential=2500),
    ]
    
    depot = Location("0", "Head Office", 6.9271, 79.8612)
    
    # Initialize optimizer
    optimizer = RouteOptimizer()
    
    # Optimize routes
    routes = optimizer.optimize_routes(
        locations=sample_locations,
        depot_location=depot,
        optimization_type="single",
        method="genetic"
    )
    
    # Print results
    for i, route in enumerate(routes):
        print(f"Route {i+1}:")
        print(f"  Locations: {' -> '.join(route.route)}")
        print(f"  Distance: {route.total_distance:.2f} km")
        print(f"  Time: {route.total_time} minutes")
        print(f"  Estimated Sales: ${route.estimated_sales:.2f}")
        print(f"  Efficiency Score: {route.efficiency_score:.4f}")
        print()