"""
Territory Boundary Optimization Module
Implements intelligent territory segmentation and boundary optimization for sales force automation.
Features:
- Clustering-based territory creation
- Balanced workload distribution
- Geographic constraints consideration
- Multi-objective optimization for territory boundaries
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import networkx as nx
import random
import math
import logging
from concurrent.futures import ThreadPoolExecutor
import json
import pickle

logger = logging.getLogger(__name__)

@dataclass
class Customer:
    """Customer data structure"""
    id: str
    name: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    sales_volume: float = 0.0
    sales_potential: float = 0.0
    visit_frequency: int = 1  # visits per month
    priority: int = 1  # 1=low, 2=medium, 3=high
    territory_id: Optional[str] = None
    last_visit_days: int = 0
    city: str = ""
    
@dataclass
class Dealer:
    """Dealer/Sales representative data structure"""
    id: str
    name: str
    home_latitude: float
    home_longitude: float
    capacity: int = 100  # customers they can handle
    skills: List[str] = field(default_factory=list)
    experience_level: int = 1  # 1=junior, 2=senior, 3=expert
    territory_id: Optional[str] = None
    current_customers: List[str] = field(default_factory=list)

@dataclass
class Territory:
    """Territory data structure"""
    id: str
    name: str
    dealer_id: str
    customers: List[str] = field(default_factory=list)
    boundary_points: List[Tuple[float, float]] = field(default_factory=list)
    center_latitude: float = 0.0
    center_longitude: float = 0.0
    total_sales_volume: float = 0.0
    total_sales_potential: float = 0.0
    estimated_travel_time: int = 0  # minutes per day
    workload_score: float = 0.0

class TerritoryMetrics:
    """Calculate territory optimization metrics"""
    
    @staticmethod
    def calculate_balance_score(territories: List[Territory]) -> float:
        """Calculate workload balance across territories"""
        if not territories:
            return 0.0
        
        workloads = [t.workload_score for t in territories]
        mean_workload = np.mean(workloads)
        
        if mean_workload == 0:
            return 1.0
        
        variance = np.var(workloads)
        balance_score = 1.0 / (1.0 + variance / (mean_workload ** 2))
        return balance_score
    
    @staticmethod
    def calculate_compactness_score(territories: List[Territory], 
                                  customers: Dict[str, Customer]) -> float:
        """Calculate territory compactness (lower travel distances)"""
        total_compactness = 0.0
        
        for territory in territories:
            if len(territory.customers) < 2:
                continue
            
            # Calculate average distance from territory center to customers
            center_lat, center_lon = territory.center_latitude, territory.center_longitude
            distances = []
            
            for customer_id in territory.customers:
                customer = customers[customer_id]
                dist = TerritoryOptimizer._haversine_distance(
                    center_lat, center_lon, customer.latitude, customer.longitude
                )
                distances.append(dist)
            
            avg_distance = np.mean(distances)
            compactness = 1.0 / (1.0 + avg_distance)
            total_compactness += compactness
        
        return total_compactness / len(territories) if territories else 0.0
    
    @staticmethod
    def calculate_coverage_score(territories: List[Territory], 
                               customers: Dict[str, Customer]) -> float:
        """Calculate customer coverage completeness"""
        total_customers = len(customers)
        covered_customers = sum(len(t.customers) for t in territories)
        
        return covered_customers / total_customers if total_customers > 0 else 0.0

class TerritoryOptimizer:
    """Main territory optimization engine"""
    
    def __init__(self, customers: List[Customer], dealers: List[Dealer]):
        self.customers = {c.id: c for c in customers}
        self.dealers = {d.id: d for d in dealers}
        self.territories: Dict[str, Territory] = {}
        self.logger = logging.getLogger(__name__)
        
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    def calculate_workload_score(self, customers: List[str], dealer: Dealer) -> float:
        """Calculate workload score for a dealer based on assigned customers"""
        if not customers:
            return 0.0
        
        total_score = 0.0
        
        for customer_id in customers:
            customer = self.customers[customer_id]
            
            # Base score from sales volume and potential
            volume_score = customer.sales_volume * 0.3
            potential_score = customer.sales_potential * 0.3
            
            # Priority and frequency multipliers
            priority_multiplier = customer.priority * 0.2
            frequency_multiplier = customer.visit_frequency * 0.2
            
            customer_score = (volume_score + potential_score) * (1 + priority_multiplier + frequency_multiplier)
            total_score += customer_score
        
        # Adjust for dealer capacity and experience
        capacity_factor = len(customers) / dealer.capacity if dealer.capacity > 0 else 1.0
        experience_factor = 1.0 / dealer.experience_level
        
        final_score = total_score * capacity_factor * experience_factor
        return final_score
    
    def calculate_territory_center(self, customer_ids: List[str]) -> Tuple[float, float]:
        """Calculate geographic center of territory customers"""
        if not customer_ids:
            return 0.0, 0.0
        
        total_lat = sum(self.customers[cid].latitude for cid in customer_ids)
        total_lon = sum(self.customers[cid].longitude for cid in customer_ids)
        
        center_lat = total_lat / len(customer_ids)
        center_lon = total_lon / len(customer_ids)
        
        return center_lat, center_lon
    
    def estimate_travel_time(self, dealer: Dealer, customer_ids: List[str]) -> int:
        """Estimate daily travel time for dealer to visit customers"""
        if not customer_ids:
            return 0
        
        total_distance = 0.0
        prev_lat, prev_lon = dealer.home_latitude, dealer.home_longitude
        
        # Calculate route through all customers
        for customer_id in customer_ids:
            customer = self.customers[customer_id]
            distance = self._haversine_distance(
                prev_lat, prev_lon, customer.latitude, customer.longitude
            )
            total_distance += distance
            prev_lat, prev_lon = customer.latitude, customer.longitude
        
        # Return to home
        total_distance += self._haversine_distance(
            prev_lat, prev_lon, dealer.home_latitude, dealer.home_longitude
        )
        
        # Estimate time: 40 km/h average speed + 30 min per customer visit
        travel_time = (total_distance / 40) * 60  # minutes
        visit_time = len(customer_ids) * 30  # minutes per customer
        
        return int(travel_time + visit_time)
    
    def create_territories_kmeans(self, n_territories: int = None) -> Dict[str, Territory]:
        """Create territories using K-means clustering"""
        if n_territories is None:
            n_territories = len(self.dealers)
        
        # Prepare customer data for clustering
        customer_data = []
        customer_ids = []
        
        for customer_id, customer in self.customers.items():
            customer_data.append([
                customer.latitude,
                customer.longitude,
                customer.sales_volume,
                customer.sales_potential,
                customer.priority
            ])
            customer_ids.append(customer_id)
        
        if not customer_data:
            return {}
        
        # Standardize features
        scaler = StandardScaler()
        customer_data_scaled = scaler.fit_transform(customer_data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_territories, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(customer_data_scaled)
        
        # Create territories
        territories = {}
        dealer_ids = list(self.dealers.keys())
        
        for cluster_id in range(n_territories):
            territory_id = f"territory_{cluster_id + 1}"
            dealer_id = dealer_ids[cluster_id] if cluster_id < len(dealer_ids) else dealer_ids[0]
            
            # Get customers in this cluster
            cluster_customers = [customer_ids[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            # Calculate territory metrics
            center_lat, center_lon = self.calculate_territory_center(cluster_customers)
            total_sales_volume = sum(self.customers[cid].sales_volume for cid in cluster_customers)
            total_sales_potential = sum(self.customers[cid].sales_potential for cid in cluster_customers)
            
            dealer = self.dealers[dealer_id]
            workload_score = self.calculate_workload_score(cluster_customers, dealer)
            travel_time = self.estimate_travel_time(dealer, cluster_customers)
            
            territory = Territory(
                id=territory_id,
                name=f"Territory {cluster_id + 1}",
                dealer_id=dealer_id,
                customers=cluster_customers,
                center_latitude=center_lat,
                center_longitude=center_lon,
                total_sales_volume=total_sales_volume,
                total_sales_potential=total_sales_potential,
                estimated_travel_time=travel_time,
                workload_score=workload_score
            )
            
            territories[territory_id] = territory
            
            # Update customer territory assignments
            for customer_id in cluster_customers:
                self.customers[customer_id].territory_id = territory_id
        
        self.territories = territories
        return territories
    
    def create_territories_hierarchical(self, n_territories: int = None) -> Dict[str, Territory]:
        """Create territories using hierarchical clustering"""
        if n_territories is None:
            n_territories = len(self.dealers)
        
        # Prepare customer data
        customer_data = []
        customer_ids = []
        
        for customer_id, customer in self.customers.items():
            customer_data.append([
                customer.latitude,
                customer.longitude,
                customer.sales_volume,
                customer.sales_potential
            ])
            customer_ids.append(customer_id)
        
        if not customer_data:
            return {}
        
        # Standardize features
        scaler = StandardScaler()
        customer_data_scaled = scaler.fit_transform(customer_data)
        
        # Perform hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_territories, linkage='ward')
        cluster_labels = hierarchical.fit_predict(customer_data_scaled)
        
        # Create territories similar to K-means
        territories = {}
        dealer_ids = list(self.dealers.keys())
        
        for cluster_id in range(n_territories):
            territory_id = f"territory_h_{cluster_id + 1}"
            dealer_id = dealer_ids[cluster_id] if cluster_id < len(dealer_ids) else dealer_ids[0]
            
            cluster_customers = [customer_ids[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            center_lat, center_lon = self.calculate_territory_center(cluster_customers)
            total_sales_volume = sum(self.customers[cid].sales_volume for cid in cluster_customers)
            total_sales_potential = sum(self.customers[cid].sales_potential for cid in cluster_customers)
            
            dealer = self.dealers[dealer_id]
            workload_score = self.calculate_workload_score(cluster_customers, dealer)
            travel_time = self.estimate_travel_time(dealer, cluster_customers)
            
            territory = Territory(
                id=territory_id,
                name=f"Hierarchical Territory {cluster_id + 1}",
                dealer_id=dealer_id,
                customers=cluster_customers,
                center_latitude=center_lat,
                center_longitude=center_lon,
                total_sales_volume=total_sales_volume,
                total_sales_potential=total_sales_potential,
                estimated_travel_time=travel_time,
                workload_score=workload_score
            )
            
            territories[territory_id] = territory
        
        self.territories = territories
        return territories
    
    def optimize_territories_genetic(self, generations: int = 100, population_size: int = 50) -> Dict[str, Territory]:
        """Optimize territories using genetic algorithm"""
        
        class Individual:
            def __init__(self, assignments: List[int]):
                self.assignments = assignments  # customer_index -> territory_index
                self.fitness = 0.0
        
        def create_individual() -> Individual:
            """Create random territory assignment"""
            assignments = [random.randint(0, len(self.dealers) - 1) for _ in range(len(self.customers))]
            return Individual(assignments)
        
        def calculate_fitness(individual: Individual) -> float:
            """Calculate fitness score for territory assignment"""
            # Group customers by territory
            territory_groups = {}
            customer_ids = list(self.customers.keys())
            
            for i, territory_idx in enumerate(individual.assignments):
                if territory_idx not in territory_groups:
                    territory_groups[territory_idx] = []
                territory_groups[territory_idx].append(customer_ids[i])
            
            # Calculate balance, compactness, and coverage
            temp_territories = []
            dealer_ids = list(self.dealers.keys())
            
            for territory_idx, customer_group in territory_groups.items():
                dealer_id = dealer_ids[territory_idx]
                dealer = self.dealers[dealer_id]
                
                center_lat, center_lon = self.calculate_territory_center(customer_group)
                workload = self.calculate_workload_score(customer_group, dealer)
                
                temp_territory = Territory(
                    id=f"temp_{territory_idx}",
                    name=f"Temp {territory_idx}",
                    dealer_id=dealer_id,
                    customers=customer_group,
                    center_latitude=center_lat,
                    center_longitude=center_lon,
                    workload_score=workload
                )
                temp_territories.append(temp_territory)
            
            # Calculate fitness components
            balance_score = TerritoryMetrics.calculate_balance_score(temp_territories)
            compactness_score = TerritoryMetrics.calculate_compactness_score(temp_territories, self.customers)
            coverage_score = TerritoryMetrics.calculate_coverage_score(temp_territories, self.customers)
            
            # Combined fitness score
            fitness = 0.4 * balance_score + 0.4 * compactness_score + 0.2 * coverage_score
            return fitness
        
        def crossover(parent1: Individual, parent2: Individual) -> Individual:
            """Crossover two parents to create offspring"""
            crossover_point = random.randint(1, len(parent1.assignments) - 1)
            child_assignments = parent1.assignments[:crossover_point] + parent2.assignments[crossover_point:]
            return Individual(child_assignments)
        
        def mutate(individual: Individual, mutation_rate: float = 0.1) -> Individual:
            """Mutate individual by randomly changing some assignments"""
            mutated_assignments = individual.assignments.copy()
            
            for i in range(len(mutated_assignments)):
                if random.random() < mutation_rate:
                    mutated_assignments[i] = random.randint(0, len(self.dealers) - 1)
            
            return Individual(mutated_assignments)
        
        # Initialize population
        population = [create_individual() for _ in range(population_size)]
        
        # Evaluate initial population
        for individual in population:
            individual.fitness = calculate_fitness(individual)
        
        best_individual = max(population, key=lambda x: x.fitness)
        
        # Evolution loop
        for generation in range(generations):
            # Selection (tournament selection)
            new_population = []
            
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 5
                tournament = random.sample(population, tournament_size)
                winner = max(tournament, key=lambda x: x.fitness)
                
                # Crossover
                if random.random() < 0.8:  # 80% crossover rate
                    partner = random.choice(population)
                    child = crossover(winner, partner)
                else:
                    child = Individual(winner.assignments.copy())
                
                # Mutation
                child = mutate(child)
                child.fitness = calculate_fitness(child)
                
                new_population.append(child)
            
            population = new_population
            
            # Track best individual
            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > best_individual.fitness:
                best_individual = current_best
            
            if generation % 20 == 0:
                self.logger.info(f"Generation {generation}: Best fitness = {best_individual.fitness:.4f}")
        
        # Convert best solution to territories
        territories = {}
        customer_ids = list(self.customers.keys())
        dealer_ids = list(self.dealers.keys())
        
        territory_groups = {}
        for i, territory_idx in enumerate(best_individual.assignments):
            if territory_idx not in territory_groups:
                territory_groups[territory_idx] = []
            territory_groups[territory_idx].append(customer_ids[i])
        
        for territory_idx, customer_group in territory_groups.items():
            territory_id = f"territory_ga_{territory_idx + 1}"
            dealer_id = dealer_ids[territory_idx]
            dealer = self.dealers[dealer_id]
            
            center_lat, center_lon = self.calculate_territory_center(customer_group)
            total_sales_volume = sum(self.customers[cid].sales_volume for cid in customer_group)
            total_sales_potential = sum(self.customers[cid].sales_potential for cid in customer_group)
            workload_score = self.calculate_workload_score(customer_group, dealer)
            travel_time = self.estimate_travel_time(dealer, customer_group)
            
            territory = Territory(
                id=territory_id,
                name=f"GA Territory {territory_idx + 1}",
                dealer_id=dealer_id,
                customers=customer_group,
                center_latitude=center_lat,
                center_longitude=center_lon,
                total_sales_volume=total_sales_volume,
                total_sales_potential=total_sales_potential,
                estimated_travel_time=travel_time,
                workload_score=workload_score
            )
            
            territories[territory_id] = territory
        
        self.territories = territories
        return territories
    
    def create_voronoi_territories(self) -> Dict[str, Territory]:
        """Create territories using Voronoi diagrams based on dealer locations"""
        if not self.dealers:
            return {}
        
        # Get dealer locations
        dealer_points = []
        dealer_ids = []
        
        for dealer_id, dealer in self.dealers.items():
            dealer_points.append([dealer.home_longitude, dealer.home_latitude])
            dealer_ids.append(dealer_id)
        
        if len(dealer_points) < 2:
            # If only one dealer, assign all customers to them
            territory_id = "territory_voronoi_1"
            dealer_id = dealer_ids[0]
            all_customers = list(self.customers.keys())
            
            center_lat, center_lon = self.calculate_territory_center(all_customers)
            total_sales_volume = sum(self.customers[cid].sales_volume for cid in all_customers)
            total_sales_potential = sum(self.customers[cid].sales_potential for cid in all_customers)
            
            dealer = self.dealers[dealer_id]
            workload_score = self.calculate_workload_score(all_customers, dealer)
            travel_time = self.estimate_travel_time(dealer, all_customers)
            
            territory = Territory(
                id=territory_id,
                name="Voronoi Territory 1",
                dealer_id=dealer_id,
                customers=all_customers,
                center_latitude=center_lat,
                center_longitude=center_lon,
                total_sales_volume=total_sales_volume,
                total_sales_potential=total_sales_potential,
                estimated_travel_time=travel_time,
                workload_score=workload_score
            )
            
            return {territory_id: territory}
        
        # Create Voronoi diagram
        vor = Voronoi(dealer_points)
        
        # Assign customers to nearest dealer
        territories = {}
        for i, dealer_id in enumerate(dealer_ids):
            territory_id = f"territory_voronoi_{i + 1}"
            territories[territory_id] = Territory(
                id=territory_id,
                name=f"Voronoi Territory {i + 1}",
                dealer_id=dealer_id,
                customers=[],
                center_latitude=0.0,
                center_longitude=0.0,
                total_sales_volume=0.0,
                total_sales_potential=0.0,
                estimated_travel_time=0,
                workload_score=0.0
            )
        
        # Assign customers to nearest dealer
        for customer_id, customer in self.customers.items():
            min_distance = float('inf')
            nearest_dealer_idx = 0
            
            for i, dealer_id in enumerate(dealer_ids):
                dealer = self.dealers[dealer_id]
                distance = self._haversine_distance(
                    customer.latitude, customer.longitude,
                    dealer.home_latitude, dealer.home_longitude
                )
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_dealer_idx = i
            
            territory_id = f"territory_voronoi_{nearest_dealer_idx + 1}"
            territories[territory_id].customers.append(customer_id)
        
        # Calculate territory metrics
        for territory in territories.values():
            if territory.customers:
                territory.center_latitude, territory.center_longitude = self.calculate_territory_center(territory.customers)
                territory.total_sales_volume = sum(self.customers[cid].sales_volume for cid in territory.customers)
                territory.total_sales_potential = sum(self.customers[cid].sales_potential for cid in territory.customers)
                
                dealer = self.dealers[territory.dealer_id]
                territory.workload_score = self.calculate_workload_score(territory.customers, dealer)
                territory.estimated_travel_time = self.estimate_travel_time(dealer, territory.customers)
        
        self.territories = territories
        return territories
    
    def optimize_boundaries(self, max_iterations: int = 100) -> Dict[str, Territory]:
        """Optimize territory boundaries using iterative improvement"""
        if not self.territories:
            return {}
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try to improve each territory
            for territory_id, territory in self.territories.items():
                # Check if any customer should be moved to a different territory
                customers_to_move = []
                
                for customer_id in territory.customers:
                    customer = self.customers[customer_id]
                    current_dealer = self.dealers[territory.dealer_id]
                    
                    # Calculate current cost
                    current_distance = self._haversine_distance(
                        customer.latitude, customer.longitude,
                        current_dealer.home_latitude, current_dealer.home_longitude
                    )
                    
                    # Check if customer is closer to another dealer
                    best_territory = territory_id
                    best_distance = current_distance
                    
                    for other_territory_id, other_territory in self.territories.items():
                        if other_territory_id == territory_id:
                            continue
                        
                        other_dealer = self.dealers[other_territory.dealer_id]
                        other_distance = self._haversine_distance(
                            customer.latitude, customer.longitude,
                            other_dealer.home_latitude, other_dealer.home_longitude
                        )
                        
                        # Check if move would be beneficial
                        if (other_distance < best_distance and 
                            len(other_territory.customers) < other_dealer.capacity):
                            best_territory = other_territory_id
                            best_distance = other_distance
                    
                    if best_territory != territory_id:
                        customers_to_move.append((customer_id, best_territory))
                
                # Move customers
                for customer_id, target_territory_id in customers_to_move:
                    territory.customers.remove(customer_id)
                    self.territories[target_territory_id].customers.append(customer_id)
                    self.customers[customer_id].territory_id = target_territory_id
                    improved = True
            
            # Recalculate territory metrics after moves
            for territory in self.territories.values():
                if territory.customers:
                    territory.center_latitude, territory.center_longitude = self.calculate_territory_center(territory.customers)
                    territory.total_sales_volume = sum(self.customers[cid].sales_volume for cid in territory.customers)
                    territory.total_sales_potential = sum(self.customers[cid].sales_potential for cid in territory.customers)
                    
                    dealer = self.dealers[territory.dealer_id]
                    territory.workload_score = self.calculate_workload_score(territory.customers, dealer)
                    territory.estimated_travel_time = self.estimate_travel_time(dealer, territory.customers)
            
            if iteration % 10 == 0:
                self.logger.info(f"Boundary optimization iteration {iteration}")
        
        self.logger.info(f"Boundary optimization completed after {iteration} iterations")
        return self.territories
    
    def evaluate_territories(self) -> Dict[str, float]:
        """Evaluate current territory configuration"""
        if not self.territories:
            return {}
        
        territories_list = list(self.territories.values())
        
        balance_score = TerritoryMetrics.calculate_balance_score(territories_list)
        compactness_score = TerritoryMetrics.calculate_compactness_score(territories_list, self.customers)
        coverage_score = TerritoryMetrics.calculate_coverage_score(territories_list, self.customers)
        
        # Calculate additional metrics
        total_travel_time = sum(t.estimated_travel_time for t in territories_list)
        avg_travel_time = total_travel_time / len(territories_list) if territories_list else 0
        
        total_workload = sum(t.workload_score for t in territories_list)
        workload_variance = np.var([t.workload_score for t in territories_list])
        
        return {
            'balance_score': balance_score,
            'compactness_score': compactness_score,
            'coverage_score': coverage_score,
            'overall_score': 0.4 * balance_score + 0.4 * compactness_score + 0.2 * coverage_score,
            'avg_travel_time': avg_travel_time,
            'total_workload': total_workload,
            'workload_variance': workload_variance,
            'num_territories': len(territories_list)
        }
    
    def save_territories(self, filepath: str):
        """Save territories to file"""
        territories_data = {}
        
        for territory_id, territory in self.territories.items():
            territories_data[territory_id] = {
                'id': territory.id,
                'name': territory.name,
                'dealer_id': territory.dealer_id,
                'customers': territory.customers,
                'boundary_points': territory.boundary_points,
                'center_latitude': territory.center_latitude,
                'center_longitude': territory.center_longitude,
                'total_sales_volume': territory.total_sales_volume,
                'total_sales_potential': territory.total_sales_potential,
                'estimated_travel_time': territory.estimated_travel_time,
                'workload_score': territory.workload_score
            }
        
        with open(filepath, 'w') as f:
            json.dump(territories_data, f, indent=2)
    
    def load_territories(self, filepath: str):
        """Load territories from file"""
        with open(filepath, 'r') as f:
            territories_data = json.load(f)
        
        self.territories = {}
        
        for territory_id, data in territories_data.items():
            territory = Territory(
                id=data['id'],
                name=data['name'],
                dealer_id=data['dealer_id'],
                customers=data['customers'],
                boundary_points=data['boundary_points'],
                center_latitude=data['center_latitude'],
                center_longitude=data['center_longitude'],
                total_sales_volume=data['total_sales_volume'],
                total_sales_potential=data['total_sales_potential'],
                estimated_travel_time=data['estimated_travel_time'],
                workload_score=data['workload_score']
            )
            
            self.territories[territory_id] = territory
    
    def visualize_territories(self, save_path: str = None):
        """Visualize territories on a map"""
        if not self.territories:
            self.logger.warning("No territories to visualize")
            return
        
        plt.figure(figsize=(12, 10))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.territories)))
        
        for i, (territory_id, territory) in enumerate(self.territories.items()):
            # Plot customers
            if territory.customers:
                customer_lats = [self.customers[cid].latitude for cid in territory.customers]
                customer_lons = [self.customers[cid].longitude for cid in territory.customers]
                
                plt.scatter(customer_lons, customer_lats, c=[colors[i]], 
                           label=f'{territory.name} (Customers)', alpha=0.6, s=30)
            
            # Plot dealer
            dealer = self.dealers[territory.dealer_id]
            plt.scatter(dealer.home_longitude, dealer.home_latitude, 
                       c=[colors[i]], marker='s', s=200, edgecolors='black', linewidth=2,
                       label=f'{territory.name} (Dealer)')
            
            # Plot territory center
            plt.scatter(territory.center_longitude, territory.center_latitude, 
                       c=[colors[i]], marker='*', s=150, edgecolors='black',
                       label=f'{territory.name} (Center)')
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Territory Visualization')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_territory_report(self) -> str:
        """Generate detailed territory report"""
        if not self.territories:
            return "No territories configured."
        
        report = "TERRITORY OPTIMIZATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Overall metrics
        metrics = self.evaluate_territories()
        report += "OVERALL METRICS:\n"
        report += f"  Balance Score: {metrics['balance_score']:.4f}\n"
        report += f"  Compactness Score: {metrics['compactness_score']:.4f}\n"
        report += f"  Coverage Score: {metrics['coverage_score']:.4f}\n"
        report += f"  Overall Score: {metrics['overall_score']:.4f}\n"
        report += f"  Average Travel Time: {metrics['avg_travel_time']:.1f} minutes\n"
        report += f"  Total Workload: {metrics['total_workload']:.2f}\n"
        report += f"  Workload Variance: {metrics['workload_variance']:.2f}\n\n"
        
        # Territory details
        report += "TERRITORY DETAILS:\n"
        report += "-" * 30 + "\n"
        
        for territory_id, territory in self.territories.items():
            dealer = self.dealers[territory.dealer_id]
            
            report += f"\nTerritory: {territory.name}\n"
            report += f"  Dealer: {dealer.name} ({dealer.id})\n"
            report += f"  Customers: {len(territory.customers)}\n"
            report += f"  Sales Volume: ${territory.total_sales_volume:,.2f}\n"
            report += f"  Sales Potential: ${territory.total_sales_potential:,.2f}\n"
            report += f"  Workload Score: {territory.workload_score:.2f}\n"
            report += f"  Travel Time: {territory.estimated_travel_time} minutes\n"
            report += f"  Center: ({territory.center_latitude:.4f}, {territory.center_longitude:.4f})\n"
            
            # Top customers by sales volume
            if territory.customers:
                customer_volumes = [(cid, self.customers[cid].sales_volume) for cid in territory.customers]
                customer_volumes.sort(key=lambda x: x[1], reverse=True)
                
                report += f"  Top Customers:\n"
                for cid, volume in customer_volumes[:3]:
                    customer = self.customers[cid]
                    report += f"    - {customer.name} ({cid}): ${volume:,.2f}\n"
        
        return report


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    customers = [
        Customer(id="C001", name="Customer 1", latitude=6.9271, longitude=79.8612, 
                sales_volume=50000, sales_potential=60000, priority=2),
        Customer(id="C002", name="Customer 2", latitude=6.9319, longitude=79.8478, 
                sales_volume=75000, sales_potential=85000, priority=3),
        Customer(id="C003", name="Customer 3", latitude=6.9147, longitude=79.8727, 
                sales_volume=30000, sales_potential=40000, priority=1),
        Customer(id="C004", name="Customer 4", latitude=6.9023, longitude=79.8607, 
                sales_volume=90000, sales_potential=100000, priority=3),
        Customer(id="C005", name="Customer 5", latitude=6.9511, longitude=79.8391, 
                sales_volume=45000, sales_potential=55000, priority=2),
    ]
    
    dealers = [
        Dealer(id="D001", name="Dealer 1", home_latitude=6.9271, home_longitude=79.8612, 
               capacity=3, experience_level=2),
        Dealer(id="D002", name="Dealer 2", home_latitude=6.9147, home_longitude=79.8727, 
               capacity=4, experience_level=3),
    ]
    
    # Initialize optimizer
    optimizer = TerritoryOptimizer(customers, dealers)
    
    # Test different territory creation methods
    print("Testing K-means clustering...")
    territories_kmeans = optimizer.create_territories_kmeans()
    print(f"Created {len(territories_kmeans)} territories using K-means")
    
    print("\nTesting Hierarchical clustering...")
    territories_hierarchical = optimizer.create_territories_hierarchical()
    print(f"Created {len(territories_hierarchical)} territories using Hierarchical clustering")
    
    print("\nTesting Voronoi territories...")
    territories_voronoi = optimizer.create_voronoi_territories()
    print(f"Created {len(territories_voronoi)} territories using Voronoi")
    
    print("\nTesting Genetic Algorithm optimization...")
    territories_ga = optimizer.optimize_territories_genetic(generations=20, population_size=20)
    print(f"Created {len(territories_ga)} territories using Genetic Algorithm")
    
    # Evaluate each method
    methods = [
        ("K-means", territories_kmeans),
        ("Hierarchical", territories_hierarchical),
        ("Voronoi", territories_voronoi),
        ("Genetic Algorithm", territories_ga)
    ]
    
    print("\nEVALUATION RESULTS:")
    print("=" * 60)
    
    for method_name, territories in methods:
        optimizer.territories = territories
        metrics = optimizer.evaluate_territories()
        
        print(f"\n{method_name}:")
        print(f"  Overall Score: {metrics.get('overall_score', 0):.4f}")
        print(f"  Balance Score: {metrics.get('balance_score', 0):.4f}")
        print(f"  Compactness Score: {metrics.get('compactness_score', 0):.4f}")
        print(f"  Coverage Score: {metrics.get('coverage_score', 0):.4f}")
    
    # Use best method and optimize boundaries
    optimizer.territories = territories_ga  # Assuming GA performed best
    print("\nOptimizing boundaries...")
    optimized_territories = optimizer.optimize_boundaries(max_iterations=50)
    
    # Generate final report
    print("\nFINAL TERRITORY REPORT:")
    print("=" * 60)
    report = optimizer.generate_territory_report()
    print(report)