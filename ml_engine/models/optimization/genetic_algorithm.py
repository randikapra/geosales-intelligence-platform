"""
Genetic Algorithm Implementation for Various Optimization Problems
Supports multiple optimization scenarios including territory optimization,
route optimization, and resource allocation.
"""

import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class GeneticAlgorithmConfig:
    """Configuration parameters for genetic algorithm"""
    population_size: int = 100
    generations: int = 500
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 5
    elite_size: int = 5
    max_stagnation: int = 50
    convergence_threshold: float = 1e-6
    parallel_execution: bool = True
    max_workers: int = 4

class Individual(ABC):
    """Abstract base class for genetic algorithm individuals"""
    
    def __init__(self, chromosome: List[Any]):
        self.chromosome = chromosome
        self.fitness: float = 0.0
        self.normalized_fitness: float = 0.0
        self.rank: int = 0
    
    @abstractmethod
    def calculate_fitness(self) -> float:
        """Calculate fitness score for the individual"""
        pass
    
    @abstractmethod
    def crossover(self, other: 'Individual') -> 'Individual':
        """Perform crossover with another individual"""
        pass
    
    @abstractmethod
    def mutate(self, mutation_rate: float) -> 'Individual':
        """Perform mutation on the individual"""
        pass
    
    @abstractmethod
    def clone(self) -> 'Individual':
        """Create a copy of the individual"""
        pass
    
    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __eq__(self, other):
        return self.fitness == other.fitness

class TerritoryIndividual(Individual):
    """Individual for territory optimization problems"""
    
    def __init__(self, chromosome: List[int], customers: Dict, dealers: Dict):
        super().__init__(chromosome)
        self.customers = customers
        self.dealers = dealers
        self.territories: Dict[int, List[str]] = {}
        self._group_customers()
    
    def _group_customers(self):
        """Group customers by territory assignment"""
        self.territories = {}
        customer_ids = list(self.customers.keys())
        
        for i, territory_id in enumerate(self.chromosome):
            if territory_id not in self.territories:
                self.territories[territory_id] = []
            self.territories[territory_id].append(customer_ids[i])
    
    def calculate_fitness(self) -> float:
        """Calculate fitness based on territory quality metrics"""
        if not self.territories:
            return 0.0
        
        # Calculate balance score
        workloads = []
        for territory_id, customer_ids in self.territories.items():
            if territory_id < len(self.dealers):
                dealer_id = list(self.dealers.keys())[territory_id]
                dealer = self.dealers[dealer_id]
                workload = self._calculate_workload(customer_ids, dealer)
                workloads.append(workload)
        
        if not workloads:
            return 0.0
        
        # Balance score (lower variance is better)
        mean_workload = np.mean(workloads)
        if mean_workload > 0:
            balance_score = 1.0 / (1.0 + np.var(workloads) / (mean_workload ** 2))
        else:
            balance_score = 0.0
        
        # Compactness score (lower average distance is better)
        compactness_score = self._calculate_compactness()
        
        # Capacity constraint penalty
        capacity_penalty = self._calculate_capacity_penalty()
        
        # Combined fitness
        fitness = 0.5 * balance_score + 0.3 * compactness_score - 0.2 * capacity_penalty
        self.fitness = max(0.0, fitness)
        return self.fitness
    
    def _calculate_workload(self, customer_ids: List[str], dealer) -> float:
        """Calculate workload for a dealer based on assigned customers"""
        if not customer_ids:
            return 0.0
        
        total_workload = 0.0
        for customer_id in customer_ids:
            customer = self.customers[customer_id]
            workload = (customer.sales_volume * 0.3 + 
                       customer.sales_potential * 0.3 + 
                       customer.priority * 0.2 + 
                       customer.visit_frequency * 0.2)
            total_workload += workload
        
        return total_workload
    
    def _calculate_compactness(self) -> float:
        """Calculate territory compactness score"""
        if not self.territories:
            return 0.0
        
        total_compactness = 0.0
        dealer_ids = list(self.dealers.keys())
        
        for territory_id, customer_ids in self.territories.items():
            if territory_id >= len(dealer_ids) or not customer_ids:
                continue
            
            dealer = self.dealers[dealer_ids[territory_id]]
            
            # Calculate average distance from dealer to customers
            distances = []
            for customer_id in customer_ids:
                customer = self.customers[customer_id]
                distance = self._haversine_distance(
                    dealer.home_latitude, dealer.home_longitude,
                    customer.latitude, customer.longitude
                )
                distances.append(distance)
            
            if distances:
                avg_distance = np.mean(distances)
                compactness = 1.0 / (1.0 + avg_distance)
                total_compactness += compactness
        
        return total_compactness / len(self.territories) if self.territories else 0.0
    
    def _calculate_capacity_penalty(self) -> float:
        """Calculate penalty for exceeding dealer capacity"""
        penalty = 0.0
        dealer_ids = list(self.dealers.keys())
        
        for territory_id, customer_ids in self.territories.items():
            if territory_id >= len(dealer_ids):
                continue
            
            dealer = self.dealers[dealer_ids[territory_id]]
            if len(customer_ids) > dealer.capacity:
                penalty += (len(customer_ids) - dealer.capacity) / dealer.capacity
        
        return penalty
    
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
    
    def crossover(self, other: 'TerritoryIndividual') -> 'TerritoryIndividual':
        """Perform crossover with another territory individual"""
        # Single-point crossover
        crossover_point = random.randint(1, len(self.chromosome) - 1)
        child_chromosome = (self.chromosome[:crossover_point] + 
                          other.chromosome[crossover_point:])
        
        return TerritoryIndividual(child_chromosome, self.customers, self.dealers)
    
    def mutate(self, mutation_rate: float) -> 'TerritoryIndividual':
        """Perform mutation on the territory individual"""
        mutated_chromosome = self.chromosome.copy()
        num_dealers = len(self.dealers)
        
        for i in range(len(mutated_chromosome)):
            if random.random() < mutation_rate:
                mutated_chromosome[i] = random.randint(0, num_dealers - 1)
        
        return TerritoryIndividual(mutated_chromosome, self.customers, self.dealers)
    
    def clone(self) -> 'TerritoryIndividual':
        """Create a copy of the territory individual"""
        return TerritoryIndividual(self.chromosome.copy(), self.customers, self.dealers)

class RouteIndividual(Individual):
    """Individual for route optimization problems (TSP-like)"""
    
    def __init__(self, chromosome: List[int], distance_matrix: np.ndarray):
        super().__init__(chromosome)
        self.distance_matrix = distance_matrix
    
    def calculate_fitness(self) -> float:
        """Calculate fitness based on total route distance"""
        if len(self.chromosome) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(self.chromosome)):
            from_node = self.chromosome[i]
            to_node = self.chromosome[(i + 1) % len(self.chromosome)]
            total_distance += self.distance_matrix[from_node][to_node]
        
        # Fitness is inverse of distance (shorter routes are better)
        self.fitness = 1.0 / (1.0 + total_distance)
        return self.fitness
    
    def crossover(self, other: 'RouteIndividual') -> 'RouteIndividual':
        """Perform order crossover (OX) for route optimization"""
        size = len(self.chromosome)
        
        # Select two random crossover points
        start = random.randint(0, size - 1)
        end = random.randint(start + 1, size)
        
        # Copy segment from parent 1
        child_chromosome = [-1] * size
        child_chromosome[start:end] = self.chromosome[start:end]
        
        # Fill remaining positions with cities from parent 2
        remaining_cities = [city for city in other.chromosome if city not in child_chromosome]
        
        j = 0
        for i in range(size):
            if child_chromosome[i] == -1:
                child_chromosome[i] = remaining_cities[j]
                j += 1
        
        return RouteIndividual(child_chromosome, self.distance_matrix)
    
    def mutate(self, mutation_rate: float) -> 'RouteIndividual':
        """Perform swap mutation on the route individual"""
        mutated_chromosome = self.chromosome.copy()
        
        if random.random() < mutation_rate:
            # Swap two random cities
            i, j = random.sample(range(len(mutated_chromosome)), 2)
            mutated_chromosome[i], mutated_chromosome[j] = mutated_chromosome[j], mutated_chromosome[i]
        
        return RouteIndividual(mutated_chromosome, self.distance_matrix)
    
    def clone(self) -> 'RouteIndividual':
        """Create a copy of the route individual"""
        return RouteIndividual(self.chromosome.copy(), self.distance_matrix)

class ResourceAllocationIndividual(Individual):
    """Individual for resource allocation optimization"""
    
    def __init__(self, chromosome: List[float], resources: Dict, demands: Dict):
        super().__init__(chromosome)
        self.resources = resources
        self.demands = demands
    
    def calculate_fitness(self) -> float:
        """Calculate fitness based on resource allocation efficiency"""
        if not self.chromosome:
            return 0.0
        
        # Calculate allocation efficiency
        total_satisfaction = 0.0
        total_waste = 0.0
        
        resource_ids = list(self.resources.keys())
        demand_ids = list(self.demands.keys())
        
        # Each gene represents allocation percentage
        for i, allocation in enumerate(self.chromosome):
            if i < len(resource_ids) and i < len(demand_ids):
                resource_capacity = self.resources[resource_ids[i]]['capacity']
                demand_requirement = self.demands[demand_ids[i]]['requirement']
                
                allocated_amount = allocation * resource_capacity
                
                # Satisfaction (how well demand is met)
                satisfaction = min(allocated_amount, demand_requirement) / demand_requirement
                total_satisfaction += satisfaction
                
                # Waste (over-allocation penalty)
                if allocated_amount > demand_requirement:
                    waste = (allocated_amount - demand_requirement) / resource_capacity
                    total_waste += waste
        
        # Fitness combines satisfaction and minimizes waste
        avg_satisfaction = total_satisfaction / len(demand_ids) if demand_ids else 0.0
        avg_waste = total_waste / len(resource_ids) if resource_ids else 0.0
        
        self.fitness = avg_satisfaction - 0.5 * avg_waste
        return max(0.0, self.fitness)
    
    def crossover(self, other: 'ResourceAllocationIndividual') -> 'ResourceAllocationIndividual':
        """Perform uniform crossover for resource allocation"""
        child_chromosome = []
        
        for i in range(len(self.chromosome)):
            if random.random() < 0.5:
                child_chromosome.append(self.chromosome[i])
            else:
                child_chromosome.append(other.chromosome[i])
        
        return ResourceAllocationIndividual(child_chromosome, self.resources, self.demands)
    
    def mutate(self, mutation_rate: float) -> 'ResourceAllocationIndividual':
        """Perform gaussian mutation on resource allocation"""
        mutated_chromosome = self.chromosome.copy()
        
        for i in range(len(mutated_chromosome)):
            if random.random() < mutation_rate:
                # Add gaussian noise and clamp to [0, 1]
                noise = random.gauss(0, 0.1)
                mutated_chromosome[i] = max(0.0, min(1.0, mutated_chromosome[i] + noise))
        
        return ResourceAllocationIndividual(mutated_chromosome, self.resources, self.demands)
    
    def clone(self) -> 'ResourceAllocationIndividual':
        """Create a copy of the resource allocation individual"""
        return ResourceAllocationIndividual(self.chromosome.copy(), self.resources, self.demands)

class GeneticAlgorithm:
    """Main genetic algorithm engine"""
    
    def __init__(self, config: GeneticAlgorithmConfig):
        self.config = config
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[float] = []
        self.generation: int = 0
        self.stagnation_count: int = 0
        self.logger = logging.getLogger(__name__)
    
    def initialize_population(self, individual_factory: Callable[[], Individual]):
        """Initialize the population with random individuals"""
        self.population = []
        
        if self.config.parallel_execution:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [executor.submit(individual_factory) for _ in range(self.config.population_size)]
                self.population = [future.result() for future in futures]
        else:
            for _ in range(self.config.population_size):
                self.population.append(individual_factory())
        
        # Evaluate initial population
        self._evaluate_population()
        self._update_best_individual()
    
    def _evaluate_population(self):
        """Evaluate fitness for all individuals in population"""
        if self.config.parallel_execution:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [executor.submit(individual.calculate_fitness) for individual in self.population]
                for future in futures:
                    future.result()
        else:
            for individual in self.population:
                individual.calculate_fitness()
        
        # Sort population by fitness
        self.population.sort(reverse=True)
        
        # Calculate normalized fitness and ranks
        self._calculate_normalized_fitness()
    
    def _calculate_normalized_fitness(self):
        """Calculate normalized fitness values for selection"""
        if not self.population:
            return
        
        fitness_values = [ind.fitness for ind in self.population]
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)
        
        if max_fitness == min_fitness:
            for individual in self.population:
                individual.normalized_fitness = 1.0
        else:
            for individual in self.population:
                individual.normalized_fitness = (individual.fitness - min_fitness) / (max_fitness - min_fitness)
        
        # Assign ranks
        for i, individual in enumerate(self.population):
            individual.rank = i + 1
    
    def _update_best_individual(self):
        """Update the best individual found so far"""
        if self.population:
            current_best = self.population[0]
            
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best.clone()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
            
            self.fitness_history.append(current_best.fitness)
    
    def tournament_selection(self) -> Individual:
        """Select an individual using tournament selection"""
        tournament = random.sample(self.population, min(self.config.tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def roulette_wheel_selection(self) -> Individual:
        """Select an individual using roulette wheel selection"""
        total_fitness = sum(ind.normalized_fitness for ind in self.population)
        
        if total_fitness == 0:
            return random.choice(self.population)
        
        r = random.uniform(0, total_fitness)
        current_sum = 0
        
        for individual in self.population:
            current_sum += individual.normalized_fitness
            if current_sum >= r:
                return individual
        
        return self.population[-1]
    
    def evolve_generation(self):
        """Evolve one generation"""
        new_population = []
        
        # Elitism: keep best individuals
        elite_size = min(self.config.elite_size, len(self.population))
        new_population.extend([ind.clone() for ind in self.population[:elite_size]])
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = parent1.clone()
            
            # Mutation
            child = child.mutate(self.config.mutation_rate)
            
            new_population.append(child)
        
        # Replace population
        self.population = new_population[:self.config.population_size]
        
        # Evaluate new population
        self._evaluate_population()
        self._update_best_individual()
        
        self.generation += 1
    
    def run(self, individual_factory: Callable[[], Individual]) -> Individual:
        """Run the genetic algorithm"""
        self.logger.info(f"Starting genetic algorithm with {self.config.population_size} individuals for {self.config.generations} generations")
        
        # Initialize population
        self.initialize_population(individual_factory)
        
        # Evolution loop
        for generation in range(self.config.generations):
            self.evolve_generation()
            
            # Check for convergence
            if self._check_convergence():
                self.logger.info(f"Converged at generation {generation}")
                break
            
            # Check for stagnation
            if self.stagnation_count >= self.config.max_stagnation:
                self.logger.info(f"Stagnation detected at generation {generation}")
                break
            
            # Log progress
            if generation % 50 == 0:
                best_fitness = self.best_individual.fitness if self.best_individual else 0.0
                self.logger.info(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
        
        self.logger.info(f"Genetic algorithm completed after {self.generation} generations")
        return self.best_individual
    
    def _check_convergence(self) -> bool:
        """Check if the algorithm has converged"""
        if len(self.fitness_history) < 10:
            return False
        
        recent_fitness = self.fitness_history[-10:]
        fitness_std = np.std(recent_fitness)
        
        return fitness_std < self.config.convergence_threshold
    
    def plot_fitness_history(self, save_path: str = None):
        """Plot the fitness evolution over generations"""
        if not self.fitness_history:
            self.logger.warning("No fitness history to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.title('Fitness Evolution Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get algorithm statistics"""
        if not self.population:
            return {}
        
        fitness_values = [ind.fitness for ind in self.population]
        
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': self.best_individual.fitness if self.best_individual else 0.0,
            'worst_fitness': min(fitness_values),
            'average_fitness': np.mean(fitness_values),
            'fitness_std': np.std(fitness_values),
            'stagnation_count': self.stagnation_count,
            'convergence_achieved': self._check_convergence()
        }

# Utility functions for specific optimization problems

def create_territory_individual_factory(customers: Dict, dealers: Dict) -> Callable[[], TerritoryIndividual]:
    """Factory function for creating territory individuals"""
    def factory():
        num_customers = len(customers)
        num_dealers = len(dealers)
        chromosome = [random.randint(0, num_dealers - 1) for _ in range(num_customers)]
        return TerritoryIndividual(chromosome, customers, dealers)
    
    return factory

def create_route_individual_factory(distance_matrix: np.ndarray) -> Callable[[], RouteIndividual]:
    """Factory function for creating route individuals"""
    def factory():
        num_cities = len(distance_matrix)
        chromosome = list(range(num_cities))
        random.shuffle(chromosome)
        return RouteIndividual(chromosome, distance_matrix)
    
    return factory

def create_resource_allocation_individual_factory(resources: Dict, demands: Dict) -> Callable[[], ResourceAllocationIndividual]:
    """Factory function for creating resource allocation individuals"""
    def factory():
        num_resources = len(resources)
        chromosome = [random.random() for _ in range(num_resources)]
        return ResourceAllocationIndividual(chromosome, resources, demands)
    
    return factory

# Example usage
if __name__ == "__main__":
    # Example: Territory optimization
    customers = {
        'C1': type('Customer', (), {'latitude': 6.9271, 'longitude': 79.8612, 'sales_volume': 50000, 'sales_potential': 60000, 'priority': 2, 'visit_frequency': 4})(),
        'C2': type('Customer', (), {'latitude': 6.9319, 'longitude': 79.8478, 'sales_volume': 75000, 'sales_potential': 85000, 'priority': 3, 'visit_frequency': 6})(),
        'C3': type('Customer', (), {'latitude': 6.9147, 'longitude': 79.8727, 'sales_volume': 30000, 'sales_potential': 40000, 'priority': 1, 'visit_frequency': 2})(),
    }
    
    dealers = {
        'D1': type('Dealer', (), {'home_latitude': 6.9271, 'home_longitude': 79.8612, 'capacity': 5, 'experience_level': 2})(),
        'D2': type('Dealer', (), {'home_latitude': 6.9147, 'home_longitude': 79.8727, 'capacity': 4, 'experience_level': 3})(),
    }
    
    # Configure and run genetic algorithm
    config = GeneticAlgorithmConfig(
        population_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        tournament_size=5,
        elite_size=3
    )
    
    ga = GeneticAlgorithm(config)
    individual_factory = create_territory_individual_factory(customers, dealers)
    
    best_solution = ga.run(individual_factory)
    
    print(f"Best solution fitness: {best_solution.fitness:.6f}")
    print(f"Territory assignments: {best_solution.chromosome}")
    
    # Plot fitness evolution
    ga.plot_fitness_history()
    
    # Print statistics
    stats = ga.get_statistics()
    print(f"Algorithm statistics: {stats}")