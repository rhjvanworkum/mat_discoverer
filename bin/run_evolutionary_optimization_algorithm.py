"""
Example: python bin/evolutionary_algorithm.py run_evolutionary_optimization_algorithm --species "['Zn', 'Cu']" --num_species "[1, 1]"
"""
import fire
from tqdm import tqdm
from typing import List
from pymatgen.core import Composition
import numpy as np
import random

from src.generate_random_crystals import generate_random_crystal_structure_from_composition
from src.ml_structure_optimizer import MLStructureOptimizer
from src.convex_hull_calculator import ConvexHullCalculator
from evolutionary_algorithm.fitness_function import fitness_function
from evolutionary_algorithm.selection import selection_fn
from evolutionary_algorithm.crossover_operator import crossover_operator
from evolutionary_algorithm.mutate_operator import mutate_operator


def run_evolutionary_optimization_algorithm(
    species: List[str],
    num_species: List[int],
    population_size: int = 20,
    generations: int = 5,
    mutation_rate: float = 0.1,
    selection_pressure: float = 2.0,
):
    """
    :param population_size: Number of structures in the population
    :param generations: Number of generations to run the evolutionary algorithm
    :param mutation_rate: Probability of mutation for each offspring
    :param selection_pressure: Selection pressure for fitness-based selection
    """
    initial_population = [
        generate_random_crystal_structure_from_composition(
            species=species, 
            num_species=num_species
        ) for _ in range(population_size)
    ]
    # NOTE: because the num_species can be floats like 0.5 or 0.333, we multiply here by the smallest mulitplyier to 1
    convex_hull_calculator = ConvexHullCalculator(
        composition=Composition({elem: num * ( 1 / min(num_species)) for elem, num in zip(species, num_species)})
    )
    optimizer = MLStructureOptimizer()

    population, fitnesses = map(list, zip(*[fitness_function(structure, optimizer, convex_hull_calculator) 
                                            for structure in tqdm(initial_population, 
                                                                  desc="Relaxation of intial population", 
                                                                  total=len(initial_population))]))

    for generation in range(generations):
        print(f'Working on generation {generation + 1}...')
        # 1. Selection: Select parents based on fitness
        selected_population, selected_fitnesses = selection_fn(population, fitnesses, selection_pressure)

        # Create new population via crossover and mutation
        new_population = []
        for _ in range(population_size):
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            offspring = crossover_operator(parent1, parent2)
            if np.random.rand() < mutation_rate:
                offspring = mutate_operator(offspring)
            new_population.append(offspring)

        new_population, new_fitnesses = map(list, zip(*[fitness_function(structure, optimizer, convex_hull_calculator) 
                                            for structure in tqdm(new_population, 
                                                                  desc="Relaxation of new population", 
                                                                  total=len(new_population))]))

        population = selected_population + new_population
        fitnesses = selected_fitnesses + new_fitnesses

        # Track the best structure
        best_structure = population[np.argmin(fitnesses)]
        best_fitness = np.min(fitnesses)
        print(f"Generation {generation + 1}: Best Fitness / e_above_hull = {best_fitness}")

    print(best_structure, best_fitness)
    best_structure.to(filename="best_structure.cif")


if __name__ == "__main__":
    fire.Fire(run_evolutionary_optimization_algorithm)