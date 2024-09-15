"""
This script finds the global minimum of a crystal structure using an evolutionary optimization algorithm.

Example:
    python bin/run_evolutionary_optimization_algorithm.py --species "['Zn', 'Cu']" --num_species "[1, 1]"
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
from src.evolutionary_algorithm.fitness_function import fitness_function
from src.evolutionary_algorithm.selection import selection_fn
from src.evolutionary_algorithm.crossover_operator import crossover_operator
from src.evolutionary_algorithm.mutate_operator import mutate_operator


def run_evolutionary_optimization_algorithm(
    species: List[str],
    num_species: List[int],
    initial_population_size: int = 100,
    population_size: int = 20,
    generations: int = 20,
    mutation_rate: float = 0.1,
    selection_pressure: float = 2.0,
):
    """
    Finds the global minimum of a crystal structure on an ML-FF potential using an evolutionary optimization algorithm.

    :param species: List of atoms types in the composition
    :param num_species: List of number of atoms for each species in the composition
    :param initial_population_size: Number of structures in the initial population
    :param population_size: Number of structures in the population
    :param generations: Number of generations to run the evolutionary algorithm
    :param mutation_rate: Probability of mutation for each offspring
    :param selection_pressure: Selection pressure for fitness-based selection
    """
    
    # Step 1: Generate the initial population
    initial_population = [
        generate_random_crystal_structure_from_composition(species, num_species)
        for _ in range(initial_population_size)
    ]
    
    # Step 2: Setup convex hull calculator
    multiplier = 1 / min(num_species)
    composition = Composition({elem: num * multiplier for elem, num in zip(species, num_species)})
    convex_hull_calculator = ConvexHullCalculator(composition)

    optimizer = MLStructureOptimizer()
    best_volume = None

    # Step 3: Relax and evaluate the initial population
    population, e_above_hulls, energies = zip(*[
        fitness_function(structure, optimizer, convex_hull_calculator)
        for structure in tqdm(initial_population, desc="Relaxing initial population", total=len(initial_population))
    ])
    
    # Track the best structure from the initial population
    best_idx = np.argmin(energies)
    best_structure = population[best_idx]
    best_e_above_hull = e_above_hulls[best_idx]
    best_fitness = energies[best_idx]
    is_in_mp = convex_hull_calculator.structure_is_in_mp(best_structure)
    best_structure.to(filename="outputs/best_structure_initial.cif")

    print(f"Initial population: Best Fitness = {best_fitness}, e_above_hull = {best_e_above_hull}, is_in_mp = {is_in_mp}")

    # Step 4: Evolutionary process
    for generation in range(generations):
        print(f"Processing generation {generation + 1}...")

        # Select parents based on fitness
        selected_indices = selection_fn(e_above_hulls, population_size, selection_pressure)
        selected_population = [population[i] for i in selected_indices]
        selected_e_above_hulls = [e_above_hulls[i] for i in selected_indices]
        selected_energies = [energies[i] for i in selected_indices]

        # Create new population via crossover and mutation
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.choices(selected_population, k=2)
            offspring = crossover_operator(parent1, parent2)
            if np.random.rand() < mutation_rate:
                offspring = mutate_operator(offspring, rescaling_volume=parent1.volume if best_volume is None else best_volume)
            new_population.append(offspring)

        # Relax and evaluate the new population
        new_population, new_e_above_hulls, new_energies = zip(*[
            fitness_function(structure, optimizer, convex_hull_calculator)
            for structure in tqdm(new_population, desc="Relaxing new population", total=len(new_population))
        ])

        # Merge old and new populations
        population = selected_population + list(new_population)
        e_above_hulls = selected_e_above_hulls + list(new_e_above_hulls)
        energies = selected_energies + list(new_energies)

        # Remove structures with different compositions
        valid_indices = [
            idx for idx, p in enumerate(population)
            if p.composition.reduced_formula == best_structure.composition.reduced_formula
        ]
        population = [population[i] for i in valid_indices]
        e_above_hulls = [e_above_hulls[i] for i in valid_indices]
        energies = [energies[i] for i in valid_indices]

        # Track the best structure
        best_idx = np.argmin(energies)
        best_structure = population[best_idx]
        best_e_above_hull = e_above_hulls[best_idx]
        best_fitness = energies[best_idx]
        is_in_mp = convex_hull_calculator.structure_is_in_mp(best_structure)
        best_structure.to(filename=f"outputs/best_structure_generation_{generation}.cif")

        print(f"Generation {generation}: Best Fitness = {best_fitness}, e_above_hull = {best_e_above_hull}, is_in_mp = {is_in_mp}")

    print("Optimization complete!")


if __name__ == "__main__":
    fire.Fire(run_evolutionary_optimization_algorithm)
