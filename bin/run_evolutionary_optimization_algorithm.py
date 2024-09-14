"""
This is the main script that tries to find the global minimum of a crystal structure using an evolutionary optimization algorithm.

Example: python bin/run_evolutionary_optimization_algorithm.py --species "['Zn', 'Cu']" --num_species "[1, 1]"
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
    population_size: int = 20,
    generations: int = 50,
    mutation_rate: float = 0.1,
    selection_pressure: float = 2.0,
):
    """
    The script tries to find the global minimum of a crystal structure using an evolutionary optimization algorithm.
    It first creates a intial population of random crystal structures created with pyxtal and then relaxes them using 
    ML FF.
    
    :param species: List of atoms types in the composition
    :param num_species: List of number of atoms for each species in the composition
    :param population_size: Number of structures in the population
    :param generations: Number of generations to run the evolutionary algorithm
    :param mutation_rate: Probability of mutation for each offspring
    :param selection_pressure: Selection pressure for fitness-based selection
    """
    # setup initial stuff
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
    best_volume = None

    # relax and evaluate the initial population
    population, fitnesses = map(list, zip(*[fitness_function(structure, optimizer, convex_hull_calculator) 
                                            for structure in tqdm(initial_population, 
                                                                  desc="Relaxation of intial population", 
                                                                  total=len(initial_population))]))

    for generation in range(generations):
        print(f'Working on generation {generation + 1}...')
        
        # select parents based on fitness
        selected_population, selected_fitnesses = selection_fn(population, fitnesses, selection_pressure)

        # Create new population via crossover and mutation
        new_population = []
        for _ in range(population_size):
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)
            offspring = crossover_operator(parent1, parent2)
            if np.random.rand() < mutation_rate:
                offspring = mutate_operator(offspring, rescaling_volume=parent1.volume if best_volume is None else best_volume)
            new_population.append(offspring)

        # relax and evaluate the new population 
        new_population, new_fitnesses = map(list, zip(*[fitness_function(structure, optimizer, convex_hull_calculator) 
                                            for structure in tqdm(new_population, 
                                                                  desc="Relaxation of new population", 
                                                                  total=len(new_population))]))
        population = selected_population + new_population
        fitnesses = selected_fitnesses + new_fitnesses

        # Track the best structure
        best_structure = population[np.argmin(fitnesses)]
        best_fitness = np.min(fitnesses)
        best_volume = best_structure.volume
        is_novel = convex_hull_calculator.check_if_materials_is_novel(best_structure)
        best_structure.to(filename=f"outputs/best_structure_{generation}.cif")
        print(f"Generation {generation}: Best Fitness / e_above_hull = {best_fitness}, is_novel: {is_novel}")
        
    print("Finished!")

if __name__ == "__main__":
    fire.Fire(run_evolutionary_optimization_algorithm)