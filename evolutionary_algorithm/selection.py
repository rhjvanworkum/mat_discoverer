from typing import List
from pymatgen.core import Structure
import numpy as np


def selection_fn(
    population: List[Structure], 
    fitness_scores: List[float],
    selection_pressure: float = 2.0
) -> List[Structure]:
    """
    Selection function for the evolutionary algorithm. The selection is based on the fitness scores of the structures.
    """
    probabilities = (1.0 / fitness_scores) ** selection_pressure
    probabilities /= probabilities.sum()  # Normalize
    selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return [population[i] for i in selected_indices], [fitness_scores[i] for i in selected_indices]