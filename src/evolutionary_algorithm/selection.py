from typing import List
from pymatgen.core import Structure
import numpy as np


def selection_fn(
    fitness_scores: List[float],
    n_to_select: int,
    selection_pressure: float = 2.0,
) -> List[Structure]:
    """
    Selects structures based on their fitness scores using fitness-proportionate selection.

    The lower the fitness score, the better the structure. Selection pressure controls how strongly selection favors
    the fittest structures.

    :param fitness_scores: List of fitness scores for each structure.
    :param n_to_select: Number of structures to select.
    :param selection_pressure: Controls the strength of selection (higher values favor fitter structures).
    :return: Indices of the selected structures.
    """
    # Shift fitness scores to avoid zero or negative values
    adjusted_fitness_scores = np.array(fitness_scores) + 1e-8  # Prevent division by zero

    # Compute selection probabilities
    probabilities = (1.0 / adjusted_fitness_scores) ** selection_pressure
    probabilities /= probabilities.sum()  # Normalize probabilities

    # Select structures based on probabilities
    return np.random.choice(len(fitness_scores), size=n_to_select, p=probabilities)
