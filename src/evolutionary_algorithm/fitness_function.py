from typing import Tuple
from pymatgen.core import Structure, Composition

from src.convex_hull_calculator import ConvexHullCalculator
from src.ml_structure_optimizer import MLStructureOptimizer


def fitness_function(
    structure: Structure,
    optimizer: MLStructureOptimizer,
    convex_hull_calculator: ConvexHullCalculator,
) -> Tuple[Structure, float, float]:
    """
    Calculates the fitness of a structure for an evolutionary algorithm. Fitness is defined as the energy per atom
    of the relaxed structure and its distance from the convex hull.

    :param structure: The pymatgen structure to optimize and evaluate.
    :param optimizer: The optimizer used for relaxing the structure.
    :param convex_hull_calculator: The convex hull calculator for evaluating the structure's stability.
    :return: A tuple containing the optimized structure, energy above the convex hull, and energy per atom.
    """
    try:
        # Optimize the structure
        opt_structure, energy = optimizer(structure, fmax=0.05, max_steps=500)

        # Calculate composition and energy above the convex hull
        composition = Composition(opt_structure.formula)
        _, e_above_hull = convex_hull_calculator(composition, energy)

        # Return optimized structure, energy above hull, and energy per atom
        return opt_structure, e_above_hull, energy / len(opt_structure)

    except Exception as e:
        # Handle any errors during optimization
        print(f"Error during optimization: {e}")
        return structure, 1000.0, 1000.0
