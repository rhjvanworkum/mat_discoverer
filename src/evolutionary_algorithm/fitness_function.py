from typing import Tuple
from pymatgen.core import Structure
from pymatgen.core import Composition

from src.convex_hull_calculator import ConvexHullCalculator
from src.ml_structure_optimizer import MLStructureOptimizer


def fitness_function(
    structure: Structure,
    optimizer: MLStructureOptimizer,
    convex_hull_calculator: ConvexHullCalculator,
) -> Tuple[Structure, float]:
    """
    Fitness function for the evolutionary algorithm. The fitness is calculated as e_above_hull of the 
    relaxed structure.
    :param structure: pymatgen structure to optimize and evaluate
    :param optimizer: structure optimizer
    :param convex_hull_calculator: convex hull calculator
    
    Returns the optimized structure and the e_above_hull
    """
    optimizer = MLStructureOptimizer()
    opt_structure, energy = optimizer(structure, fmax=0.05, max_steps=500)
    composition = Composition(opt_structure.reduced_formula)
    _, e_above_hull = convex_hull_calculator(composition, energy)
    return opt_structure, e_above_hull