from typing import Tuple
from pymatgen.core import Structure
from pymatgen.core import Composition

from convex_hull_calculator import ConvexHullCalculator
from ml_structure_optimizer import MLStructureOptimizer
from utils import pymatgen_to_ase, ase_to_pymatgen

def fitness_function(
    structure: Structure,
    optimizer: MLStructureOptimizer,
    convex_hull_calculator: ConvexHullCalculator,
) -> Tuple[Structure, float]:
    """
    Fitness function for the evolutionary algorithm. The fitness is calculated as 1 - e_above_hull of the 
    relaxed structure
    :param structure: pymatgen structure to optimize and evaluate
    :param optimizer: structure optimizer
    :param convex_hull_calculator: convex hull calculator
    """
    optimizer = MLStructureOptimizer()
    opt_strucutre, energy = optimizer(pymatgen_to_ase(structure), fmax=0.05, max_steps=500)
    energy_per_atom = energy / len(opt_strucutre)
    composition = Composition(opt_strucutre.reduced_formula)
    decomp_energy, e_above_hull = convex_hull_calculator(composition, energy)
    return ase_to_pymatgen(opt_strucutre), 1 - e_above_hull