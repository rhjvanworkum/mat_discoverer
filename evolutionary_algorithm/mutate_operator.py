from pymatgen.core import Structure
import numpy as np

from utils import swap_atoms_in_pymatgen_structure, modify_lattice_matrix, get_cell_transformation_matrix


def mutate_operator(structure: Structure) -> Structure:
    """
    Mutates a offspring structure by applying a random amount of atom permutations and 
    modifying the cell vectors.
    :param structure: offspring structure to mutate
    """
    # 1. permutate atoms
    n_transpositions = np.random.randint(1, len(structure) // 2)
    for _ in range(n_transpositions):
        i, j = np.random.choice(len(structure), size=2, replace=False)
        structure = swap_atoms_in_pymatgen_structure(structure, i, j)

    # 2. mutate cell vectors
    structure = modify_lattice_matrix(structure, get_cell_transformation_matrix())

    # 3. TODO: add also mutation of atomic position vectors here
    
    return structure