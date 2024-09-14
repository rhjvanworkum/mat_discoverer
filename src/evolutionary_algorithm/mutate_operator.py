from pymatgen.core import Structure
import numpy as np

from src.utils import swap_atoms_in_pymatgen_structure, modify_lattice_matrix, get_cell_transformation_matrix, randomly_sample_strain_matrix


def mutate_operator(
    structure: Structure,
    rescaling_volume: float,
) -> Structure:
    """
    Mutates a offspring structure by applying a random amount of atom permutations and modifying the cell vectors.
    :param structure: offspring structure to mutate
    :param rescaling_volume: factor to rescale the volume of the structure
    
    Returns the mutated offspring structure
    """
    # 1. Permutate atoms
    n_transpositions = np.random.randint(1, len(structure) // 2)
    for _ in range(n_transpositions):
        i, j = np.random.choice(len(structure), size=2, replace=False)
        structure = swap_atoms_in_pymatgen_structure(structure, i, j)

    # 2. Mutate cell vectors (deform the lattice basically)
    structure = modify_lattice_matrix(structure, get_cell_transformation_matrix(randomly_sample_strain_matrix()))

    # TODO: add also mutation of atomic position vectors here
    
    # 3. Rescale the volume
    structure.scale_lattice(rescaling_volume)
    
    return structure