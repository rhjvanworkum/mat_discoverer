from pymatgen.core import Structure
import numpy as np

from src.utils import (
    swap_atoms_in_pymatgen_structure,
    modify_lattice_matrix,
    get_cell_transformation_matrix,
    randomly_sample_strain_matrix,
    modify_atomic_positions,
)


def mutate_operator(
    structure: Structure,
    rescaling_volume: float,
) -> Structure:
    """
    Applies mutations to an offspring structure by permuting atoms, modifying cell vectors, and rescaling the volume.

    :param structure: Offspring structure to mutate.
    :param rescaling_volume: Factor to rescale the volume of the structure.
    :return: The mutated offspring structure.
    """
    # 1. Permute atoms
    structure = _permute_atoms(structure)

    # 2. Mutate cell vectors (deform the lattice) and atomic positions
    structure = _mutate_lattice_and_positions(structure)

    # 3. Rescale the volume
    structure.scale_lattice(rescaling_volume)

    return structure


def _permute_atoms(structure: Structure) -> Structure:
    """
    Permutes atoms in the structure by randomly swapping atom positions.

    :param structure: The structure in which to permute atoms.
    :return: The structure after atom permutation.
    """
    num_atoms = len(structure)
    n_transpositions = 1 if num_atoms < 3 else np.random.randint(1, max(2, num_atoms // 2))

    for _ in range(n_transpositions):
        i, j = np.random.choice(num_atoms, size=2, replace=False)
        structure = swap_atoms_in_pymatgen_structure(structure, i, j)

    return structure


def _mutate_lattice_and_positions(structure: Structure) -> Structure:
    """
    Applies random strain to the lattice and modifies the atomic positions accordingly.

    :param structure: The structure to modify.
    :return: The structure with mutated lattice and atomic positions.
    """
    strain_matrix = randomly_sample_strain_matrix()
    transformation_matrix = get_cell_transformation_matrix(strain_matrix)

    # Modify lattice vectors
    structure = modify_lattice_matrix(structure, transformation_matrix)

    # Modify atomic positions
    structure = modify_atomic_positions(structure, strain_matrix)

    return structure
