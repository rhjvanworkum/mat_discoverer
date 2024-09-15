import random
import numpy as np
from pymatgen.core import Structure, Lattice
from src.utils import normalize_lattice

# TODO: perform slicing in truly random directions, not just in the x, y, or z direction

def crossover_operator(parent_structure_1: Structure, parent_structure_2: Structure) -> Structure:
    """
    Generates a child structure from two parent structures by performing a crossover on their atomic positions.

    :param parent_structure_1: The first parent structure.
    :param parent_structure_2: The second parent structure.
    :return: A child structure that combines atomic positions and lattice vectors from both parents.
    """
    # Step 1: Normalize both parent structures to a common volume (average volume)
    target_volume = (parent_structure_1.volume + parent_structure_2.volume) / 2
    parent_structure_1 = normalize_lattice(parent_structure_1, target_volume)
    parent_structure_2 = normalize_lattice(parent_structure_2, target_volume)

    # Step 2: Perform crossover on atomic positions
    child_species, child_coords = _crossover_atomic_positions(parent_structure_1, parent_structure_2)

    # Step 3: Create the child lattice using a weighted average of the parent lattices
    child_lattice = _create_weighted_lattice(parent_structure_1.lattice, parent_structure_2.lattice)

    # Return the final child structure
    return Structure(child_lattice, child_species, child_coords)


def _crossover_atomic_positions(parent_structure_1: Structure, parent_structure_2: Structure):
    """
    Performs a crossover on the atomic positions of two parent structures.

    :param parent_structure_1: The first parent structure.
    :param parent_structure_2: The second parent structure.
    :return: A tuple containing the child species and child coordinates.
    """
    child_species = []
    child_coords = []

    slicing_dim = random.randint(0, 2)
    slicing_threshold = random.uniform(0, 1)

    # Add species and coordinates from parent 1 based on slicing threshold
    parent_1_indices = _get_slicing_indices(parent_structure_1, slicing_dim, slicing_threshold, direction="below")
    child_species.extend(parent_structure_1.species[i] for i in parent_1_indices)
    child_coords.extend(parent_structure_1.frac_coords[i] for i in parent_1_indices)

    # Add species and coordinates from parent 2 based on slicing threshold
    parent_2_indices = _get_slicing_indices(parent_structure_2, slicing_dim, slicing_threshold, direction="above")
    child_species.extend(parent_structure_2.species[i] for i in parent_2_indices)
    child_coords.extend(parent_structure_2.frac_coords[i] for i in parent_2_indices)

    return child_species, child_coords


def _get_slicing_indices(structure: Structure, slicing_dim: int, slicing_threshold: float, direction: str):
    """
    Gets the slicing indices for a given structure based on the slicing threshold and direction.

    :param structure: The structure to slice.
    :param slicing_dim: The dimension to slice along (0 for x, 1 for y, 2 for z).
    :param slicing_threshold: The fractional threshold used for slicing.
    :param direction: Whether to select indices 'above' or 'below' the slicing threshold.
    :return: An array of indices that satisfy the slicing condition.
    """
    if direction == "below":
        return np.where(structure.frac_coords[:, slicing_dim] <= slicing_threshold * structure.frac_coords[:, slicing_dim].max())[0]
    else:
        return np.where(structure.frac_coords[:, slicing_dim] > slicing_threshold * structure.frac_coords[:, slicing_dim].max())[0]


def _create_weighted_lattice(lattice_1: Lattice, lattice_2: Lattice) -> Lattice:
    """
    Creates a child lattice as a weighted average of two parent lattices.

    :param lattice_1: The first parent lattice.
    :param lattice_2: The second parent lattice.
    :return: The new child lattice.
    """
    weight = random.random()  # Random weight between 0 and 1
    child_lattice_matrix = (1 - weight) * lattice_1.matrix + weight * lattice_2.matrix
    return Lattice(child_lattice_matrix)
