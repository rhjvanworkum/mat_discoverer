import random
import numpy as np
from pymatgen.core import Structure, Lattice

from src.utils import normalize_lattice

# TODO: perform slicing in truly random directions not just in the x, y, or z direction


def crossover_operator(parent_structure_1: Structure, parent_structure_2: Structure) -> Structure:
    """
    Operator that creates a child between two parent structures.
    
    :param parent_structure_1: First parent structure
    :param parent_structure_2: Second parent structure
    
    Returns a child structure that is a combination of the two parent structures.
    """
    child_species = []
    child_coords = []

    # Step 1: Normalize both parents to a common volume (average volume)
    target_volume = (parent_structure_1.volume + parent_structure_2.volume) / 2
    parent_structure_1 = normalize_lattice(parent_structure_1, target_volume)
    parent_structure_2 = normalize_lattice(parent_structure_2, target_volume)

    # Step 2: Perform crossover on atomic positions
    slicing_dim = random.randint(0, 2)
    slicing_threshold = random.uniform(0, 1)

    # Get slicing indices for parent 1 based on slicing threshold
    parent_1_threshold = slicing_threshold * parent_structure_1.frac_coords[:, slicing_dim].max()
    parent_1_indices = np.where(parent_structure_1.frac_coords[:, slicing_dim] <= parent_1_threshold)[0]

    # Add species and coordinates from parent 1
    child_species.extend(parent_structure_1.species[i] for i in parent_1_indices)
    child_coords.extend(parent_structure_1.frac_coords[i] for i in parent_1_indices)

    # Get slicing indices for parent 2 based on slicing threshold
    parent_2_threshold = slicing_threshold * parent_structure_2.frac_coords[:, slicing_dim].max()
    parent_2_indices = np.where(parent_structure_2.frac_coords[:, slicing_dim] > parent_2_threshold)[0]

    # Add species and coordinates from parent 2
    child_species.extend(parent_structure_2.species[i] for i in parent_2_indices)
    child_coords.extend(parent_structure_2.frac_coords[i] for i in parent_2_indices)

    # Step 3: Create the child structure using a weighted average of the parent lattices
    weight = random.random()  # Random weighted average for lattice
    child_lattice_matrix = (1 - weight) * parent_structure_1.lattice.matrix + weight * parent_structure_2.lattice.matrix
    child_lattice = Lattice(child_lattice_matrix)

    # Create the final child structure
    child_structure = Structure(child_lattice, child_species, child_coords)

    return child_structure