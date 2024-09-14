import random
from pymatgen.core import Structure, Lattice

from src.utils import normalize_lattice


def crossover_operator(parent_structure_1: Structure, parent_structure_2: Structure) -> Structure:
    """
    Operator that creates a child between two parent structures.
    
    :param parent_structure_1: First parent structure
    :param parent_structure_2: Second parent structure
    
    Returns a child structure that is a combination of the two parent structures.
    """
    # Step 1: Normalize both parents to a common volume (e.g., average volume)
    target_volume = (parent_structure_1.volume + parent_structure_2.volume) / 2
    parent_structure_1 = normalize_lattice(parent_structure_1, target_volume)
    parent_structure_2 = normalize_lattice(parent_structure_2, target_volume)
    
    # Step 2: Perform crossover on atomic positions
    child_species = []
    child_coords = []
    
    # We will alternate between picking atoms from parent_structure_1 and parent_structure_2
    for i in range(max(len(parent_structure_1), len(parent_structure_2))):
        if i < len(parent_structure_1) and random.random() < 0.5:
            # Pick an atom from parent_structure_1
            child_species.append(parent_structure_1.species[i])
            child_coords.append(parent_structure_1.frac_coords[i])
        elif i < len(parent_structure_2):
            # Pick an atom from parent_structure_2
            child_species.append(parent_structure_2.species[i])
            child_coords.append(parent_structure_2.frac_coords[i])
    
    # Step 3: Create the child structure with the combined species and coordinates
    child_lattice = Lattice(parent_structure_1.lattice.matrix)  # Use the lattice from parent1 (already normalized)
    child_structure = Structure(child_lattice, child_species, child_coords)
    
    return child_structure