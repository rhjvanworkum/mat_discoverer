import random
from pymatgen.core import Structure, Lattice

from utils import normalize_lattice


def crossover_operator(parent1: Structure, parent2: Structure) -> Structure:
    """
    """
    # Step 1: Normalize both parents to a common volume (e.g., average volume)
    target_volume = (parent1.volume + parent2.volume) / 2
    parent1 = normalize_lattice(parent1, target_volume)
    parent2 = normalize_lattice(parent2, target_volume)
    
    # Step 2: Perform crossover on atomic positions
    child_species = []
    child_coords = []
    
    # We will alternate between picking atoms from parent1 and parent2
    for i in range(max(len(parent1), len(parent2))):
        if i < len(parent1) and random.random() < 0.5:
            # Pick an atom from parent1
            child_species.append(parent1.species[i])
            child_coords.append(parent1.frac_coords[i])
        elif i < len(parent2):
            # Pick an atom from parent2
            child_species.append(parent2.species[i])
            child_coords.append(parent2.frac_coords[i])
    
    # Step 3: Create the child structure with the combined species and coordinates
    child_lattice = Lattice(parent1.lattice.matrix)  # Use the lattice from parent1 (already normalized)
    child_structure = Structure(child_lattice, child_species, child_coords)
    
    return child_structure