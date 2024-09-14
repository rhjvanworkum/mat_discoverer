import random
from pyxtal import pyxtal
from pyxtal.msg import Comp_CompatibilityError
from pymatgen.core import Structure
from typing import List


def generate_random_crystal_structure_from_composition(
    species: List[str],
    num_species: List[int],
) -> Structure:
    """
    This function is meant to sample random crystal structures from a given composition using PyXtal.
    The inputs of a structure generation function are basically:
    1) The spacegroup
    2) The composition
    3) The volume factor
    4) The lattice parameters

    Luckily PyXtal generates random lattice parameters for us consistent with the symmetry of the space group.
    Therefore, we only need to randomly sample the space group, number of atoms and volume factor.
    """
    volume_factor = random.uniform(0.9, 1.1)
    composition_frac = 1 / min(num_species)
    composition_multiplicative_factor = random.randint(1, 3)

    success = False
    while not success:
        try:
            my_crystal = pyxtal()
            my_crystal.from_random(
                dim=3,
                group=random.randint(1, 230),
                species=species,
                numIons=[int(num * composition_multiplicative_factor * composition_frac) for num in num_species],
                factor=volume_factor,
            )
            success = True
        except Comp_CompatibilityError as e:
            continue

    return my_crystal.to_pymatgen()