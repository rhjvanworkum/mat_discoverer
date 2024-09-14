from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
import ase


def normalize_lattice(structure: Structure, target_volume: float) -> Structure:
    """
    Scale the lattice to have a target volume while maintaining the original lattice ratios.
    """
    scale_factor = (target_volume / structure.volume) ** (1/3)
    new_lattice = Lattice(structure.lattice.matrix * scale_factor)
    return Structure(new_lattice, structure.species, structure.frac_coords)

def ase_to_pymatgen(structure: ase.Atoms) -> Structure:
    """
    Convert an ASE Atoms object to a pymatgen Structure object.
    """
    pymatgen_structure = AseAtomsAdaptor.get_structure(structure)
    return pymatgen_structure

def pymatgen_to_ase(structure: Structure) -> ase.Atoms:
    """
    Convert a pymatgen Structure object to an ASE Atoms object.
    """
    ase_structure = AseAtomsAdaptor.get_atoms(structure)
    return ase_structure

def swap_atoms_in_pymatgen_structure(structure: Structure, i: int, j: int) -> Structure:
    """
    Swap the atoms at indices i and j in a pymatgen Structure object.
    """
    species_i = structure[i].species_string
    coords_i = structure[i].frac_coords
    species_j = structure[j].species_string
    coords_j = structure[j].frac_coords

    structure.replace(i, species_j, coords_i)
    structure.replace(j, species_i, coords_j)

    return structure

def modify_lattice_matrix(structure: Structure, strain_matrix: np.ndarray) -> Structure:
    """
    Modify the lattice matrix of a pymatgen Structure object using a strain matrix.
    """
    lattice_matrix = structure.lattice.matrix
    strained_lattice_matrix = np.dot(lattice_matrix, strain_matrix)
    new_lattice = Lattice(strained_lattice_matrix)
    return Structure(new_lattice, structure.species, structure.frac_coords)

def randomly_sample_strain_matrix() -> np.ndarray:
    """
    Randomly sample a 3x3 strain matrix.
    """
    return np.random.uniform(low=-1, high=1, size=(3, 3))

def get_cell_transformation_matrix(strain_matrix: np.ndarray) -> np.ndarray:
    """
    Get the transformation matrix from a strain matrix.
    """
    return np.identity(3) + (0.5 * strain_matrix + 0.5 * np.diag(np.diag(strain_matrix)))
