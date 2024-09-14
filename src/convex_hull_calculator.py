import os
from typing import Tuple
from mp_api.client import MPRester
from pymatgen.core import Composition, Structure
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher


class ConvexHullCalculator:

    def __init__(self, composition: Composition) -> None:
        """
        This class uses the Materials Project API to calculate the convex hull of a given composition.
        :param composition: pymatgen Composition object
        """
        MAPI_KEY = os.getenv("MAPI_KEY")
        with MPRester(MAPI_KEY) as mpr:
            # Obtain only corrected GGA and GGA+U
            self.entries = mpr.get_entries_in_chemsys(elements=[e.symbol for e in composition.elements], additional_criteria={"thermo_types": ["GGA_GGA+U"]})

    def __call__(self, composition: Composition, energy: float) -> Tuple[float]:
        """
        Computes energy above the convex hull metric.
        :param composition: pymatgen Composition object
        :param energy: energy of the material
        
        Returns the decomposition energy and energy above the hull
        """
        new_entry = ComputedEntry(composition, energy)
        self.entries.append(new_entry)
        pd = PhaseDiagram(self.entries)

        decomp_energy, e_above_hull = pd.get_decomp_and_e_above_hull(new_entry)
        return decomp_energy, e_above_hull
    
    def check_if_materials_is_novel(self, discovered_structure: Structure) -> bool:
        """
        Checks if a material is already in the MP databse or is actually novel
        :param discovered_structure: pymatgen Structure object
        
        Returns True if the material is novel, False otherwise
        """
        MAPI_KEY = os.getenv("MAPI_KEY")
        with MPRester(MAPI_KEY) as mpr:
            # Get a list of structures from Materials Project by chemical formula
            structures_from_mp = mpr.get_structures(discovered_structure.composition.formula)
            
            matcher = StructureMatcher()
            for mp_structure in structures_from_mp:
                if matcher.fit(discovered_structure, mp_structure):
                    return False
            else:
                return True