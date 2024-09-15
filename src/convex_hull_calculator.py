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
        Initializes the ConvexHullCalculator class, which uses the Materials Project API to calculate the 
        convex hull for a given composition.

        :param composition: pymatgen Composition object
        """
        self.entries = self._fetch_entries(composition)

    @staticmethod
    def _fetch_entries(composition: Composition):
        """
        Fetches the thermodynamic entries from the Materials Project database for the given composition's chemical system.

        :param composition: pymatgen Composition object
        :return: List of computed entries for the chemical system
        """
        MAPI_KEY = os.getenv("MAPI_KEY")
        with MPRester(MAPI_KEY) as mpr:
            return mpr.get_entries_in_chemsys(
                elements=[e.symbol for e in composition.elements],
                additional_criteria={"thermo_types": ["GGA_GGA+U"]}
            )

    def __call__(self, composition: Composition, energy: float) -> Tuple[float, float]:
        """
        Computes the decomposition energy and energy above the convex hull for a given material.

        :param composition: pymatgen Composition object
        :param energy: Energy of the material
        :return: Tuple containing decomposition energy and energy above the convex hull
        """
        new_entry = ComputedEntry(composition, energy)
        self.entries.append(new_entry)
        pd = PhaseDiagram(self.entries)

        return pd.get_decomp_and_e_above_hull(new_entry)

    def structure_is_in_mp(self, discovered_structure: Structure) -> bool:
        """
        Checks if the provided structure already exists in the Materials Project database.

        :param discovered_structure: pymatgen Structure object
        :return: True if the structure is novel, False if it already exists in the database
        """
        structures_from_mp = self._fetch_structures(discovered_structure)
        matcher = StructureMatcher()

        return not any(matcher.fit(discovered_structure, mp_structure) for mp_structure in structures_from_mp)

    @staticmethod
    def _fetch_structures(discovered_structure: Structure):
        """
        Fetches structures from the Materials Project database based on the formula of the discovered structure.

        :param discovered_structure: pymatgen Structure object
        :return: List of structures from the Materials Project database
        """
        MAPI_KEY = os.getenv("MAPI_KEY")
        with MPRester(MAPI_KEY) as mpr:
            return mpr.get_structures(discovered_structure.composition.formula)
