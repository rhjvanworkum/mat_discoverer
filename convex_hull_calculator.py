from typing import Tuple
from mp_api.client import MPRester
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram


class ConvexHullCalculator:

    MAPI_KEY = "Oic28Apgrq7Ka7P0grP71S5rq7IVRP1C"

    def __init__(self, composition: Composition) -> None:
        with MPRester(self.MAPI_KEY) as mpr:
            # Obtain only corrected GGA and GGA+U
            self.entries = mpr.get_entries_in_chemsys(elements=[e.symbol for e in composition.elements], additional_criteria={"thermo_types": ["GGA_GGA+U"]})

    def __call__(self, composition: Composition, energy: float) -> Tuple[float]:
        new_entry = ComputedEntry(composition, energy)
        self.entries.append(new_entry)
        pd = PhaseDiagram(self.entries)

        decomp_energy, e_above_hull = pd.get_decomp_and_e_above_hull(new_entry)
        return decomp_energy, e_above_hull