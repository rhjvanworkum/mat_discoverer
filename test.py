
from src.convex_hull_calculator import ConvexHullCalculator
from pymatgen.core import Composition

convex_hull_calculator = ConvexHullCalculator(
    composition=Composition({'Li': 4, 'O': 1})
)

convex_hull_calculator.check_if_materials_is_novel()