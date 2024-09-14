from typing import Callable, Tuple
from ase import Atoms
from ase.filters import ExpCellFilter, FrechetCellFilter
from ase.optimize import FIRE, LBFGS
from ase.optimize.optimize import Optimizer
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from pymatgen.core import Structure

from src.utils import ase_to_pymatgen, pymatgen_to_ase

class MLStructureOptimizer:

    FILTER_CLS: Callable[[Atoms], Atoms] = {
        "frechet": FrechetCellFilter,
        "exp": ExpCellFilter,
    }
    OPTIM_CLS: Callable[..., Optimizer] = {"FIRE": FIRE, "LBFGS": LBFGS}

    def __init__(
        self,
        device: str = "cuda",
        ase_filter: str = "frechet",
        ase_optimizer: str = "FIRE",
    ):
        orb_ff = pretrained.orb_v1(device=device) # or choose another model using ORB_PRETRAINED_MODELS[model_name]()
        self.calculator = ORBCalculator(orb_ff, device=device)

        self.filter_cls = self.FILTER_CLS[ase_filter]
        self.optim_cls = self.OPTIM_CLS[ase_optimizer]

    def __call__(self, structure: Structure, fmax: float = 0.05, max_steps: int = 500) -> Tuple[Structure, float]:
        atoms = pymatgen_to_ase(structure)
        atoms.calc = self.calculator
        if max_steps > 0:
            filtered_atoms = self.filter_cls(atoms)
            optimizer = self.optim_cls(filtered_atoms, logfile="/dev/null")
            optimizer.run(fmax=fmax, steps=max_steps)
        energy = atoms.get_potential_energy()  # relaxed energy      
        return ase_to_pymatgen(atoms), energy