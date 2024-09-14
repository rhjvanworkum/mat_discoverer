from typing import Callable, Tuple
import torch
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        ase_filter: str = "frechet",
        ase_optimizer: str = "FIRE",
    ):
        """
        Class for optimizing structures using a machine learning forcefield, in this case orb model.
        
        :param device: Device to run the model on
        :param ase_filter: Filter to apply to the atoms object before optimization
        :param ase_optimizer: Optimizer to use for the optimization
        """
        orb_ff = pretrained.orb_v1(device=device) # model trained on MPTraj + Alexandria dataset, SOTA on matbench
        self.calculator = ORBCalculator(orb_ff, device=device)

        self.filter_cls = self.FILTER_CLS[ase_filter]
        self.optim_cls = self.OPTIM_CLS[ase_optimizer]

    def __call__(self, structure: Structure, fmax: float = 0.05, max_steps: int = 500) -> Tuple[Structure, float]:
        """
        Optimize a structure using the machine learning forcefield
        :param structure: pymatgen structure to optimize
        :param fmax: Maximum force tolerance for the optimization
        :param max_steps: Maximum number of optimization steps
        
        Returns the optimized structure and the energy
        """
        atoms = pymatgen_to_ase(structure)
        atoms.calc = self.calculator
        if max_steps > 0:
            filtered_atoms = self.filter_cls(atoms)
            optimizer = self.optim_cls(filtered_atoms, logfile="/dev/null")
            optimizer.run(fmax=fmax, steps=max_steps)
        energy = atoms.get_potential_energy()  # relaxed energy      
        return ase_to_pymatgen(atoms), energy