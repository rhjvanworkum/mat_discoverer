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
    """
    Class for optimizing structures using a machine learning forcefield, specifically the ORB model.

    Attributes:
        FILTER_CLS: Mapping of filter names to ASE filter classes.
        OPTIM_CLS: Mapping of optimizer names to ASE optimizer classes.
    """
    
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
        Initializes the optimizer with a pretrained ORB model.

        :param device: Device to run the model on ('cuda' or 'cpu').
        :param ase_filter: Filter to apply to the ASE Atoms object before optimization.
        :param ase_optimizer: Optimizer to use for the structure optimization.
        """
        self.calculator = self._initialize_calculator(device)
        self.filter_cls = self.FILTER_CLS[ase_filter]
        self.optim_cls = self.OPTIM_CLS[ase_optimizer]

    @staticmethod
    def _initialize_calculator(device: str) -> ORBCalculator:
        """
        Initializes the ORB calculator using the pretrained ORB model.

        :param device: Device to run the model on.
        :return: Initialized ORBCalculator object.
        """
        orb_ff = pretrained.orb_v1(device=device)  # Pretrained model on MPTraj + Alexandria dataset
        return ORBCalculator(orb_ff, device=device)

    def __call__(self, structure: Structure, fmax: float = 0.05, max_steps: int = 500) -> Tuple[Structure, float]:
        """
        Optimizes a given pymatgen structure using the machine learning forcefield.

        :param structure: The pymatgen Structure object to optimize.
        :param fmax: Maximum force tolerance for optimization.
        :param max_steps: Maximum number of optimization steps.
        :return: A tuple of the optimized pymatgen Structure and the final energy.
        """
        atoms = pymatgen_to_ase(structure)
        atoms.calc = self.calculator

        if max_steps > 0:
            self._optimize_structure(atoms, fmax, max_steps)

        energy = atoms.get_potential_energy()  # Relaxed energy after optimization
        return ase_to_pymatgen(atoms), energy

    def _optimize_structure(self, atoms: Atoms, fmax: float, max_steps: int) -> None:
        """
        Runs the optimization process using the specified filter and optimizer.

        :param atoms: ASE Atoms object to optimize.
        :param fmax: Maximum force tolerance.
        :param max_steps: Maximum number of steps allowed for the optimizer.
        """
        filtered_atoms = self.filter_cls(atoms)
        optimizer = self.optim_cls(filtered_atoms, logfile="/dev/null")
        optimizer.run(fmax=fmax, steps=max_steps)
