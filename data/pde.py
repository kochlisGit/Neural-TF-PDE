from typing import Callable
from data.sampling import TrainDistribution
from deepxde.geometry.geometry import Geometry
import deepxde.data.pde as pde


class PDE(pde.PDE):
    def __init__(
            self,
            geometry: Geometry,
            pde_function: Callable,
            bcs: list,
            num_domain_points: int,
            num_boundary_points: int,
            num_test_points: int or None,
            train_points_distribution: str = TrainDistribution.SOBOL_SEQUENCE,
            additional_train_points: list or None = None,
            excluded_points: list or None = None,
            solution: Callable or None = None
    ):
        self._auxiliary_var_function = None

        super().__init__(
            geometry=geometry,
            pde=pde_function,
            bcs=bcs,
            num_domain=num_domain_points,
            num_boundary=num_boundary_points,
            train_distribution=train_points_distribution,
            anchors=additional_train_points,
            exclusions=excluded_points,
            solution=solution,
            num_test=num_test_points,
            auxiliary_var_function=self._auxiliary_var_function
        )
