from typing import Callable
from deepxde.geometry.geometry import Geometry
import deepxde.icbcs as icbcs
import numpy as np


def _is_boundary(x: np.ndarray, boundary: np.ndarray):
    for x_val, b_val in zip(x, boundary):
        if x_val != b_val:
            return False
    return True


class Dirichlet(icbcs.DirichletBC):
    def __init__(self, geometry: Geometry, output_func: Callable, boundary: list, output_var_id: int):
        super().__init__(
            geom=geometry,
            func=output_func,
            on_boundary=lambda x, _: _is_boundary(x, np.float32(boundary)),
            component=output_var_id
        )


class Neumann(icbcs.NeumannBC):
    def __init__(self, geometry: Geometry, output_func: Callable, boundary: list, output_var_id: int):
        super().__init__(
            geom=geometry,
            func=output_func,
            on_boundary=lambda x, _: _is_boundary(x, np.float32(boundary)),
            component=output_var_id
        )


class Robin(icbcs.RobinBC):
    def __init__(self, geometry: Geometry, output_func: Callable, boundary: list, output_var_id: int):
        super().__init__(
            geom=geometry,
            func=output_func,
            on_boundary=lambda x, _: _is_boundary(x, np.float32(boundary)),
            component=output_var_id
        )


class Operator(icbcs.OperatorBC):
    def __init__(self, geometry: Geometry, output_func: Callable, boundary: list):
        super().__init__(
            geom=geometry,
            func=output_func,
            on_boundary=lambda x, _: _is_boundary(x, np.float32(boundary))
        )


class PointSet(icbcs.PointSetBC):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, output_var_id: int):
        super().__init__(
            points=inputs,
            values=outputs,
            component=output_var_id
        )
