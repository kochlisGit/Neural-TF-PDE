from models.model import Model
from deepxde.data.pde import PDE
import deepxde.nn.tensorflow_compat_v1 as tf_nn


class FNN(Model):
    def __init__(self, num_inputs: int, hidden_units: list, num_outputs: int):
        super().__init__(num_inputs, hidden_units, num_outputs)

    def build(
            self,
            pde_data: PDE,
            activation: str,
            dropout_rate: float = 0,
            batch_normalization: bool = False
    ):
        net = tf_nn.FNN(
            layer_sizes=self._layer_sizes,
            activation=activation,
            kernel_initializer=self._weight_initializer,
            dropout_rate=dropout_rate,
            batch_normalization='before' if batch_normalization else None
        )
        self._build(pde_data=pde_data, net=net)
