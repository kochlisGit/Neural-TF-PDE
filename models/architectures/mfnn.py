from models.model import Model
from deepxde.data.pde import PDE
import deepxde.nn.tensorflow_compat_v1 as tf_nn


class MFNN(Model):
    def __init__(self, num_inputs: int, hidden_units: list, num_outputs: int):
        super().__init__(num_inputs, hidden_units, num_outputs)

    def build(
            self,
            pde_data: PDE,
            activation: str
    ):
        net = tf_nn.MfNN(
            layer_sizes_low_fidelity=self._layer_sizes,
            layer_sizes_high_fidelity=self._layer_sizes,
            activation=activation,
            kernel_initializer=self._weight_initializer
        )
        self._build(pde_data=pde_data, net=net)
