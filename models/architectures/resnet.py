from models.model import Model
from deepxde.data.pde import PDE
import deepxde.nn.tensorflow_compat_v1 as tf_nn


class FNN(Model):
    def __init__(self, num_inputs: int, num_residual_block_neurons: int, num_residual_blocks: int, num_outputs: int):
        hidden_units = [num_residual_block_neurons] * num_residual_blocks
        super().__init__(num_inputs, hidden_units, num_outputs)

    def build(
            self,
            pde_data: PDE,
            activation: str
    ):
        net = tf_nn.ResNet(
            input_size=self._num_inputs,
            output_size=self._num_outputs,
            num_neurons=self._hidden_units[0],
            num_blocks=len(self._hidden_units),
            activation=activation,
            kernel_initializer=self._weight_initializer
        )
        self._build(pde_data=pde_data, net=net)
