from abc import ABC
from typing import Callable
from models.training.optimizers import Optimizer
from models.training.losses import Loss
from models.training.metrics import Metric
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from deepxde.data.pde import PDE
from deepxde.nn.tensorflow_compat_v1.nn import NN
import deepxde.model as dde_model
import numpy as np


class Model(ABC):
    def __init__(self, num_inputs: int, hidden_units: list, num_outputs: int):
        self._num_inputs = num_inputs
        self._hidden_units = hidden_units
        self._num_outputs = num_outputs

        self._layer_sizes = [num_inputs] + hidden_units + [num_outputs]
        self._weight_initializer = 'Glorot uniform'
        self._disregard_previous_best_model = True
        self._model = None

        self._default_optimizer = Optimizer.ADAM
        self._default_learning_rate = 0.001
        self._default_loss = Loss.MSE
        self._default_metrics = None
        self._default_decay = None

    def _build(self, pde_data: PDE, net: NN):
        self._model = dde_model.Model(data=pde_data, net=net)
        self._metrics = Metric.L2_RELATIVE_ERROR if pde_data.soln is not None else None

    def compile(
            self,
            optimizer: str,
            learning_rate: float or None,
            loss: list or str,
            metrics: list or None,
            decay: LearningRateSchedule or None = None,
            trainable_parameters: list or None = None):
        assert self._model is not None, 'Model was not built.'

        self._model.compile(
            optimizer=optimizer,
            lr=learning_rate,
            loss=loss,
            metrics=metrics,
            decay=decay,
            external_trainable_variables=trainable_parameters
        )

    def compile_default(self, trainable_parameters: list or None = None):
        assert self._model is not None, 'Model was not built.'

        self._model.compile(
            optimizer=self._default_optimizer,
            lr=self._default_learning_rate,
            loss=self._default_loss,
            metrics=self._default_metrics,
            decay=self._default_decay,
            external_trainable_variables=trainable_parameters
        )

    def train(
            self,
            epochs: int or None = None,
            batch_size: int or None = None,
            display_loss_and_metrics_steps: int = 1000,
            callbacks: list or None = None,
            model_restore_path: str or None = None
    ):
        loss_history, train_state = self._model.train(
            epochs=epochs,
            batch_size=batch_size,
            display_every=display_loss_and_metrics_steps,
            disregard_previous_best=self._disregard_previous_best_model,
            callbacks=callbacks,
            model_restore_path=model_restore_path,
            model_save_path=None
        )
        return loss_history, train_state

    def predict(self, inputs, pde_function: Callable) -> np.array:
        return self._model.predict(x=inputs, operator=pde_function)

    def save(self, filepath: str):
        self._model.save(filepath, verbose=1)

    def restore(self, filepath: str):
        self._model.restore(filepath, verbose=1)
