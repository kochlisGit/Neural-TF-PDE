import deepxde.callbacks as callbacks


class BestModelCheckpoint(callbacks.ModelCheckpoint):
    def __init__(self, filepath: str):
        super().__init__(filepath, save_better_only=True)


class EarlyStopping(callbacks.EarlyStopping):
    def __init__(self, min_loss_improvement: float, patience_epochs: int):
        super().__init__(min_delta=min_loss_improvement, patience=patience_epochs)


class ParameterCheckpoint(callbacks.VariableValue):
    def __init__(self, trainable_parameters: list, filepath: str, epochs: int):
        super().__init__(var_list=trainable_parameters, filename=filepath, period=epochs)


class PDEInputResampling(callbacks.PDEResidualResampler):
    def __init__(self, epochs: int):
        super().__init__(period=epochs)
