from dataclasses import dataclass

from music.common.hyperparameters import HyperParameters


@dataclass
class StepScalarData:
    step: int
    data: dict[str, float]

class MlFlowLogger:
    def __init__(self, exp: str):
        import mlflow
        self._mlflow = mlflow
        self._exp = exp

    def log_hp(self, hp: HyperParameters):
        param_dict = hp.model_dump()
        self._mlflow.log_params(param_dict)

    def log_scalars(self, log_data: list[StepScalarData]):
        for ld in log_data:
            self._mlflow.log_metrics(metrics=ld.data, step=ld.step)
