from enum import Enum

from torch.optim import SGD, Adam


class OptimizerType(str, Enum):
    ADAM = "adam"
    SGD = "sgd"


def get_optimizer(
    optimizer_type: OptimizerType,
    parameters: list,
    learning_rate: float = 0.01,
) -> Adam | SGD:
    match optimizer_type:
        case OptimizerType.ADAM:
            return Adam(parameters, lr=learning_rate)

        case OptimizerType.SGD:
            return SGD(parameters, lr=learning_rate, momentum=0.9)

        case _:
            raise ValueError(f"Optimizer type {optimizer_type} is not supported")
