from typing import Callable, Iterator

from torch.nn import Parameter
from torch.optim import Optimizer, Adam, SGD


def simple_adam_optimizer() -> Callable[[Iterator[Parameter]], Optimizer]:
    def optimizer_for_params(params: Iterator[Parameter]) -> Optimizer:
        return Adam(params)
    return optimizer_for_params


def sgd_with_momentum(lr: float, momentum: float) -> Callable[[Iterator[Parameter]], Optimizer]:
    def optimizer_for_params(params: Iterator[Parameter]) -> Optimizer:
        return SGD(params, lr=lr, momentum=momentum)
    return optimizer_for_params
