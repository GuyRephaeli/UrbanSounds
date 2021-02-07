from typing import List, Callable, Tuple, Iterator
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module, Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Model(object):
    """
    A class to represent a full model. Exposes fit for training and predict for prediction.
    This class enables flexible composition of model parts such as data loader, optimizer and network architecture.
    """
    def __init__(self, device,
                 nn: Module,
                 loader_for_data: Callable[[List[Tensor], List[int]], DataLoader[Tuple[Tensor, int]]],
                 optimizer_for_params: Callable[[Iterator[Parameter]], Optimizer],
                 epochs: int):
        self.nn = nn
        self.device = device
        self.nn.to(self.device)
        self.loader_for_data = loader_for_data
        self.optimizer = optimizer_for_params(filter(lambda p: p.requires_grad, self.nn.parameters()))
        self.criterion = CrossEntropyLoss()
        self.epochs = epochs

    def fit(self, items: List[Tensor], labels: List[int]):
        loader = self.loader_for_data(items, labels)
        self.nn.train()
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
            for batch_items, batch_labels in loader:
                batch_items = batch_items.to(self.device)
                batch_labels = batch_labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.nn(batch_items)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

    def predict(self, items: List[Tensor]) -> List[int]:
        def predict_item(item: Tensor, model: Module, device) -> int:
            with torch.no_grad():
                item = item.unsqueeze(0).to(device)
                outputs = model(item)
                _, predicted = torch.max(outputs, 1)
            return predicted.detach().cpu().numpy()[0]

        self.nn.eval()
        return [predict_item(item, self.nn, self.device) for item in items]


class ModelFactory(object):
    """
    A class for creating identical models that can be trained independently.
    """
    def __init__(self, device,
                 create_nn: Callable[[], Module],
                 loader_for_data: Callable[[List[Tensor], List[int]], DataLoader[Tuple[Tensor, int]]],
                 optimizer_for_params: Callable[[Iterator[Parameter]], Optimizer],
                 epochs: int):
        self.create_nn = create_nn
        self.device = device
        self.loader_for_data = loader_for_data
        self.optimizer_for_params = optimizer_for_params
        self.epochs = epochs

    def get(self) -> Model:
        return Model(self.device,
                     self.create_nn(),
                     self.loader_for_data,
                     self.optimizer_for_params,
                     self.epochs)
