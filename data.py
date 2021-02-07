from typing import List, Tuple, Callable

from torch import Tensor
from torch.utils.data import Dataset, DataLoader


def shuffle_batch_loader(batch_size: int) -> Callable[[List[Tensor], List[int]], DataLoader[Tuple[Tensor, int]]]:
    def loader_for_data(items: List[Tensor], labels: List[int]) -> DataLoader[Tuple[Tensor, int]]:
        data = Data(items, labels)
        return DataLoader(data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
    return loader_for_data


class Data(Dataset[Tuple[Tensor, int]]):
    def __init__(self, items: List[Tensor], labels: List[int]):
        self.labels = labels
        self.items = items

    def __getitem__(self, index) -> Tuple[Tensor, int]:
        return self.items[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)


