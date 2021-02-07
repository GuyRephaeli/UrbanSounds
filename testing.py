import os
import glob
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import Tensor

from model import ModelFactory
import numpy as np


def test_model(model_factory: ModelFactory, spectrogram_path: str, all_folds: List[int]) -> Tuple[float, float, np.ndarray]:
    """
    Test a specific model on all folds.
    For each fold a new model is trained on all other folds and then tested on the
    fold and on all the other folds to yield train accuracy, test accuracy and test confusion matrix.
    The result for all the folds are then aggregated: accuracies are averaged and confusion matrices are summed.
    :param model_factory: a factory for a specific model
    :param spectrogram_path: the path for the preprocessed data
    :param all_folds: list of all fold indices
    :return: train accuracy, test accuracy and test confusion matrix
    """
    train_folds_of_fold = {f: other_folds(f, all_folds) for f in all_folds}
    files_of_fold = {f: files_in_fold(f, spectrogram_path) for f in all_folds}
    data_of_fold = {fold: load_data(files) for fold, files in files_of_fold.items()}
    train_accuracies, test_accuracies, test_confusions = tuple(zip(
        *[test_on_fold(fold, data_of_fold, train_folds_of_fold, model_factory) for fold in all_folds]))
    train_accuracy = np.average(train_accuracies)
    test_accuracy = np.average(test_accuracies)
    test_confusion = np.sum(test_confusions, 0)
    return train_accuracy, test_accuracy, test_confusion


def other_folds(test_fold: int, all_folds: List[int]):
    return [f for f in all_folds if f is not test_fold]


def files_in_fold(fold: int, spectrogram_path: str) -> List[str]:
    return [os.path.normpath(path) for path in glob.glob(f"{spectrogram_path}/fold{fold}/*")]


def load_data(paths: List[str]) -> Tuple[List[Tensor], List[int]]:
    pairs = [data_from_path(path) for path in paths]
    items = [item for item, _ in pairs]
    labels = [label for _, label in pairs]
    return items, labels


def data_from_path(path: str) -> Tuple[Tensor, int]:
    return torch.load(path), label_from_path(path)


def label_from_path(path: str) -> int:
    name = Path(path).stem
    return int(str.split(name, '-')[1])


def test_on_fold(fold: int,
                 data_of_fold: Dict[int, Tuple[List[Tensor], List[int]]],
                 train_folds_of_fold: Dict[int, List[int]],
                 model_factory: ModelFactory) -> Tuple[float, float, np.ndarray]:
    print(f"Testing fold {fold}")
    train_items, train_labels = train_data_for_fold(fold, data_of_fold, train_folds_of_fold)
    test_items, test_labels = data_of_fold[fold]

    model = model_factory.get()
    model.fit(train_items, train_labels)

    train_prediction = model.predict(train_items)
    test_prediction = model.predict(test_items)

    train_accuracy = accuracy_score(train_labels, train_prediction)
    test_accuracy = accuracy_score(test_labels, test_prediction)
    test_confusion = confusion_matrix(test_labels, test_prediction)
    return train_accuracy, test_accuracy, test_confusion


def train_data_for_fold(fold: int,
                        data_of_fold: Dict[int, Tuple[List[Tensor], List[int]]],
                        train_folds_of_fold: Dict[int, List[int]]) -> Tuple[List[Tensor], List[int]]:
    items_of_fold = {f: items for f, (items, _) in data_of_fold.items()}
    labels_of_fold = {f: labels for f, (_, labels) in data_of_fold.items()}
    train_items = sum([items_of_fold[f] for f in train_folds_of_fold[fold]], [])
    train_labels = sum([labels_of_fold[f] for f in train_folds_of_fold[fold]], [])
    return train_items, train_labels
