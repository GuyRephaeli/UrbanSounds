import os

import torch

from data import shuffle_batch_loader
from model import ModelFactory
from nn import frozen_resnet18, pretrained_resnet18
from optim import simple_adam_optimizer, sgd_with_momentum
from preprocessing import preprocess
from testing import test_model

if __name__ == '__main__':
    all_folds = list(range(1, 11))

    base_path = os.path.abspath("UrbanSound8K")
    audio_path = os.path.normpath(f"{base_path}/audio")
    spectrogram_path = os.path.normpath(f"{base_path}/spectrogram")

    if not os.path.exists(spectrogram_path):
        resample_rate = 20050
        mel_filters = 64
        seconds = 3
        preprocess(audio_path, spectrogram_path, all_folds, resample_rate, mel_filters, seconds)

    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')

    classes = 10
    batch_size = 8
    epochs = 6
    lr = 0.001
    momentum = 0.9

    model_factory = ModelFactory(device,
                                 pretrained_resnet18(classes),
                                 shuffle_batch_loader(batch_size),
                                 sgd_with_momentum(lr, momentum),
                                 epochs)

    train_accuracy, test_accuracy, test_confusion = test_model(model_factory, spectrogram_path, all_folds)
    print(f"Train Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Confusion: {test_confusion}")
