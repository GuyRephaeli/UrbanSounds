from typing import Callable

from torch.nn import Module, Conv2d, Linear
from torchvision.models import resnet18


def pretrained_resnet18(classes: int) -> Callable[[], Module]:
    def create_nn() -> Module:
        nn = resnet18(pretrained=True)
        nn.conv1 = Conv2d(in_channels=1,
                          out_channels=nn.conv1.out_channels,
                          kernel_size=nn.conv1.kernel_size[0],
                          stride=nn.conv1.stride[0],
                          padding=nn.conv1.padding[0])
        nn.fc = Linear(nn.fc.in_features, classes)
        return nn
    return create_nn


def frozen_resnet18(classes: int) -> Callable[[], Module]:
    def create_nn() -> Module:
        nn = resnet18(pretrained=True)
        for child in nn.children():
            for param in child.parameters():
                param.requires_grad = False

        nn.conv1 = Conv2d(in_channels=1,
                          out_channels=nn.conv1.out_channels,
                          kernel_size=nn.conv1.kernel_size[0],
                          stride=nn.conv1.stride[0],
                          padding=nn.conv1.padding[0])
        nn.fc = Linear(nn.fc.in_features, classes)
        return nn
    return create_nn
