import torch
from torch import nn


class ControllerModel(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 act,
                 filter_sz=5,
                 num_filters=24,
                 num_dense_neurons=512,
                 ):
        super().__init__()

        self._in_channels = in_channels
        self._num_classes = num_classes

        self._activation = nn.ELU()
        self._num_filters = num_filters
        self._filter_sz = filter_sz
        self._num_dense_neurons = num_dense_neurons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.Conv2d(1, self._num_filters,  (self._filter_sz, self._filter_sz),
                      padding='same')(x)
        x = self._activation(x)
        x = nn.MaxPool2d(2, 2)(x)

        x = nn.Conv2d(2 * self._num_filters, 1, (self._filter_sz, self._filter_sz),
                      padding='same')(x)
        x = self._activation(x)
        x = nn.MaxPool2d(2, 2)(x)

        x = nn.Conv2d(4 * self._num_filters, 1, (self._filter_sz, self._filter_sz),
                      padding='same')(x)
        x = self._activation(x)
        x = nn.MaxPool2d(2, 2)(x)

        x = nn.Dropout(.5)(x)
        x = nn.Flatten()(x)

        x = nn.Linear(self._num_dense_neurons)(x)
        x = self._activation(x)
        x = nn.Dropout(.5)(x)
        x = nn.Linear(self._num_dense_neurons // 2)(x)
        x = self._activation(x)
        x = nn.Linear(self._num_classes)(x)

        return x