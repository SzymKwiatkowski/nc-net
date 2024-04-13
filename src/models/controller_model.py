import torch
from torch import nn


class ControllerNetworkModel(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 filter_sz=5,
                 num_filters=24,
                 num_dense_neurons=512,
                 ):
        super().__init__()

        self._in_channels = in_channels
        self._num_classes = num_classes

        self._num_filters = num_filters
        self._filter_sz = filter_sz
        self._num_dense_neurons = num_dense_neurons

        self._activation = nn.ELU()
        self._max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._conv1 = nn.Conv2d(1, self._num_filters,
                                (self._filter_sz, self._filter_sz), padding='same')
        self._conv2 = nn.Conv2d(1, 2 * self._num_filters,
                                (self._filter_sz, self._filter_sz), padding='same')

        self._conv3 = nn.Conv2d(1, 4 * self._num_filters,
                                (self._filter_sz, self._filter_sz), padding='same')

        self._dropout = nn.Dropout2d(.5)
        self._lin1 = nn.Linear(1, self._num_dense_neurons)
        self._lin2 = nn.Linear(self._num_dense_neurons, self._num_dense_neurons // 2)
        self._linOut = nn.Linear(self._num_dense_neurons // 2, self._num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv1(x)
        x = self._activation(x)
        x = self._max_pool(x)

        x = self._conv2(x)
        x = self._activation(x)
        x = self._max_pool(x)

        x = self._conv3(x)
        x = self._activation(x)
        x = self._max_pool(x)

        x = self._dropout(x)
        x = nn.Flatten()(x)

        x = self.lin1(x)
        x = self._activation(x)
        x = self._dropout(x)
        x = self._lin2(x)
        x = self._activation(x)
        x = self._linOut(x)

        return x
