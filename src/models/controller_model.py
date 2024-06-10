"""Implementation of network for controller model."""
import torch
from torch import nn


# pylint: disable=R0902
class ControllerNetworkModel(nn.Module):
    """Class implementing network for controller model."""
    def __init__(self, input_size: int, output_size: int, num_dense_neurons=512):
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._num_dense_neurons = num_dense_neurons

        self._lin1 = nn.Linear(self._input_size, self._num_dense_neurons)
        self._lin2 = nn.Linear(self._num_dense_neurons // 2, self._num_dense_neurons // 2)
        self._lin3 = nn.Linear(self._num_dense_neurons, self._num_dense_neurons // 2)
        self._lin_out = nn.Linear(self._num_dense_neurons // 2, self._output_size)
        self._activation = nn.Tanh()
        self._dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of forward pass of the controller network.
        :rtype: torch.tensor
        """

        x = self._lin1(x)
        x = self._activation(x)
        x = self._dropout(x)

        x1, x2 = torch.split(x, self._num_dense_neurons, dim=1)

        x1 = self._lin2(x1)
        x1 = self._activation(x1)
        x1 = self._dropout(x1)

        x2 = self._lin2(x2)
        x2 = self._activation(x2)
        x2 = self._dropout(x2)

        x_cat = torch.cat((x1, x2), dim=1)

        x = torch.add(x, x_cat)
        x = self._activation(x)
        x = self._dropout(x)

        x = self._lin3(x)
        x = self._activation(x)
        x = self._dropout(x)

        x = self._lin_out(x)

        return x
