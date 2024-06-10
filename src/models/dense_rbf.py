"""Network model for dense RBF network."""
import torch
from torch import nn

from models.rbf.rbf_network import RbfNetwork
from models.rbf.rbf import poisson_two


# pylint: disable=R0902, R0801
class DenseRBF(nn.Module):
    """Network model for dense RBF network."""
    def __init__(self, input_size: int, output_size: int, num_dense_neurons=512):
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._num_dense_neurons = num_dense_neurons

        self._lin1 = nn.Linear(self._input_size, self._num_dense_neurons)
        self._lin2 = nn.Linear(self._num_dense_neurons, self._num_dense_neurons // 2)
        self._activation = nn.Tanh()
        self._dropout = nn.Dropout(0.5)
        self.rbf_layers = RbfNetwork(
                [
                    self._num_dense_neurons // 2,
                    self._num_dense_neurons // 2,
                    self._num_dense_neurons // 2,
                    1,
                ],
                [
                    self._num_dense_neurons,
                    self._num_dense_neurons // 4,
                    self._num_dense_neurons // 2]
                ,
                poisson_two
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of forward pass of the controller network.
        :rtype: torch.tensor
        """

        x = self._lin1(x)
        x = self._activation(x)
        x = self._dropout(x)

        x = self._lin2(x)
        x = self._activation(x)
        x = self._dropout(x)

        x = torch.squeeze(x, dim=0)
        x = torch.squeeze(x, dim=0)

        x = self.rbf_layers(x)

        return x
