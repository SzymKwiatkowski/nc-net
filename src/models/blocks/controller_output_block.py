"""Controller output block for controller network ensemble."""
import torch
from torch import nn


class ControllerOutputBlock(nn.Module):
    """Initialize output block for controller network ensemble."""
    def __init__(self, input_size: int, output_size: int, num_dense_neurons=512):
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._num_dense_neurons = num_dense_neurons

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

        return x
