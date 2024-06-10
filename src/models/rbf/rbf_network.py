"""RBF network implementation."""
import torch
from torch import nn
from models.rbf import rbf


class RbfNetwork(nn.Module):
    """RbfNetwork pytorch implementation."""
    def __init__(self, layer_widths, layer_centres, basis_func):
        super(RbfNetwork, self).__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(len(layer_widths) - 1):
            self.rbf_layers.append(rbf.RBF(layer_widths[i], layer_centres[i], basis_func))
            self.linear_layers.append(nn.Linear(layer_centres[i], layer_widths[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @param x: torch.Tensor input
        @return: torch.Tensor output
        """
        out = x
        # pylint: disable=C0200
        for i in range(len(self.rbf_layers)):
            out = self.rbf_layers[i](out)
            out = self.linear_layers[i](out)
        return out
