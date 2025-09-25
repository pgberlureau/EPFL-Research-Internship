from torch import nn, no_grad, log, exp, tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential

class Mu(nn.Module):
    """A neural network that learns node potentials for use in diffusion models.
    Args:
        in_channels (int): The number of input channels (features) per node.
    Shape:
        - Input: (x, edge_index) where x is the node feature matrix and edge_index is the edge index tensor.
        - Output: V (Tensor): The learned node potentials.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, 1)
        self.net = Sequential( 'x, edge_index',
            [
            (self.conv, 'x, edge_index -> x'),
            nn.Softplus()
            ]
        )

    def forward(self, x, edge_index):
        return self.net(x, edge_index)