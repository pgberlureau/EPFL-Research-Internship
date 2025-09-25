from torch import nn, float32
from torch.linalg import matrix_exp
from torch_geometric.utils import to_dense_batch, to_dense_adj, get_laplacian
from models.mu import Mu

class DiffusionLayer(nn.Module):
    """A single layer of diffusion followed by a linear transformation, ReLU activation, and batch normalization.
    Args:
        hidden_channels (int): The number of hidden channels.
    Shape:
        - Input: (x, mask, U) where x is the node feature matrix, mask is a boolean mask holding information about the existence of fake-nodes in the dense representation, and U is the diffusion matrix.
        - Output: (x, mask, U) where x is the transformed node feature matrix.
    """
    def __init__(self, hidden_channels):
        super().__init__()

        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm1d(hidden_channels)

    def forward(self, x):
        x, mask, U = x

        x = U @ x
        x = self.lin(x)
        x = self.act(x)
        x = x.clone()
        x[mask] = self.norm(x[mask])

        return x, mask, U

class HK(nn.Module):
    """Heat Kernel diffusion model.
    Args:
        hidden_channels (int): The number of hidden channels.
        num_layers (int): The number of diffusion layers.
    Shape:
        - Input: data (Data): The input graph data object with attributes x, edge_index, and batch.
        - Output: x (Tensor): The output node feature matrix.
    """
    def __init__(self, hidden_channels, num_layers):
        super().__init__()

        seq = []
        for _ in range(num_layers):
            seq.append(DiffusionLayer(hidden_channels))
        
        self.gnn = nn.Sequential(*seq)

    def _compute_U(self, data, max_num_nodes):

        edge_index, edge_weight = get_laplacian(data.edge_index, num_nodes=data.num_nodes)
        L = to_dense_adj(edge_index=edge_index, batch=data.batch, edge_attr=edge_weight, max_num_nodes=max_num_nodes)
        U = matrix_exp(-L)

        return U

    def forward(self, data):
        max_num_nodes = (data.ptr[1:] - data.ptr[:-1]).max()
        x, mask = to_dense_batch(data.x, batch=data.batch, max_num_nodes=max_num_nodes)

        U = self._compute_U(data, max_num_nodes)

        x, _, _ = self.gnn((x, mask, U))
        
        return x[mask]

class HK_mu(HK):
    """Heat Kernel diffusion model with learned node potentials.
    Args:
        hidden_channels (int): The number of hidden channels.
        dim_pe (int): The dimension of the positional encodings.
        num_layers (int): The number of diffusion layers.
        mu_with_features (bool, optional): If True, use node features as input to the Mu network. Default is False.
    Shape:
        - Input: data (Data): The input graph data object with attributes x, edge_index, pe, and batch.
        - Output: (x, V) where x is the output node feature matrix and V are the learned node potentials.
    """
    def __init__(self, hidden_channels, dim_pe, num_layers, mu_with_features=False):
        super().__init__(hidden_channels, num_layers)
        self.mu_with_features = mu_with_features
        if mu_with_features:
            dim_pe = hidden_channels

        self.mu = Mu(dim_pe)

    def _compute_U(self, data, max_num_nodes):
        if self.mu_with_features:
            pe = data.x
        else:
            pe = data.pe

        V = self.mu(pe, data.edge_index).squeeze(-1)

        s = 0.5 * (V[data.edge_index[0]] + V[data.edge_index[1]])

        edge_index, edge_weight = get_laplacian(data.edge_index, edge_weight=s, num_nodes=data.num_nodes, normalization=None)
        L = to_dense_adj(edge_index=edge_index, batch=data.batch, edge_attr=edge_weight, max_num_nodes=max_num_nodes)

        U = matrix_exp(-L)

        return U, V

    def forward(self, data):
        x = data.x.to(float32)
        max_num_nodes = (data.ptr[1:] - data.ptr[:-1]).max()

        x, mask = to_dense_batch(x, batch=data.batch, max_num_nodes=max_num_nodes)

        U, V = self._compute_U(data, max_num_nodes)

        x, _, _ = self.gnn((x, mask, U))
            
        return x[mask], V
