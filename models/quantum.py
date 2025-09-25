from torch import cat, nn, Tensor, complex, tensor, complex64, relu, stack
from torch_geometric.utils import to_dense_batch, to_dense_adj, get_laplacian
from torch.linalg import matrix_exp
from models.mu import Mu

class ModReLU(nn.Module):
    """Modified ReLU activation function.
    Shape:
        - Input: x (Tensor): The input tensor.
        - Output: (Tensor): The output tensor after applying the ModReLU activation.
    """
    def __init__(self):
        super(ModReLU, self).__init__()

        self.b = nn.Parameter(tensor(0.))

    def forward(self, x: Tensor) -> Tensor:
        shift = x.abs() + self.b
        c = relu(shift) / (x.abs() + 1e-8)
        return c * x

class CLinear(nn.Module):
    """Complex-valued linear layer followed by ModReLU activation.
    Args:
        channels (int): The number of input and output channels.
    Shape:
        - Input: x (Tensor): The input complex-valued tensor.
        - Output: (Tensor): The output complex-valued tensor after linear transformation and ModReLU activation.
    """
    def __init__(self, channels: int):
        super(CLinear, self).__init__()
        self.channels = channels
        self.real_lin = nn.Linear(channels, channels)
        self.imag_lin = nn.Linear(channels, channels)
        
        self.act = ModReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = complex(real=self.real_lin(x.real) - self.imag_lin(x.imag),
                             imag=self.real_lin(x.imag) + self.imag_lin(x.real))
        
        return self.act(x)
    
class CNorm(nn.Module):
    """Complex-valued batch normalization.
    Args:
        channels (int): The number of channels.
    Shape:
        - Input: x (Tensor): The input complex-valued tensor.
        - Output: (Tensor): The output complex-valued tensor after batch normalization.
    """
    def __init__(self, channels: int):
        super(CNorm, self).__init__()
        self.channels = channels
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: Tensor) -> Tensor:
        z = stack([x.real, x.imag], dim=-1)
        z = self.norm(z)
        x.real = z[..., 0]
        x.imag = z[..., 1]
        return x

class CTQWLayer(nn.Module): 
    """A single layer of continuous-time quantum walk (CTQW) diffusion followed by a complex-valued linear transformation, ModReLU activation, and complex-valued batch normalization.
    Args:
        hidden_channels (int): The number of hidden channels.
    Shape:
        - Input: (x, mask, U) where x is the complex-valued node feature matrix, mask is a boolean mask holding information about the existence of fake-nodes in the dense representation, and U is the complex-valued diffusion matrix.
        - Output: (x, mask, U) where x is the transformed complex-valued node feature matrix.
    """ 
    def __init__(self, hidden_channels: int):
        super().__init__()

        self.lin = CLinear(hidden_channels)
        self.act = ModReLU()
        self.norm = CNorm(hidden_channels)

    def forward(self, x):
        x, mask, U = x

        x = U @ x
        x = self.lin(x)
        x = self.act(x)
        x[mask] = self.norm(x[mask])

        return (x, mask, U)

class CTQW(nn.Module):
    """Continuous-Time Quantum Walk (CTQW) diffusion model.
    Args:
        hidden_channels (int): The number of hidden channels.
        num_layers (int): The number of diffusion layers.
    Shape:
        - Input: data (Data): The input graph data object with attributes x, edge_index, and batch.
        - Output: x (Tensor): The output node feature matrix.
    """
    def __init__(self, hidden_channels: int, num_layers: int):
        super().__init__()

        seq = []
        for _ in range(num_layers):
            seq.append(CTQWLayer(hidden_channels))
        
        self.gnn = nn.Sequential(*seq)

        self.fc = nn.Linear(2*hidden_channels, hidden_channels)

    def _compute_U(self, data, max_num_nodes):

        edge_index, edge_weight = get_laplacian(data.edge_index, num_nodes=data.num_nodes, normalization=None)
        L = to_dense_adj(edge_index=edge_index, batch=data.batch, edge_attr=edge_weight, max_num_nodes=max_num_nodes).to(complex64)
        U = matrix_exp(-1j*L)

        return U
    
    def forward(self, data):
        x = data.x.to(complex64)
        max_num_nodes = (data.ptr[1:] - data.ptr[:-1]).max().item()

        x, mask = to_dense_batch(x, batch=data.batch, batch_size=data.batch_size, max_num_nodes=max_num_nodes)

        U = self._compute_U(data, max_num_nodes)

        x, _, _ = self.gnn((x, mask, U))

        x = cat([x.real, x.imag], dim=-1)
        x = self.fc(x)

        return x[mask]

class CTQW_mu(CTQW):
    """Continuous-Time Quantum Walk (CTQW) diffusion model with learned node potentials.
    Args:
        hidden_channels (int): The number of hidden channels.
        dim_pe (int): The dimension of the positional encodings.
        num_layers (int): The number of diffusion layers.
        mu_with_features (bool, optional): If True, use node features as input to the Mu network. Default is False.
    Shape:
        - Input: data (Data): The input graph data object with attributes x, edge_index, pe, and batch.
        - Output: (x, V) where x is the output node feature matrix and V are the learned node potentials.
    """
    def __init__(self, hidden_channels: int, dim_pe: int, num_layers: int, mu_with_features=False):
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
        L = to_dense_adj(edge_index=edge_index, batch=data.batch, edge_attr=edge_weight, max_num_nodes=max_num_nodes).to(complex64)

        U = matrix_exp(-1j*L)

        return U, V

    def forward(self, data):
        x = data.x.to(complex64)
        max_num_nodes = (data.ptr[1:] - data.ptr[:-1]).max()

        x, mask = to_dense_batch(x, batch=data.batch, batch_size=data.batch_size, max_num_nodes=max_num_nodes)

        U, V = self._compute_U(data, max_num_nodes)

        x, _, _ = self.gnn((x, mask, U))

        x = cat([x.real, x.imag], dim=-1)
        x = self.fc(x)

        return x[mask], V
