from torch import nn, sparse_coo_tensor
from torch.linalg import eigh
from torch_geometric.utils import get_laplacian

class LaplacianPE(nn.Module):
    """Computes the Laplacian positional encodings for a graph.
    Args:
        dim_pe (int): The dimension of the positional encodings to compute.
    Shape:
        - Input: data (Data): The input graph data object with attribute edge_index.
        - Output: data (Data): The output graph data object with added attribute pe containing the positional encodings.
    """
    def __init__(self, dim_pe):
        super().__init__()
        self.dim_pe = dim_pe

    def forward(self, data):
        num_nodes = data.num_nodes
        L = get_laplacian(data.edge_index, num_nodes=num_nodes, normalization='sym')
        L = sparse_coo_tensor(indices=L[0], values=L[1], size=(num_nodes, num_nodes))
        L = L.to_dense()

        eig, eigvec = eigh(L)
        
        data.pe = eigvec[:, :self.dim_pe]
        return data