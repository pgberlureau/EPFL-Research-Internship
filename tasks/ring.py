from torch import randn, randint, zeros, bool
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx
from networkx import cycle_graph
from tasks.pe import LaplacianPE

class RingDataset(Dataset):
    """Generates a dataset of ring (cycle) graphs with random node features and Laplacian positional encodings.
    Each graph consists of a simple cycle. The task is to predict which feature dimension has the highest value at one of the nodes in the cycle graph.
    Args:
        root (str): Root directory where the dataset should be saved.
        n (int): The number of nodes in the cycle graph.
        size (int): The number of graphs to generate in the dataset.
        dim_emb (int, optional): The dimension of the random node features. Default is 10.
        dim_pe (int, optional): The dimension of the Laplacian positional encodings. Default is 4.
    """
    def __init__(self, root, n: int, size: int, dim_emb=10, dim_pe=4):
        super().__init__(root)
        self.data = []
        self.path_lengths = n
        self.size = size
        self.dim_emb = dim_emb
        self.dim_pe = dim_pe
        laplacian_pe = LaplacianPE(self.dim_pe)
        for _ in range(self.size):
            path = cycle_graph(2*self.path_lengths)
            path = from_networkx(path)
            path.x = zeros(path.num_nodes, self.dim_emb)
            path.x[:self.path_lengths] = (1/self.path_lengths)*randn(self.path_lengths, self.dim_emb)
            path.x[self.path_lengths:] = randn(self.path_lengths, self.dim_emb)
            
            i = randint(0, self.dim_emb, (1,))
            path.x[self.path_lengths,i] = path.num_nodes
            path.y = i

            path = laplacian_pe(path)

            path.mask = zeros(path.num_nodes, dtype=bool)
            path.mask[0] = True

            path = path.coalesce()
            assert path.is_undirected()

            self.data.append(path)

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]