from torch import randn, randint, zeros, bool
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx
from networkx import path_graph
from tasks.pe import LaplacianPE

class PathDataset(Dataset):
    """Generates a dataset of path graphs with random node features and Laplacian positional encodings.
    Each graph consists of a simple path. The task is to predict which feature dimension has the highest value at the last node of the path graph.
    Args:
        root (str): Root directory where the dataset should be saved.
        n (int): The number of nodes in the path graph.
        size (int): The number of graphs to generate in the dataset.
        dim_emb (int, optional): The dimension of the random node features. Default is 10.
        dim_pe (int, optional): The dimension of the Laplacian positional encodings. Default is 4.
    """
    def __init__(self, root, n: int, size: int, dim_emb=10, dim_pe=4):
        super().__init__(root)
        self.data = []
        self.n = n + 1
        self.size = size
        self.dim_emb = dim_emb
        self.dim_pe = dim_pe
        laplacian_pe = LaplacianPE(dim_pe)
        for _ in range(self.size):
            path = path_graph(self.n)
            path = from_networkx(path)
            path.x = randn(path.num_nodes, self.dim_emb)

            path = laplacian_pe(path)

            i = randint(0, self.dim_emb, (1,))
            path.x[-1,i] = path.num_nodes
            path.y = i

            path.mask = zeros(path.num_nodes, dtype=bool)
            path.mask[0] = True

            path = path.coalesce()
            assert path.is_undirected()

            self.data.append(path)

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
    