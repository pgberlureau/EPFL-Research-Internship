from torch import randn, randint, zeros, bool, tensor
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx
from networkx import barbell_graph, relabel_nodes, compose
from tasks.pe import LaplacianPE

class BarbellStarDataset(Dataset):
    """Generates a dataset of barbell graphs connected in a star-like structure, with random node features and Laplacian positional encodings.
    Each graph consists of multiple barbell graphs connected at a central node. The task is to predict which feature dimension has the highest value at one of the central nodes of the barbell graphs.
    Args:
        root (str): Root directory where the dataset should be saved.
        n (int): The number of nodes in each cluster of the barbell graph.
        size (int): The number of graphs to generate in the dataset.
        dim_emb (int, optional): The dimension of the random node features. Default is 10.
        dim_pe (int, optional): The dimension of the Laplacian positional encodings. Default is 4.
    """
    def __init__(self, root, n: int, size: int, dim_emb=10, dim_pe=4):
        super().__init__(root)
        self.data = []
        self.cluster_size = n
        self.path_lengths = n - 3
        self.size = size
        self.dim_emb = dim_emb
        self.dim_pe = dim_pe
        laplacian_pe = LaplacianPE(dim_pe)
        n_ends = self.dim_emb // 2
        ends = tensor([(i+1)*(self.path_lengths + self.cluster_size) + self.cluster_size - 1 for i in range(n_ends)])

        for _ in range(self.size):
            G = barbell_graph(self.cluster_size, self.path_lengths)
            for i in range(1, n_ends):
                H = barbell_graph(self.cluster_size, self.path_lengths)
                H = relabel_nodes(H, lambda x: x if x < self.cluster_size else x + i * (self.cluster_size + self.path_lengths))
                G = compose(G, H)

            path = from_networkx(G)
            path.x = randn(path.num_nodes, self.dim_emb)
            path = laplacian_pe(path)

            i = randint(0, self.dim_emb, (1,))
            j = randint(0, n_ends, (1,))

            path.x[ends[j], i] = path.num_nodes
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
