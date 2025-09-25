from torch import cat, nn
from torch_geometric.nn import Linear, MLP, GCN, GAT
from models.diffusion import HK, HK_mu
from models.quantum import CTQW, CTQW_mu

class Custom_GNN(nn.Module):
    """A customizable Graph Neural Network (GNN) model that can be configured to use different architectures such as GCN, GAT, HK, CTQW, HK with learned node potentials, and CTQW with learned node potentials.
    Args:
        in_channels (int): The number of input channels (features) per node.
        hidden_channels (int): The number of hidden channels.
        out_channels (int): The number of output channels (features) per node.
        num_layers (int): The number of layers in the GNN.
        model (str): The type of GNN model to use. Must be one of 'GCN', 'GAT', 'HK', 'CTQW', 'HK_mu', or 'CTQW_mu'.
        dim_pe (int, optional): The dimension of the positional encodings. Default is 4.
        mu_with_features (bool, optional): If True, use node features as input to the Mu network (only applicable for 'HK_mu' and 'CTQW_mu'). Default is False.
    Shape:
        - Input: data (Data): The input graph data object with attributes x, edge_index, pe (optional), and batch.
        - Output: x (Tensor): The output node feature matrix.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, model='GCN', dim_pe=4, mu_with_features=False):
        super(Custom_GNN, self).__init__()

        assert model in ['GCN', 'GAT', 'HK', 'CTQW', 'HK_mu', 'CTQW_mu'], "Model must be one of 'GCN', 'GAT', 'HK', 'CTQW', 'HK_mu', or 'CTQW_mu'."

        self.model = model

        self.encoder = Linear(in_channels=in_channels+dim_pe,
                            out_channels=hidden_channels)

        if model in ['GCN', 'GAT']:
            self.gnn = eval(model)(in_channels=hidden_channels,
                           hidden_channels=hidden_channels,
                           out_channels=hidden_channels,
                           num_layers=num_layers,
                           norm="batch_norm")

        elif model in ['HK', 'CTQW']:
            self.gnn = eval(model)(hidden_channels=hidden_channels,
                                   num_layers=num_layers)

        else:
            self.gnn = eval(model)(hidden_channels=hidden_channels,
                                   dim_pe=dim_pe,
                                   num_layers=num_layers,
                                   mu_with_features=mu_with_features)

        self.decoder = Linear(in_channels=hidden_channels,
                           out_channels=out_channels)

    def forward(self, data):
        concat = cat([data.x, data.pe], dim=-1) if hasattr(data, 'pe') else data.x
        data.x = concat

        x = self.encoder(data.x)
        data.x = x

        if self.model in ['GCN', 'GAT']:
            x = self.gnn(data.x, data.edge_index)
        else:
            x = self.gnn(data)

        if isinstance(x, tuple):
            x, V = x

        x = self.decoder(x)
        
        return x
