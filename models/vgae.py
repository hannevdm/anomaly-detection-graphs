import torch
from torch_geometric.nn import GCNConv, GAE
import torch.nn.functional as F



class VGAEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VGAEModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.normalize(h, p=2, dim=-1)  # normalize feature-wise
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)

    # Dot product decoder
    def decode(self, z, edge_index):
        # Efficient edge-wise decoding: dot product for each edge (i,j)
        z_i = z[edge_index[0]]
        z_j = z[edge_index[1]]
        return torch.sigmoid((z_i * z_j).sum(dim=1))

    def reparametrize(self, mu, logstd):
        std = torch.exp(torch.clamp(logstd, min=-10, max=10))  # prevent exploding/vanishing
        if self.training:
            return mu + torch.randn_like(std) * std
        else:
            return mu

    def forward(self, x, edge_index):
        mu, logstd = self.encode(x, edge_index)
        z = self.reparametrize(mu, logstd)
        adj_pred = self.decode(z, edge_index)
        return adj_pred, mu, logstd

    def kl_loss(self, mu, logstd):
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - torch.exp(2 * logstd), dim=1))

    # Modified Training loop with Laplacian Loss
    def train_with_laplacian(model, optimizer, data, device):

        # Graph Regularization (Laplacian Regularization)
        def laplacian_loss(z, edge_index):
            row, col = edge_index
            z_row = z[row]
            z_col = z[col]

            return torch.sum((z_row - z_col) ** 2) / edge_index.size(1)

        model.train()
        optimizer.zero_grad()

        # Get mu and logstd from encoder
        mu, logstd = model.encode(data.x, data.edge_index)

        # Sample latent embeddings
        z = model.reparametrize(mu, logstd)
        print("z mean:", z.mean().item(), "std:", z.std().item())
        print("z min/max:", z.min().item(), z.max().item())

        # Decode edges from sampled z
        adj_pred = model.decode(z, data.edge_index)  # outputs in [0, 1]

        print("adj_pred range:", adj_pred.min().item(), adj_pred.max().item())

        bce_loss = F.binary_cross_entropy(
            adj_pred,
            torch.ones_like(adj_pred).to(device)
        )
        num_nodes = data.x.size(0)
        kl_loss = 1 / num_nodes * model.kl_loss(mu, logstd)
        #kl_loss = model.kl_loss(mu, logstd)
        laplacian_reg = laplacian_loss(z, data.edge_index)
        total_loss = bce_loss + 0.1 * kl_loss + 0.05 * laplacian_reg
        total_loss.backward()
        optimizer.step()
        return float(total_loss)

    # Training loop with KL divergence from Pytorch Geometric GAE
    def train_with_gae_kl(model, optimizer, data):
        model.train()
        optimizer.zero_grad()
        adj_pred, mu, logstd = model(data.x, data.edge_index)
        loss = GAE.loss(adj_pred, data.edge_index)
        loss.backward()
        optimizer.step()
        return float(loss)
