import torch
from torch_geometric.nn import Node2Vec, GraphSAGE, DeepGraphInfomax, SAGEConv
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from .visualization import plot_loss


# ----------- GRAPH SAGE ----------- #


# Define GraphSAGE model
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3, hidden_channels=32, learning_rate=0.01):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = GraphSAGE(in_channels, hidden_channels, num_layers=num_layers) # Example in_channels=number of features, out_channels=embedding size
        self.conv2 = GraphSAGE(hidden_channels, out_channels, num_layers=num_layers)

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)  # Move the model to the device
        self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    @property #getter
    def optimizer(self):
        return self._optimizer

    @property #getter
    def device(self):
        return self._device


def train_gs(gs_model, d, epochs=100):
    losses = []
    for epoch in range(epochs):
        gs_model.train()
        gs_model.optimizer.zero_grad()
        out = gs_model(d.x, d.edge_index)
        loss = F.cross_entropy(out[d.train_mask], d.y[d.train_mask])
        loss.backward()
        gs_model.optimizer.step()
        losses.append(float(loss.item()))

        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')

    plot_loss(losses, 'GraphSAGE')


def get_emb_gs(gs_model, data,*,scaled, pca_scaled):
    gs_model.eval()
    emb = gs_model(data.x, data.edge_index).detach().cpu().numpy()

    # Scale the embeddings to prepare for model
    if scaled:
        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(emb)
        return emb_scaled
    if pca_scaled:
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(emb)
        pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization or LOF
        embeddings_pca = pca.fit_transform(embeddings_scaled)
        return embeddings_pca
    else:
        return emb


# ----------- GRAPH SAGE DGI ----------- #


class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout = 0.9):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def corruption_dgi(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index

def create_dgi_model(in_channels, hidden_channels, out_channels):
    encoder = GraphSAGEEncoder(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels)

    dgi = DeepGraphInfomax(
        hidden_channels=hidden_channels,
        encoder=encoder,
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption_dgi
    )
    return dgi


def train_dgi(model, data, epochs=300, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(data.x, data.edge_index)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    plot_loss(losses, 'GraphSAGE')

    return model

def get_dgi_embeddings(model, data, scaled=False, pca_scaled=False):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        z, _, _ = model(data.x.to(device), data.edge_index.to(device))
        emb = z.cpu().numpy()

    if scaled:
        scaler = StandardScaler()
        emb = scaler.fit_transform(emb)

    if pca_scaled:
        scaler = StandardScaler()
        emb_scaled = scaler.fit_transform(emb)
        pca = PCA(n_components=2)
        emb = pca.fit_transform(emb_scaled)

    return emb


# ----------- NODE 2 VEC ----------- #


# strictly not necessary to create our own n2v class, but for further customization (adding more specific layers etc.)
# and for harmony & nice encapsulation (all n2v functions are here), we do it anyway.

class Node2VecModel(torch.nn.Module):
    def __init__(self, edge_index, embedding_dim, walk_length, context_size, walks_per_node, num_negative_samples, p, q,
                 learning_rate=0.01):
        super(Node2VecModel, self).__init__()
        self.n2v = Node2Vec(edge_index, embedding_dim=32, walk_length=10, context_size=10, walks_per_node=100, num_negative_samples=1,
                            p=2, q=0.5)
        self.embedding_dim = embedding_dim
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)  # Move the model to the device
        self._loader = self.n2v.loader(batch_size=512, shuffle=True, num_workers=0) # FROM IF
        self._optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, batch=None):
        return self.n2v(batch)

    def loss(self, batch):
        return self.n2v.loss(batch)

    def embedding(self, u=None):
        return self.n2v.embedding(u)

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def device(self):
        return self._device

    @property
    def loader(self):
        return self._loader


def train_n2v(n2v_model, epochs=5):

    def train():
        n2v_model.train()
        total_loss = 0
        for pos_rw, neg_rw in n2v_model.loader:
            n2v_model.optimizer.zero_grad()
            loss = n2v_model.n2v.loss(pos_rw.to(n2v_model.device), neg_rw.to(n2v_model.device))  # note in __init__
            # we named our Node2Vec model 'self.n2v' -> n2v_model.N2V!
            loss.backward()
            n2v_model.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(n2v_model.loader)

    losses = []
    for epoch in tqdm(range(epochs), leave=False):
        loss = train()
        losses.append(float(loss),)
        tqdm.write(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    plot_loss(losses,'Node2Vec')


# Get the node embeddings after training
def get_emb_n2v(n2v_model, *, scaled: bool, pca_scaled: bool):  # all arguments after * are required

    n2v_model.eval()
    # or: embeddings = n2v_model.model()  # this calls the forward method, i.e. the actual learned embeddings
    # or: filtered_embeddings = embeddings[valid_node_indices]
    emb = n2v_model.n2v.embedding.weight.detach().cpu().numpy() # .numpy: turn them into np array with structure (num_nodes, emb_dim)

    if scaled:
        scaler = StandardScaler()
        emb_sc = scaler.fit_transform(emb)
        return emb_sc
    if pca_scaled:
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(emb)
        pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization or LOF
        embeddings_pca = pca.fit_transform(embeddings_scaled)
        return embeddings_pca
    else:
        return emb

"""
Node 2 Vec train function before IF incorporation: does not use a Loader (was also not present in __init__)

def train_n2v(n2v_model, d, epochs=200):

    for epoch in range(epochs):
        n2v_model.train()
        n2v_model.optimizer.zero_grad()
        batch_size = 128  # You can adjust the batch size depending on your memory
        batch_nodes = torch.randint(0, d.num_nodes, (batch_size,)).to(device)
        pos_rw, neg_rw = n2v_model.sample(batch_nodes)  # Positive and negative random walks
        loss = n2v_model.loss(pos_rw, neg_rw)
        loss.backward()
        n2v_model.optimizer.step()

        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')



"""