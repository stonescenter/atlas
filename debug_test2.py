import math
import torch
from torch.utils.data import DataLoader
from torch import Tensor
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from model.time_encoder import TimeEncoder 
from utils.data_processing import Graph


class TemporalTransitionModel(nn.Module):

    def __init__(
        self,
        num_nodes,
        embedding_dim=64,
        time_dim=32,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_nodes,
            embedding_dim
        )

        self.time_encoder = TimeEncoder(
            time_dim
        )

        temporal_dim = 2 * time_dim

        input_dim = (
            1
            + temporal_dim
            + 2 * embedding_dim
        )

        self.proj = nn.Linear(
            input_dim,
            embedding_dim
        )

        self.attn = nn.Linear(
            embedding_dim,
            1
        )

    # --------------------------------------------------------
    # Structural normalization
    # --------------------------------------------------------

    def structural_score(
        self,
        deg_u,
        deg_v,
    ):

        return (
            1.0 /
            torch.sqrt(
                deg_u * deg_v + 1e-8
            )
        ).unsqueeze(-1)

    # --------------------------------------------------------
    # Compute transition logits
    # --------------------------------------------------------

    def forward(
        self,
        u,
        neighbors,
        deg_u,
        deg_neighbors,
        delta_t,
    ):

        """
        u:
            [K]

        neighbors:
            [K]

        delta_t:
            [K]
        """

        h_u = self.embedding(u)
        h_v = self.embedding(neighbors)

        s_uv = self.structural_score(
            deg_u,
            deg_neighbors,
        )

        phi_t = self.time_encoder(
            delta_t
        )

        print("s_uv:", s_uv.shape)
        print("phi_t:", phi_t.shape)
        print("h_u:", h_u.shape)
        print("h_v:", h_v.shape)

        x = torch.cat(
            [
                s_uv,
                phi_t,
                h_u,
                h_v,
            ],
            dim=-1
        )

        z = F.leaky_relu(
            self.proj(x),
            negative_slope=0.2
        )

        logits = self.attn(z).squeeze(-1)

        return logits

device = 'cpu'

data = {
  "source": [0, 1, 2, 2, 3, 1, 0, 0],
  "target": [1, 2, 3, 4, 4, 4, 3, 2],
  "timestamps":[1, 2, 3, 7, 5, 6, 8, 4]
  }

is_forward = True
df = pd.DataFrame(data)
df = df.sort_values(by='timestamps', ascending=is_forward)

sources = df.source.values
destinations = df.target.values
timestamps = df.timestamps.values

print("sources: ", sources)
print("target: ", destinations)
print('timestamps:', timestamps)

graph = Graph(
    sources=sources,
    destinations=destinations,
    timestamps=timestamps,
    edge_idxs=list(range(sources.shape[0])),
    labels=[1] * sources.shape[0]
)

num_nodes = len(graph.get_nodes())
train_end = math.ceil(int(0.80 * num_nodes))
val_end = int(0.85 * num_nodes)

train_data = graph.sources[:train_end]
val_data = graph.sources[train_end:]

print(f"Nodes {graph.get_nodes()}, len {num_nodes}")


loader = DataLoader(
    graph,
    batch_size=1,
    shuffle=False # the model is trained in time dependency
)

model = TemporalTransitionModel(
    num_nodes=5,
    embedding_dim=64,
    time_dim=32,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.CrossEntropyLoss()

# ============================================================
# TRAINING
# ============================================================

def train(epochs=10):

    seed = True
    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for batch in loader:

            # u-v
            #if seed:
            src, dst, t = batch['src'][0], batch['dst'][0], batch['ts'][0]
            src, dst, t = src.item(), dst.item(), t.item()
            current_node = dst
            #seed = False

            #neighbors_u, _ = graph.get_neighbors_array(current_node, include_edge_weight=True)
            neighbors, times = graph.get_neighbors_array(current_node, include_edge_weight=True)
            # filter times with history 
            #if is_forward:
            #    mask = times > t
            #else:
            #    mask = times < t

            mask = times >= t # future neighbors
            neighbors = neighbors[mask]
            times = times[mask]
            
            # retorna o grau de uma lista de nós
            deg_neighbors  = graph.get_degree_neighbors(neighbors)
            neighbors_nodes = torch.tensor(neighbors, dtype=torch.long)

            if len(neighbors) == 0:
                continue

            deg_current = len(neighbors)
            
            current_node = torch.full((deg_current,), current_node,  dtype=torch.long)
            deg_current = torch.full((deg_current,), deg_current, dtype=torch.float)
            deg_neighbors = torch.tensor(deg_neighbors, dtype=torch.float)

            #delta_t = torch.tensor(t - times, dtype=torch.float)
            delta_t = torch.tensor(times - t, dtype=torch.float)
            print(f"src: {src}, dst: {dst}, t: {t} neighbors: {neighbors}, neighbors_v: {neighbors}, times: {times}")
            print("delta_t :", delta_t)

            print("neighbors_u:", len(neighbors))
            print("times:", len(times))


            # model (u, v, deg_u, deg_v)  
            scores = model(
                current_node, 
                neighbors_nodes,
                deg_current, 
                deg_neighbors,
                delta_t,    # derived from neighbors_v
            )

            # probs are target , it contain probabilities
            probs = torch.softmax(scores, dim=0)
  
            print("scores : ", scores.unsqueeze(0))
            print(f"prob: {probs}, argmax {probs.argmax()}, sum: {probs.sum(axis=0)} ")

            #target_index = np.where(neighbors == v)[0][0]
            idx = np.argmin(times)
            target_node = neighbors[idx]
            target_index = torch.tensor([idx], dtype=torch.long, device=device)
            print(f"target_index: {idx}, target: {target_node}")
            
            # ----------------------------------------------------
            # CrossEntropy over neighbors
            # ----------------------------------------------------
            # inputs = scores
            inputs = scores.unsqueeze(0)
            # internally loss function compute softmax(scores)
            loss = criterion(inputs, target_index)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch:03d} "
            f"| Loss {total_loss:.4f}"
        )
         
train(50)


def test():
    logits = model(
        u_tensor,
        neighbors_tensor,
        deg_u,
        deg_neighbors_tensor,
        delta_t,
    )

    probs = torch.softmax(
        logits,
        dim=0
    )

    print("logits : ", logits)
    print(f"prob: {probs}, argmax {probs.argmax()}")

    predicted_neighbor = neighbors_u[
        probs.argmax()
    ]
    print("predicted: ", predicted_neighbor)



train()
