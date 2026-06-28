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

class TemporalAttentionLayer(nn.Module):

    def __init__(
        self,
        embedding_dim=64,
        time_dim=32,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.time_encoder = TimeEncoder(dimension=time_dim)

        temporal_dim = 2 * time_dim

        input_dim = (
            1                    # structural score
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

    def structural_score(self, deg_u, deg_v):
        score = (
            1.0 /
            torch.sqrt(
                deg_u * deg_v + 1e-8
            )
        )

        return score.unsqueeze(-1)

    # --------------------------------------------------------
    # Attention score
    # --------------------------------------------------------

    def forward(
        self,
        h_u,
        h_v,
        deg_u,
        deg_v,
        delta_t,
    ):

        s_uv = self.structural_score(deg_u, deg_v)

        phi_t = self.time_encoder(delta_t)
        
        x = torch.cat([s_uv, phi_t, h_u, h_v], dim=-1)
        # --------------------------------------------
        # Attention network
        # --------------------------------------------

        z = F.leaky_relu(self.proj(x), negative_slope=0.2)
            
        e_uv = self.attn(z).squeeze(-1)

        return e_uv


# ============================================================
# Temporal Link Predictor
# ============================================================

class TemporalLinkPredictor(nn.Module):

    def __init__(
        self,
        num_nodes,
        embedding_dim=64,
        time_dim=32,
    ):
        super().__init__()

        self.node_embedding = nn.Embedding(
            num_nodes,
            embedding_dim
        )

        nn.init.xavier_uniform_(
            self.node_embedding.weight
        )

        self.temporal_attention = TemporalAttentionLayer(
            embedding_dim=embedding_dim,
            time_dim=time_dim,
        )

    def forward(
        self,
        u,
        v,
        deg_u,
        deg_v,
        delta_t,
    ):

        h_u = self.node_embedding(u)
        h_v = self.node_embedding(v)
        #print(f"h_u.shape: {h_u.shape}, h_v.shape: {h_v.shape}")
        

        # raw attention scores logits
        e_uv = self.temporal_attention(
            deg_u=deg_u,
            deg_v=deg_v,
            delta_t=delta_t,
            h_u=h_u,
            h_v=h_v         
        )

        # probabilities distributions between 0-1
        print("e_uv=", e_uv)
        p_uv = F.softmax(e_uv, dim=0)

        return p_uv


device = 'cpu'

data = {
  "source": [0, 1, 2, 2, 3, 1, 0, 0],
  "target": [1, 2, 3, 4, 4, 4, 3, 2],
  "timestamps":[1, 2, 3, 7, 5, 6, 8, 4]
  }

df = pd.DataFrame(data)
df = df.sort_values(by='timestamps', ascending=True)

sources = df.source.values
destinations = df.target.values
timestamps = df.timestamps.values

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


model = TemporalLinkPredictor(
    num_nodes=num_nodes,
    embedding_dim=64,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss() # -log(softmax)

num_epochs = 10
is_forward = False
model.train()

#print(data.mapping_degrees)
#print(data.inv_sqrt_degree)

def train_one():

    for epoch in range(num_epochs):
        total_loss = 0.0
        for d in graph:
            u, v, t =  d 
    
            # Convert to list to modify
            neighbors_u, times = graph.get_neighbors_array(u, include_edge_weight=True)
            print(f"t={t}, u={u}, v={v}, times={times}")
    
            #if is_forward:
            #    mask = times > t
            #else:
            #    mask = times < t
            #neighbors_u = neighbors_u[mask]
            #times = times[mask]

            deg_v_neighbors  = graph.get_degree_neighbors(neighbors_u)
            candidate_v = torch.tensor(neighbors_u, dtype=torch.long)
            
            deg_u = len(neighbors_u)
            deg_v = graph.get_degree(v) 
            


            u_tensor = torch.full((len(candidate_v ),), u,  dtype=torch.long)
            deg_u_tensor = torch.full((len(candidate_v),), deg_u, dtype=torch.float)
            deg_v_tensor = torch.tensor(deg_v_neighbors, dtype=torch.float)

            delta_t = torch.tensor(t - times, dtype=torch.float)
            print(f"t={t}, u={u}, v={v}, times={times}, delta_t={delta_t}")
            print(f"neighbors_u={neighbors_u}, deg({u})={deg_u}, deg({v})={deg_v} ")
            #print(f"debug tensor u: {u_tensor}, {deg_u_tensor}, {deg_v_tensor} ")
            #print(f"debug u_tensor.shape: {u_tensor.shape}, deg_u_tensor: {deg_u_tensor.shape} ")
            #print(f"debug deg_v_tensor.shape: {deg_v_tensor.shape}, delta_t: {delta_t.shape} ")

            # ---------------------------------------------
            # Transition probabilities
            # ---------------------------------------------

            probs_uv = model(
                u=u_tensor,
                v=candidate_v,
                deg_u=deg_u_tensor,
                deg_v=deg_v_tensor,
                delta_t=delta_t,
            )

            print(f"probs_uv={probs_uv}")
            # ---------------------------------------------
            # True neighbor index
            # ---------------------------------------------
            #idx = (neighbors_u == v).nonzero(as_tuple=True)[0]

            target_idx = (candidate_v == v)
            #print(f"target={target_idx}")
                    
            target_idx = (candidate_v == v).nonzero(as_tuple=True)[0]
            target_idx = target_idx.item()
            print(f"idx={target_idx}, probs_uv[{target_idx}]={probs_uv[target_idx]}")

            
            loss = -torch.log(probs_uv[target_idx])
            #loss = criterion(scores, target)
            print("loss=", loss.item())
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch} "
            f"| Loss {total_loss:.4f}"
        )
