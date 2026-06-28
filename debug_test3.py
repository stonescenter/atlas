import numpy as np
import pandas as pd
import random

import torch
from torch.utils.data import DataLoader

from utils.graph import GraphStorage
from utils.sampler import TemporalNeighborSampler
from utils.data_processing import EdgeDataset

import math
from torch import Tensor
import torch_geometric.transforms as T
import torch.nn as nn
import torch.nn.functional as F


from model.time_encoder import TimeEncoder 
from utils.data_processing import get_data, Graph



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

        #print("s_uv:", s_uv.shape)
        #print("phi_t:", phi_t.shape)
        #print("h_u:", h_u.shape)
        #print("h_v:", h_v.shape)

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

def teacher_build_target_distribution(
    deg_node,
    deg_candidates,
    delta_t,
    lambda_t=0.2,
):
    """
    Structural + temporal target policy
    """

    s = (1.0/np.sqrt(deg_node * deg_candidates + 1e-8))

    temporal = np.exp(-lambda_t * delta_t)

    q = s * temporal

    q = q / q.sum()

    return q

PATH_DATASET = '/exp-local/steve/datasets/temporal/ml_preprocess/'
dataset_name = 'wikipedia'

graph_df = pd.read_csv('{}/ml_{}.csv'.format(PATH_DATASET, dataset_name))
sources = graph_df.u.values
destinations = graph_df.i.values
edge_idxs = graph_df.idx.values
labels = graph_df.label.values
timestamps = graph_df.ts.values


data = {
  "source": [0, 1, 2, 2, 3, 1, 0, 0],
  "target": [1, 2, 3, 4, 4, 4, 3, 2],
  "timestamps": [1, 2, 3, 7, 5, 6, 8, 4]
}

is_forward = True
df = pd.DataFrame(data)
df = df.sort_values(by='timestamps', ascending=is_forward)

sources = df.source.values
destinations = df.target.values
timestamps = df.timestamps.values

train_data = EdgeDataset(sources, destinations, timestamps, edge_idxs, labels)


'''
random.seed(2020)
val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

train_mask =  timestamps <= test_time
test_mask = timestamps > test_time
val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) 

train_data = EdgeDataset(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

val_data = EdgeDataset(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

test_data = EdgeDataset(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

'''

graph = GraphStorage(sources, destinations, timestamps)
sampler = TemporalNeighborSampler(graph, num_neighbors=20)

loader = DataLoader(train_data,batch_size=1, shuffle=False,)

device = 'cpu'

# create models
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
criterion = nn.KLDivLoss()


'''
Este modelo aprende uma politica de destilação 
 teacher policy : s(u,v) * e^(lambda *delta_t)

'''
def train(epochs=10):

    seed = True
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in loader:

            src = batch["src"][0]
            dst = batch["dst"][0]
            t = batch["ts"][0]
            src, dst, t = src.item(), dst.item(), t.item()
            current_node = dst
            
            neighbors, times = sampler.sample(node_id=dst, current_time=t, is_forward=True)

     
            #print(f"src: {}, neighbors: {neighbors}, times:{times}")
            

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
            print(f"src: {src}, dst: {dst}, current time: {t}, neighbors: {neighbors}, times:{times}, delta: {delta_t}")
            
            print("neighbors len:", len(neighbors))
            print("times len:", len(times))

            # model (u, v, deg_u, deg_v)  
            scores = model(
                current_node, 
                neighbors_nodes,
                deg_current, 
                deg_neighbors,
                delta_t,    # derived from neighbors_v
            )

            # probs are target , it contain probabilities
            pred_probs = F.log_softmax(scores, dim=-1)
            #pred_probs = torch.softmax(scores, dim=0)

            print("scores : ", scores.unsqueeze(0))
            print(f"prob: {pred_probs}, argmax {pred_probs.argmax()}, sum: {pred_probs.sum(axis=0)} ")

            target_dist = teacher_build_target_distribution(deg_current, deg_neighbors, delta_t.numpy(), lambda_t=0.2)
            target_dist = torch.tensor(target_dist, dtype=torch.float, device=device)

            loss = F.kl_div(pred_probs, target_dist, reduction='batchmean')        
            #loss = -(target_dist *torch.log(pred_probs + 1e-12)).sum()
            #loss = criterion(pred_probs, target_dist)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(
            f"Epoch {epoch:03d} "
            f"| Loss {total_loss:.4f}"
        )

train(10)

