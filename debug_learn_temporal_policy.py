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


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device available ", device)

class TemporalTransitionModel(nn.Module):

    def __init__(
        self,
        num_nodes,
        embedding_dim=64,
        time_dim=32,
        pad_node=None,
        debug=False
    ):

        super().__init__()

        #self.embedding = nn.Embedding(num_nodes, embedding_dim)

        self.embedding = nn.Embedding(
            num_nodes + 1,
            embedding_dim,
            padding_idx=pad_node,
        )

        self.time_encoder = TimeEncoder(time_dim)

        temporal_dim = 2 * time_dim

        #input_dim = ( 1 + temporal_dim + 1 + embedding_dim)
        input_dim = (
            1 +             # structural
            temporal_dim +  # temporal
            1 +             # semantic
            2*embedding_dim # h_prev , h_current
        )

        self.proj = nn.Linear(
            input_dim,
            embedding_dim
        )

        self.attn = nn.Linear(
            embedding_dim,
            1
        )

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.lambda_t = 0.2
        self.debug = debug 

    # --------------------------------------------------------
    # Structural normalization
    # --------------------------------------------------------

    def s(self, deg_u, deg_v):
        # remueve a dimensao sem de tamnaho 1 sem cambiar os dados atuais
        deg_u = deg_u.unsqueeze(1)

        return (1.0 / torch.sqrt(deg_u * deg_v + 1e-8)).unsqueeze(-1)

    # --------------------------------------------------------
    # Compute transition logits
    # --------------------------------------------------------

    def forward(
        self,
        previous_nodes,
        current_nodes,
        neighbor_nodes,
        deg_current,
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
        B, K = neighbor_nodes.shape

        h_prev = self.embedding(previous_nodes)   # [B,D]
        h_curr = self.embedding(current_nodes)    # [B,D]

        h_prev = h_prev.unsqueeze(1).expand(-1, K, -1)
        h_curr = h_curr.unsqueeze(1).expand(-1, K, -1)

        h_next = self.embedding(neighbor_nodes)   # [B,K,D]

        #h_prev = self.embedding(previous_nodes)  # src
        #h_current = self.embedding(current_nodes) # dst
        #h_next = self.embedding(neighbor_nodes) # candidates


        s = self.s(deg_current, deg_neighbors)
        #structural = self.alpha * torch.log(s+1e-8)
        structural = torch.log(s+1e-8)
    

        temporal = self.time_encoder(delta_t)
        #temporal = self.beta * (-self.lambda_t * delta_t)
        semantic = (h_curr * h_next).sum(dim=-1, keepdim=True)

        if self.debug:
            print("h_prev:", h_prev.shape)
            print("h_current:", h_curr.shape)
            print("h_next:", h_next.shape)
            print("s_uv:", s.shape)
            print("phi_t:", temporal.shape)
        
        # network learn nonlinear interactions between structure, time, semantic. but not between raw embeddings
        x = torch.cat([
                structural,
                temporal,
                semantic,
                h_prev,
                h_curr
            ], dim=-1)
        '''
        # the previous node contains information that is not present in h_v^T.h_x. Policy(x| u, v, t) and not policy(X|v,t)
        x = torch.cat([
                structural,
                temporal,
                semantic,
                h_prev,
                h_current,
                h_next
            ], dim=-1)
        '''    

        z = F.leaky_relu(self.proj(x), negative_slope=0.2)
        logits = self.attn(z).squeeze(-1)

        return logits

def train(model, train_loader, epochs=10, debug=False):

    seed = True
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:

            src = batch["src"]
            dst = batch["dst"]
            t = batch["ts"]

            #src, dst, t = src.item(), dst.item(), t.item()
            previous_node = src
            current_node = dst
            
            #neighbors, times = sampler.sample(node_id=dst, current_time=t, is_forward=True)
            neighbors, times = sampler.sample_k(node_id=dst, current_time=t, is_forward=True)
     
            #print(f"src: {}, neighbors: {neighbors}, times:{times}")
            
            if len(neighbors) == 0:
                continue

            deg_current = graph.get_degree(current_node)
            k = len(neighbors)

            # retorna o grau de uma lista de nós
            deg_neighbors  = graph.get_degree_neighbors(neighbors)
            neighbors_nodes = torch.tensor(neighbors, dtype=torch.long)

            previous_node = torch.full((k,), previous_node,  dtype=torch.long)
            current_node = torch.full((k,), current_node,  dtype=torch.long)
            deg_current = torch.full((k,), deg_current, dtype=torch.float)
            deg_neighbors = torch.tensor(deg_neighbors, dtype=torch.float)

            #delta_t = torch.tensor(t - times, dtype=torch.float)
            delta_t = torch.tensor(times - t, dtype=torch.float)
            
            if debug:
                print(f"src: {src}, dst: {dst}, current time: {t}, neighbors: {neighbors}, times:{times}, delta: {delta_t}")
                print("neighbors:", len(neighbors))
                print("deg_neighbors:", len(deg_neighbors))
                print("delta_t:", len(delta_t))
                print("current_node:", len(current_node))

            # model (u, v, deg_u, deg_v)  
            scores = model(
                previous_node,
                current_node, 
                neighbors_nodes,
                deg_current, 
                deg_neighbors,
                delta_t,    # derived from neighbors_v
            )

            # probs are target , it contain probabilities
            probs = torch.softmax(scores, dim=0)
  
            #target_index = np.where(neighbors == v)[0][0]
            #idx = np.random.choice(len(neighbors))
            idx = np.argmin(times)
            target_node = neighbors[idx]
            target_index = torch.tensor([idx], dtype=torch.long, device=device)
            if debug:
                print("scores : ", scores.unsqueeze(0))
                print(f"prob: {probs}, argmax {probs.argmax()}, sum: {probs.sum(axis=0)} ")

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

def test(model, test_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in test_loader:

            src = batch["src"][0].item()
            dst = batch["dst"][0].item()
            t   = batch["ts"][0].item()

            neighbors, times = sampler.sample(
                node_id=dst,
                current_time=t,
                is_forward=True
            )

            if len(neighbors) == 0:
                continue

            # -----------------------------
            # ground truth
            # -----------------------------

            target_idx = np.argmin(times)

            # -----------------------------
            # build tensors
            # -----------------------------

            k = len(neighbors)

            previous_node = torch.full(
                (k,),
                src,
                dtype=torch.long
            )

            current_node = torch.full((k,), dst, dtype=torch.long)
            neighbor_nodes = torch.tensor(neighbors, dtype=torch.long)

            deg_current = torch.full((k,), graph.get_degree(dst), dtype=torch.float)
            deg_neighbors = torch.tensor(graph.get_degree_neighbors(neighbors), dtype=torch.float)

            delta_t = torch.tensor(times - t, dtype=torch.float)

            # -----------------------------
            # prediction
            # -----------------------------

            scores = model(
                previous_node,
                current_node,
                neighbor_nodes,
                deg_current,
                deg_neighbors,
                delta_t
            )

            pred_idx = scores.argmax().item()

            if pred_idx == target_idx:
                correct += 1

            total += 1

    accuracy = correct / total

    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy


def train2(model, train_loader, epochs=10, debug=False):

    for epoch in range(epochs):

        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:

            src_batch = batch["src"]
            dst_batch = batch["dst"]
            ts_batch  = batch["ts"]

            B = len(src_batch)

            batch_prev = []
            batch_curr = []
            batch_neighbors = []
            batch_deg_current = []
            batch_deg_neighbors = []
            batch_delta_t = []
            batch_masks = []
            batch_targets = []

            for i in range(B):

                src = src_batch[i].item()
                dst = dst_batch[i].item()
                ts  = ts_batch[i].item()

                neighbors, times, mask, n_valid = sampler.sample_k(
                    node_id=dst,
                    current_time=ts,
                    is_forward=True
                )

                if n_valid == 0:
                    continue

                deg_current = graph.get_degree(dst)
                deg_neighbors = graph.get_degree_neighbors(neighbors)

                delta_t = times - ts
                delta_t[~mask] = 0
                #print(delta_t.min())

                batch_prev.append(src)
                batch_curr.append(dst)

                batch_neighbors.append(neighbors)
                batch_deg_current.append(deg_current)
                batch_deg_neighbors.append(deg_neighbors)
                batch_delta_t.append(delta_t)
                batch_masks.append(mask)

                # earliest future interaction
                batch_targets.append(0)

            if len(batch_prev) == 0:
                continue

            # ------------------------------------
            # Convert to tensors
            # ------------------------------------

            previous_nodes = torch.tensor(
                batch_prev,
                dtype=torch.long,
                device=device
            )

            current_nodes = torch.tensor(
                batch_curr,
                dtype=torch.long,
                device=device
            )

            neighbor_nodes = torch.tensor(
                np.stack(batch_neighbors),
                dtype=torch.long,
                device=device
            )

            deg_current = torch.tensor(
                batch_deg_current,
                dtype=torch.float,
                device=device
            )

            deg_neighbors = torch.tensor(
                np.stack(batch_deg_neighbors),
                dtype=torch.float,
                device=device
            )

            delta_t = torch.tensor(
                np.stack(batch_delta_t),
                dtype=torch.float,
                device=device
            )

            mask = torch.tensor(
                np.stack(batch_masks),
                dtype=torch.bool,
                device=device
            )

            target_idx = torch.tensor(
                batch_targets,
                dtype=torch.long,
                device=device
            )

            # ------------------------------------
            # Forward
            # ------------------------------------

            scores = model(
                previous_nodes,
                current_nodes,
                neighbor_nodes,
                deg_current,
                deg_neighbors,
                delta_t,
            )

            # padded candidates receive zero probability
            scores = scores.masked_fill(
                ~mask,
                -1e9
            )

            loss = criterion(
                scores,
                target_idx
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = (
            total_loss / max(num_batches, 1)
        )

        print(
            f"Epoch {epoch:03d} "
            f"| Loss {avg_loss:.4f}"
        )

@torch.no_grad()
def predict_next_node(
    model,
    graph,
    sampler,
    previous_node,
    current_node,
    current_time,
    device,
):
    model.eval()

    neighbors, times, mask, n_valid = sampler.sample_k(
        node_id=current_node,
        current_time=current_time,
        is_forward=True
    )

    if n_valid == 0:
        return None

    deg_current = graph.get_degree(current_node)
    deg_neighbors = graph.get_degree_neighbors(neighbors)

    delta_t = times - current_time
    delta_t[~mask] = 0

    previous_nodes = torch.tensor(
        [previous_node],
        dtype=torch.long,
        device=device
    )

    current_nodes = torch.tensor(
        [current_node],
        dtype=torch.long,
        device=device
    )

    neighbor_nodes = torch.tensor(
        neighbors[None, :],
        dtype=torch.long,
        device=device
    )

    deg_current = torch.tensor(
        [deg_current],
        dtype=torch.float,
        device=device
    )

    deg_neighbors = torch.tensor(
        deg_neighbors,
        dtype=torch.float,
        device=device
    ).unsqueeze(0)

    delta_t = torch.tensor(
        delta_t,
        dtype=torch.float,
        device=device
    ).unsqueeze(0)

    mask = torch.tensor(
        mask,
        dtype=torch.bool,
        device=device
    ).unsqueeze(0)

    scores = model(
        previous_nodes,
        current_nodes,
        neighbor_nodes,
        deg_current,
        deg_neighbors,
        delta_t,
    )

    scores = scores.masked_fill(
        ~mask,
        -1e9
    )

    probs = torch.softmax(
        scores,
        dim=-1
    )

    pred_idx = probs.argmax(dim=-1).item()

    predicted_node = neighbors[pred_idx]

    return {
        "predicted_node": int(predicted_node),
        "probability": float(probs[0, pred_idx]),
        "neighbors": neighbors[:n_valid],
        "probabilities": probs[0, :n_valid].cpu().numpy(),
    }

@torch.no_grad()
def evaluate(model, test_loader):

    model.eval()

    correct = 0
    total = 0

    for batch in test_loader:

        src_batch = batch["src"]
        dst_batch = batch["dst"]
        ts_batch  = batch["ts"]

        B = len(src_batch)

        for i in range(B):

            src = src_batch[i].item()
            dst = dst_batch[i].item()
            ts  = ts_batch[i].item()

            neighbors, times, mask, n_valid = sampler.sample_k(
                node_id=dst,
                current_time=ts,
                is_forward=True
            )

            if n_valid == 0:
                continue

            result = predict_next_node(
                model,
                graph,
                sampler,
                src,
                dst,
                ts,
                device
            )

            predicted = result["predicted_node"]

            true_node = neighbors[0]

            if predicted == true_node:
                correct += 1

            total += 1

    acc = correct / max(total, 1)

    print(
        f"Accuracy = {acc:.4f}"
    )

    return acc

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

'''
batch_size = 2
num_neighbors = 2
is_forward = True
df = pd.DataFrame(data)
df = df.sort_values(by='timestamps', ascending=is_forward)

sources = df.source.values
destinations = df.target.values
timestamps = df.timestamps.values

train_data = EdgeDataset(sources, destinations, timestamps, edge_idxs, labels)

'''

random.seed(2020)

batch_size = 64
num_neighbors = 30
#graph_df = graph_df.head(1000)
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



graph = GraphStorage(sources, destinations, timestamps)
PAD_NODE = graph.num_nodes()

max_node_id = max(graph.get_nodes())
NUM_NODES = graph.num_nodes()

sampler = TemporalNeighborSampler(graph, num_neighbors=num_neighbors, pad_node=PAD_NODE)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

DEBUG = False

print("Num nodes:", NUM_NODES)

if max_node_id < NUM_NODES :
    NUM_NODES = max_node_id + 1

print("Max : ", max_node_id)
print("Num nodes:", NUM_NODES)

PAD_NODE = NUM_NODES
# create models
model = TemporalTransitionModel(
    num_nodes=NUM_NODES,
    embedding_dim=32,
    time_dim=16,
    debug=DEBUG,
    pad_node=PAD_NODE
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

'''
This method Learn a temporal walk continuation policy. Where given a (u, v, t) we know who is the next time 
but we can not predict the future link between two nodes. And the accuracy is too low 33%

The problem with testing function is that I am feeding the time to the model. 
Right now your model is trying to do both:
 - Learn a walk continuation policy.
 - Predict future edges.
Those are related, but not identical tasks.

Advantange:

 - A learned temporal walk policy can learn these temporal paths contain information that simple edge prediction misses:
   airport network
    JFK -> ATL
    ATL -> DFW
    DFW -> LAX
    for example : JFK rarely connects directly to LAX

    but
     JFK -> ATL -> DFW -> LAX
    occurs frequently
    A learned policy can capture P(X| (u, v, t)) Given I'm currently at v, where does traffic tend to flow next?
'''

epochs = 5
train2(model, train_loader, epochs=epochs, debug=DEBUG)

evaluate(model, test_loader)
