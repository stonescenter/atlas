import numpy as np
import pandas as pd
import random

import torch
from torch.utils.data import DataLoader

import math
from torch import Tensor
import torch_geometric.transforms as T
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.graph import GraphStorage
from utils.sampler import TemporalNeighborSampler
from utils.data_processing import EdgeDataset
from model.time_encoder import TimeEncoder 
from utils.data_processing import get_data, Graph, TemporalWalkDataset, TemporalWalkLinkDataset, collate_temporal_walk

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
    
class TemporalWalkEncoder(nn.Module):

    def __init__(
        self,
        num_nodes,
        embedding_dim=64,
        time_dim=32,
        hidden_dim=128,
        pad_node=None,
        debug = False
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_nodes + 1,
            embedding_dim,
            padding_idx=pad_node,
        )

        self.time_encoder = TimeEncoder(
            time_dim
        )

        temporal_dim = 2 * time_dim

        input_dim = (
            1 +                    # structural
            temporal_dim +         # time encoding
            embedding_dim +        # h_prev * h_next
            embedding_dim +        # h_curr * h_next
            3 * embedding_dim      # h_prev,h_curr,h_next
        )

        self.encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU(),
        )

        self.debug = debug

    def structural_score(
        self,
        deg_current,
        deg_neighbor,
    ):
        deg_current = deg_current.unsqueeze(1)

        return (
            1.0 /
            torch.sqrt(
                deg_current * deg_neighbor + 1e-8
            )
        ).unsqueeze(-1)

    def forward(
        self,
        previous_nodes,
        current_nodes,
        neighbor_nodes,
        deg_current,
        deg_neighbors,
        delta_t,
        mask=None,
    ):
        """
        previous_nodes : [B]
        current_nodes  : [B]

        neighbor_nodes : [B,K]

        deg_current    : [B]
        deg_neighbors  : [B,K]

        delta_t        : [B,K]

        returns:
            z_walk : [B,K,H]
        """

        B, K = neighbor_nodes.shape

        h_prev = self.embedding(
            previous_nodes
        )

        h_curr = self.embedding(
            current_nodes
        )

        h_next = self.embedding(
            neighbor_nodes
        )

        h_prev = (
            h_prev
            .unsqueeze(1)
            .expand(-1, K, -1)
        )

        h_curr = (
            h_curr
            .unsqueeze(1)
            .expand(-1, K, -1)
        )

        structural = torch.log(
            self.structural_score(
                deg_current,
                deg_neighbors
            ) + 1e-8
        )

        temporal = self.time_encoder(
            delta_t
        )

        prev_next = h_prev * h_next
        curr_next = h_curr * h_next

        x = torch.cat(
            [
                structural,
                temporal,
                prev_next,
                curr_next,
                h_prev,
                h_curr,
                h_next,
            ],
            dim=-1,
        )

        z_walk = self.encoder(x)

        if mask is not None:
            z_walk = z_walk.masked_fill(
                ~mask.unsqueeze(-1),
                0.0
            )

        return z_walk

class TemporalLinkPredictor(nn.Module):

    def __init__(
        self,
        hidden_dim=128,
    ):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(
        self,
        z_walk,
    ):
        return self.score(
            z_walk
        ).squeeze(-1)
    

class TemporalWalkLinkPrediction(nn.Module):        

    def __init__(
        self,
        encoder,
        predictor,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor

    def forward(
        self,
        previous_nodes,
        current_nodes,
        neighbor_nodes,
        deg_current,
        deg_neighbors,
        delta_t,
        mask=None,
    ):

        z_walk = self.encoder(
            previous_nodes,
            current_nodes,
            neighbor_nodes,
            deg_current,
            deg_neighbors,
            delta_t,
            mask,
        )

        scores = self.predictor(
            z_walk
        )

        return scores

class TemporalWalkModel(nn.Module):
    def __init__(
        self,
        num_nodes,
        embedding_dim=64,
        time_dim=32,
        hidden_dim=128,
        pad_node=None,
        dropout=0.1,
        debug=False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_nodes + 1,
            embedding_dim,
            padding_idx=pad_node,
        )

        self.time_encoder = TimeEncoder(time_dim)

        temporal_dim = 2 * time_dim

        input_dim = (
            1 +                 # structural score
            temporal_dim +      # time encoding
            embedding_dim +     # h_prev * h_next
            embedding_dim +     # h_curr * h_next
            3 * embedding_dim   # h_prev, h_curr, h_next
        )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Walk policy head: predicts next candidate among K neighbors
        self.walk_head = nn.Linear(hidden_dim, 1)

        # Attention pooling for link prediction
        self.pool_attn = nn.Linear(hidden_dim, 1)

        # Link prediction head: predicts whether edge exists
        self.link_head = nn.Sequential(
            nn.Linear(hidden_dim + 2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.debug = debug

    def structural_score(self, deg_current, deg_neighbors):
        """
        deg_current   : [B]
        deg_neighbors : [B,K]

        returns:
            structural : [B,K,1]
        """

        deg_current = deg_current.unsqueeze(1)  # [B,1]

        deg_current = torch.clamp(deg_current, min=1.0)
        deg_neighbors = torch.clamp(deg_neighbors, min=1.0)

        score = 1.0 / torch.sqrt(
            deg_current * deg_neighbors + 1e-8
        )

        return torch.log(score + 1e-8).unsqueeze(-1)

    def encode_walks(
        self,
        previous_nodes,
        current_nodes,
        neighbor_nodes,
        deg_current,
        deg_neighbors,
        delta_t,
        mask=None,
    ):
        """
        previous_nodes : [B]
        current_nodes  : [B]
        neighbor_nodes : [B,K]
        deg_current    : [B]
        deg_neighbors  : [B,K]
        delta_t        : [B,K]
        mask           : [B,K]

        returns:
            z_walk      : [B,K,H]
        """

        B, K = neighbor_nodes.shape

        h_prev = self.embedding(previous_nodes)      # [B,D]
        h_curr = self.embedding(current_nodes)       # [B,D]
        h_next = self.embedding(neighbor_nodes)      # [B,K,D]

        h_prev_exp = h_prev.unsqueeze(1).expand(-1, K, -1)
        h_curr_exp = h_curr.unsqueeze(1).expand(-1, K, -1)

        # Avoid invalid temporal values for padded entries
        delta_t = torch.clamp(delta_t, min=0.0)

        structural = self.structural_score(
            deg_current,
            deg_neighbors,
        )                                            # [B,K,1]

        temporal = self.time_encoder(delta_t)        # [B,K,2*time_dim]

        prev_next = h_prev_exp * h_next
        curr_next = h_curr_exp * h_next

        x = torch.cat(
            [
                structural,
                temporal,
                prev_next,
                curr_next,
                h_prev_exp,
                h_curr_exp,
                h_next,
            ],
            dim=-1,
        )

        z_walk = self.encoder(x)                     # [B,K,H]

        if mask is not None:
            z_walk = z_walk.masked_fill(
                ~mask.unsqueeze(-1),
                0.0,
            )

        if self.debug:
            print("h_prev:", h_prev.shape)
            print("h_curr:", h_curr.shape)
            print("h_next:", h_next.shape)
            print("structural:", structural.shape)
            print("temporal:", temporal.shape)
            print("z_walk:", z_walk.shape)

        return z_walk, h_prev, h_curr

    def pool_walks(self, z_walk, mask):
        """
        z_walk : [B,K,H]
        mask   : [B,K]

        returns:
            z_pool : [B,H]
        """

        attn_logits = self.pool_attn(z_walk).squeeze(-1)  # [B,K]

        attn_logits = attn_logits.masked_fill(
            ~mask,
            -1e9,
        )

        attn = torch.softmax(attn_logits, dim=-1)         # [B,K]

        z_pool = torch.sum(
            z_walk * attn.unsqueeze(-1),
            dim=1,
        )                                                 # [B,H]

        return z_pool

    def forward(
        self,
        previous_nodes,
        current_nodes,
        neighbor_nodes,
        deg_current,
        deg_neighbors,
        delta_t,
        mask,
    ):
        """
        returns:
            walk_logits : [B,K]
            edge_score  : [B]
        """

        z_walk, h_prev, h_curr = self.encode_walks(
            previous_nodes,
            current_nodes,
            neighbor_nodes,
            deg_current,
            deg_neighbors,
            delta_t,
            mask,
        )

        # Output 1: temporal walk policy
        walk_logits = self.walk_head(z_walk).squeeze(-1)  # [B,K]

        # Output 2: temporal edge prediction
        z_pool = self.pool_walks(z_walk, mask)            # [B,H]

        edge_repr = torch.cat(
            [
                z_pool,
                h_prev,
                h_curr,
            ],
            dim=-1,
        )

        edge_score = self.link_head(edge_repr).squeeze(-1)  # [B]

        return walk_logits, edge_score
    
def train1(model, train_loader, epochs=10, debug=False):

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

def train2(model, train_loader, epochs=20, debug=False):

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch in train_loader:

            previous_nodes = batch["src"].to(
                device,
                non_blocking=True
            )

            current_nodes = batch["dst"].to(
                device,
                non_blocking=True
            )

            deg_current = batch["deg_current"].to(
                device,
                non_blocking=True
            )

            neighbor_nodes = batch["neighbors"].to(
                device,
                non_blocking=True
            )

            deg_neighbors = batch["deg_neighbors"].to(
                device,
                non_blocking=True
            )

            delta_t = batch["delta_t"].to(
                device,
                non_blocking=True
            )

            mask = batch["mask"].to(
                device,
                non_blocking=True
            )

            target_idx = batch["target_idx"].to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

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

            loss = criterion(
                scores,
                target_idx
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss /
            len(train_loader)
        )

        print(
            f"Epoch {epoch:03d}"
            f" | Loss {avg_loss:.4f}"
        )

def train3(
    model,
    train_loader,
    optimizer,
    device,
    epochs=20,
    lambda_walk=0.3,
    lambda_link=0.7,
):

    walk_criterion = nn.CrossEntropyLoss()

    link_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        total_walk_loss = 0
        total_link_loss = 0

        for batch in train_loader:

            previous_nodes = batch["src"].long().to(device)

            current_nodes = batch["dst"].long().to(device)

            negative_nodes = batch["neg_dst"].int().to(device)

            deg_current = batch["deg_current"].int().to(device)

            neighbor_nodes = batch["neighbors"].int().to(device)

            deg_neighbors = batch["deg_neighbors"].float().to(device)

            delta_t = batch["delta_t"].float().to(device)

            mask = batch["mask"].bool().to(device)

            target_idx = batch["target_idx"].int().to(device)

            ################################################
            # POSITIVE EDGE
            ################################################

            walk_logits, pos_score = model(
                previous_nodes,
                current_nodes,
                neighbor_nodes,
                deg_current,
                deg_neighbors,
                delta_t,
                mask,
            )

            walk_logits = walk_logits.masked_fill(
                ~mask,
                -1e9
            )

            walk_loss = walk_criterion(
                walk_logits,
                target_idx
            )

            ################################################
            # NEGATIVE EDGE
            ################################################

            _, neg_score = model(
                previous_nodes,
                negative_nodes,
                neighbor_nodes,
                deg_current,
                deg_neighbors,
                delta_t,
                mask,
            )

            ################################################
            # LINK LOSS
            ################################################

            pos_labels = torch.ones_like(
                pos_score
            )

            neg_labels = torch.zeros_like(
                neg_score
            )

            link_loss_pos = link_criterion(
                pos_score,
                pos_labels
            )

            link_loss_neg = link_criterion(
                neg_score,
                neg_labels
            )

            link_loss = (
                link_loss_pos +
                link_loss_neg
            )

            ################################################
            # TOTAL LOSS
            ################################################

            loss = (
                lambda_walk * walk_loss +
                lambda_link * link_loss
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()

            total_walk_loss += walk_loss.item()

            total_link_loss += link_loss.item()

        n_batches = len(train_loader)

        print(
            f"Epoch {epoch:03d} "
            f"| Total {total_loss/n_batches:.4f} "
            f"| Walk {total_walk_loss/n_batches:.4f} "
            f"| Link {total_link_loss/n_batches:.4f}"
        )


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


train_walks = TemporalWalkDataset(train_data, graph, sampler)

train_loader = DataLoader(
    train_walks,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_temporal_walk,
    num_workers=4,
    pin_memory=True,
)

# create models
model = TemporalWalkEncoder(
    num_nodes=NUM_NODES,
    embedding_dim=32,
    time_dim=16,
    pad_node=PAD_NODE,
    debug=DEBUG
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
walk_criterion = nn.CrossEntropyLoss()
link_criterion = nn.BCEWithLogitsLoss()


train2(model, train_loader, epochs=5, debug=DEBUG)

#evaluate(model, test_loader)


# this is the architecture
'''
(u,v,t)
      |
      |
      V
TemporalWalkEncoder
      |
      |
      V
      z
     // \\
    //   \\
   //     \\
WalkHead  LinkHead
   |         |
   |         |
walk_logits edge_score

'''