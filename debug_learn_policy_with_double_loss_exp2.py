import os
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
from utils.data_processing import get_data, Graph, TemporalWalkSupervisionDataset, TemporalWalkDataset, TemporalWalkLinkDataset
from utils.data_processing import collate_temporal_walk, collate_temporal_walk_link
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.plots import *
from utils.util import gradient_direction_stats

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device available ", device)

def set_seed(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    
def train(
        model, 
        train_loader,
        device, 
        epochs=20, 
        lambda_walk=0.3,  
        lambda_link=0.7,
        debug=False
    ):


    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_walk = 0.0
        total_link = 0.0

        epoch_scores = []
        epoch_labels = []

        for batch in train_loader:
            src = batch["src"].long().to(device)
            dst = batch["dst"].long().to(device)
            neg_dst = batch["neg_dst"].long().to(device)

            target_idx = batch["target_idx"].long().to(device)

            # positive context
            pos_neighbors = batch["pos_neighbors"].long().to(device)
            pos_deg_current = batch["pos_deg_current"].long().to(device)
            pos_deg_neighbors = batch["pos_deg_neighbors"].long().to(device)
            pos_delta_t = batch["pos_delta_t"].float().to(device)
            pos_mask = batch["pos_mask"].bool().to(device)

            # negative context
            neg_neighbors = batch["neg_neighbors"].long().to(device)
            neg_deg_current = batch["neg_deg_current"].long().to(device)
            neg_deg_neighbors = batch["neg_deg_neighbors"].long().to(device)
            neg_delta_t = batch["neg_delta_t"].float().to(device)
            neg_mask = batch["neg_mask"].bool().to(device)

            # positive edge: (src, dst, t)
            walk_logits, pos_score = model(
                src,
                dst,
                pos_neighbors,
                pos_deg_current,
                pos_deg_neighbors,
                pos_delta_t,
                pos_mask,
            )

            walk_logits = walk_logits.masked_fill(~pos_mask, -1e9)

            walk_loss = walk_criterion(walk_logits, target_idx )

            # negative edge: (src, neg_dst, t)
            _, neg_score = model(
                src,
                neg_dst,
                neg_neighbors,
                neg_deg_current,
                neg_deg_neighbors,
                neg_delta_t,
                neg_mask,
            )

            pos_labels = torch.ones_like(pos_score)
            neg_labels = torch.zeros_like(neg_score)

            link_loss = ( link_criterion(pos_score, pos_labels) +   (neg_score, neg_labels))

            loss = (lambda_walk * walk_loss + lambda_link * link_loss)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()
            total_walk += walk_loss.item()
            total_link += link_loss.item()

            scores = torch.cat([pos_score, neg_score], dim=0)
            labels = torch.cat([pos_labels, neg_labels], dim=0)

            epoch_scores.append(torch.sigmoid(scores).detach().cpu())
            epoch_labels.append(labels.detach().cpu())

        n = max(len(train_loader), 1)
        y_score = torch.cat(epoch_scores).numpy()
        y_true = torch.cat(epoch_labels).numpy()

        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        print(
            f"Epoch {epoch:03d} "
            f"| Total {total_loss/n:.4f} "
            f"| Walk {total_walk/n:.4f} "
            f"| Link {total_link/n:.4f} "
            f"| AUC {auc:.4f} "
            f"| AP {ap:.4f}"
        )

def train_walk_policy(
    model, 
    train_loader, 
    device,
    optimizer,
    link_criterion,
    epochs=20,
    lambda_walk=0.5,  
    lambda_link=0.5,    
    supervision_mode="earliest",
    track_gradients=False,
    gradient_plot_path=None,
    temperature=1.0, 
    label_smoothing=0.05,
    gradient_every=1, 
    debug=False
):
    assert supervision_mode in {
        "earliest",
        "sampled",
        "soft",
    }

    gradient_history = []
    global_step = 0

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_walk = 0.0
        total_link = 0.0

        epoch_scores = []
        epoch_labels = []

        for batch in train_loader:
            previous_nodes = batch["src"].long().to(device)
            current_nodes = batch["dst"].long().to(device)
            neg_dst = batch["neg_dst"].long().to(device)

            # positive context
            pos_neighbors = batch["pos_neighbors"].long().to(device)
            pos_deg_current = batch["pos_deg_current"].long().to(device)
            pos_deg_neighbors = batch["pos_deg_neighbors"].long().to(device)
            pos_delta_t = batch["pos_delta_t"].float().to(device)
            pos_mask = batch["pos_mask"].bool().to(device)

            # negative context
            neg_neighbors = batch["neg_neighbors"].long().to(device)
            neg_deg_current = batch["neg_deg_current"].long().to(device)
            neg_deg_neighbors = batch["neg_deg_neighbors"].long().to(device)
            neg_delta_t = batch["neg_delta_t"].float().to(device)
            neg_mask = batch["neg_mask"].bool().to(device)

            walk_logits, pos_score = model(
                previous_nodes,
                current_nodes,
                pos_neighbors,
                pos_deg_current,
                pos_deg_neighbors,
                pos_delta_t,
                pos_mask,
            )

            walk_logits = walk_logits.masked_fill(~pos_mask, -1e9)

            # Experiments 1 and 2
            if supervision_mode in {"earliest", "sampled"}:
                target_idx = batch["target_idx"].long().to(device)
                target_idx = target_idx.clamp(min=0, max=walk_logits.size(1) - 1)
                has_valid = pos_mask.any(dim=1)
                if has_valid.any():
                    first_valid = pos_mask.long().argmax(dim=1)
                    target_idx = torch.where(has_valid, target_idx, first_valid.to(device))
                walk_loss = F.cross_entropy(walk_logits / temperature, target_idx, label_smoothing=label_smoothing)

            # Experiment 3
            elif supervision_mode == "soft":
                target_probs = batch["target_probs"].float().to(device)
                #log_probs = F.log_softmax(walk_logits, dim=-1)
                log_probs = F.log_softmax(walk_logits / temperature, dim=-1)
                walk_loss = F.kl_div(log_probs, target_probs, reduction="batchmean")

            # negative edge: (src, neg_dst, t)
            _, neg_score = model(
                previous_nodes,
                neg_dst,
                neg_neighbors,
                neg_deg_current,
                neg_deg_neighbors,
                neg_delta_t,
                neg_mask,
            )

            pos_labels = torch.ones_like(pos_score)
            neg_labels = torch.zeros_like(neg_score)
            link_loss = ( link_criterion(pos_score, pos_labels) +  link_criterion(neg_score, neg_labels))
            loss = ( lambda_walk * walk_loss + lambda_link * link_loss)

            if track_gradients and global_step % gradient_every == 0:
                grad_stats = gradient_direction_stats(model, walk_loss, link_loss)
                grad_stats["step"] = global_step
                grad_stats["epoch"] = epoch
                gradient_history.append(grad_stats)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_walk += walk_loss.item()
            total_link += link_loss.item()

            scores = torch.cat([pos_score, neg_score], dim=0)
            labels = torch.cat([pos_labels, neg_labels], dim=0)

            epoch_scores.append(torch.sigmoid(scores).detach().cpu())
            epoch_labels.append(labels.detach().cpu())

            global_step += 1

        n = max(len(train_loader), 1)
        y_score = torch.cat(epoch_scores).numpy()
        y_true = torch.cat(epoch_labels).numpy()

        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        grad_msg = ""
        if track_gradients and gradient_history:
            epoch_cosines = [
                row["cosine"]
                for row in gradient_history
                if row["epoch"] == epoch and not np.isnan(row["cosine"])
            ]
            if epoch_cosines:
                grad_msg = f" | GradCos {np.mean(epoch_cosines):.4f}"

        print(
            f"Epoch {epoch:03d} "
            f"| Total {total_loss/n:.4f} " 
            f"| Walk {total_walk/n:.4f} "
            f"| Link {total_link/n:.4f} "
            f"| AUC {auc:.4f} "
            f"| AP {ap:.4f}"
            f"{grad_msg}"
        )

    if track_gradients and gradient_plot_path is not None:
        try:
            plot_gradient_directions(gradient_history, gradient_plot_path)
            #print(f"Gradient direction plot saved to {gradient_plot_path}")
        except Exception as exc:
            print(f"Warning: could not save gradient plot to {gradient_plot_path}: {exc}")

    return gradient_history

@torch.no_grad()
def evaluate_link_prediction(model, loader, device):
    model.eval()

    all_scores = []
    all_labels = []

    for batch in loader:
        src = batch["src"].long().to(device)
        dst = batch["dst"].long().to(device)
        neg_dst = batch["neg_dst"].long().to(device)

        pos_neighbors = batch["pos_neighbors"].long().to(device)
        pos_deg_current = batch["pos_deg_current"].float().to(device)
        pos_deg_neighbors = batch["pos_deg_neighbors"].float().to(device)
        pos_delta_t = batch["pos_delta_t"].float().to(device)
        pos_mask = batch["pos_mask"].bool().to(device)

        neg_neighbors = batch["neg_neighbors"].long().to(device)
        neg_deg_current = batch["neg_deg_current"].float().to(device)
        neg_deg_neighbors = batch["neg_deg_neighbors"].float().to(device)
        neg_delta_t = batch["neg_delta_t"].float().to(device)
        neg_mask = batch["neg_mask"].bool().to(device)

        _, pos_score = model(
            src,
            dst,
            pos_neighbors,
            pos_deg_current,
            pos_deg_neighbors,
            pos_delta_t,
            pos_mask,
        )

        _, neg_score = model(
            src,
            neg_dst,
            neg_neighbors,
            neg_deg_current,
            neg_deg_neighbors,
            neg_delta_t,
            neg_mask,
        )

        scores = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score),
        ], dim=0)

        scores = torch.sigmoid(scores)

        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())

    y_score = torch.cat(all_scores).numpy()
    y_true = torch.cat(all_labels).numpy()

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    print(f"Test AUC-ROC: {auc:.4f} | AP: {ap:.4f}")

    return {
        "auc_roc": auc,
        "ap": ap,
    }

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
#graph_df = graph_df.head(5000)
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

DEBUG = False

print("Num nodes:", NUM_NODES)

if max_node_id < NUM_NODES :
    NUM_NODES = max_node_id + 1

print("Max : ", max_node_id)
print("Num nodes:", NUM_NODES)

PAD_NODE = NUM_NODES

sampler = TemporalNeighborSampler(graph, num_neighbors=num_neighbors, pad_node=PAD_NODE)

test_walks = TemporalWalkLinkDataset(test_data, graph, sampler, num_nodes=NUM_NODES)
test_loader = DataLoader(test_walks, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_temporal_walk_link)


epochs = 10
lambda_walk = 0.3
lambda_link = 0.7

lambda_walks =  [0.1]
lambda_links =  [0.9]
lambda_walks =  [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 ]
lambda_links =  [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]

modes = ["earliest", "sampled", "soft"]

optimizer = None

experiment = "results/exp2"
os.makedirs(experiment, exist_ok=True)

for mode in modes:
    train_dataset = TemporalWalkSupervisionDataset(train_data, graph, sampler, num_nodes=NUM_NODES, supervision_mode=mode)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_temporal_walk_link)

    for lambda_walk, lambda_link in zip(lambda_walks, lambda_links):
        set_seed(2020)

        model = TemporalWalkModel(
            num_nodes=NUM_NODES,
            embedding_dim=32, 
            time_dim=16, 
            pad_node=PAD_NODE, 
            debug=False).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        link_criterion = nn.BCEWithLogitsLoss()

        print(f"Model params:")
        print(f"\tlambda_walk: {lambda_walk}, lambda_link: {lambda_link}")
        print(f"\tsupervision_mode: {mode}")
        print("Training...")

        path_plot = f"{experiment}/gradient_cosine_walk_{mode}_{lambda_walk}_link_{lambda_link}.png"
        train_walk_policy(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            link_criterion=link_criterion,
            epochs=epochs,
            lambda_walk=lambda_walk,
            lambda_link=lambda_link,
            supervision_mode=mode,
            track_gradients=True,
            gradient_plot_path=path_plot,
            gradient_every=10,
            temperature=1.2,
            label_smoothing=0.05,
            debug=False,
        )
        evaluate_link_prediction(model, test_loader, device)


# Multi-Task Temporal Walk Policy Network for Temporal Link Prediction (Temporal Walk Policy + Link Prediction)
# The model learns two related tasks simultaneously:
#   Task 1: Temporal Walk Policy  : π(x∣u,v,t)
#     Ex: Question:
#       Given that a temporal walk arrived from node u to node v at time t,
#       which candidate node x is the most likely continuation?
#  Task 2: Temporal Link Prediction: P((u,v,t+Δt))
#      Ex: Question:
#       Will an interaction between nodes u and v occur in the future?


# Scientific hypothesis
# The central hypothesis of the method is:
#   Learning a temporal transition policy captures dynamic interaction patterns that improve future edge prediction.

# Architecture
'''
     (u,v,t)
        |
        |   
TemporalWalkEncoder
        |
        |
      z_walk
     // \\
    //   \\
   //     \\
WalkHead  LinkHead
   |         |
   |         |
walk_logits edge_score

'''
# results in results_debug_double_loss-23-06-2026.txt