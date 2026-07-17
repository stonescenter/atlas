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
from utils.data_processing import EdgeDataset, ObservedTrajectoryIndex
from model.time_encoder import TimeEncoder 
from utils.data_processing import get_data, Graph, TemporalWalkSupervisionDataset, TemporalWalkLinkDataset
from utils.data_processing import collate_temporal_walk, collate_temporal_walk_link, TemporalWalkSupervisionDataDriven
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.plots import *
from utils.util import gradient_direction_stats
from model.atlas import TemporalWalkEncoder

if torch.cuda.is_available():
    dev = 'cuda'
else:
    dev = 'cpu'
    print("Device cuda not available, using cpu")    

device = torch.device(dev)

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
    
class TemporalWalkEncoderOld(nn.Module):

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
                #walk_loss = F.cross_entropy(walk_logits, target_idx)
                walk_loss = F.cross_entropy(walk_logits / temperature, target_idx, label_smoothing=label_smoothing)

            # Experiment 3
            elif supervision_mode == "soft":
                target_probs = batch["target_probs"].float().to(device)
                if target_probs.ndim != 2:
                    target_probs = target_probs.unsqueeze(0)
                if target_probs.shape[1] != walk_logits.shape[1]:
                    target_probs = target_probs[:, : walk_logits.shape[1]]
                    if target_probs.shape[1] < walk_logits.shape[1]:
                        pad = torch.zeros(
                            (target_probs.shape[0], walk_logits.shape[1] - target_probs.shape[1]),
                            dtype=target_probs.dtype,
                            device=target_probs.device,
                        )
                        target_probs = torch.cat([target_probs, pad], dim=1)
                valid_mask = pos_mask.to(target_probs.dtype)
                target_probs = target_probs * valid_mask
                target_probs = target_probs / (target_probs.sum(dim=-1, keepdim=True) + 1e-12)
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

    #print("Scores:", len(all_scores))
    #print("Labels:", len(all_labels))

    if not all_scores or not all_labels:
        print("Warning: evaluation loader produced no examples; returning NaN metrics.")
        return {"auc_roc": float("nan"), "ap": float("nan")}

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

datasets = ['wikipedia', 'enron', 'collegemsg', 'mooc', 'reddit']
datasets = ['wikipedia']

random.seed(2020)


batch_size = 64
num_neighbors = 30
epochs = 10
lambda_walks = [0.3]
lambda_links = [0.7]
modes = ["earliest", "sampled", "soft"]

modes = ["soft"]

experiment_root = "results/exp2"
os.makedirs(experiment_root, exist_ok=True)

n_runs = 5
optimizer = None

for dataset_name in datasets:
    data_file = os.path.join(PATH_DATASET, f"ml_{dataset_name}.csv")
    if not os.path.exists(data_file):
        print(f"Skipping dataset {dataset_name}: file not found at {data_file}")
        continue

    print(f"\n=== Dataset: {dataset_name} ===")
    graph_df = pd.read_csv(data_file)
    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    graph_df = graph_df.head(1000)
    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    #train_mask = timestamps <= test_time
    #test_mask = timestamps > test_time
    #val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    
    # much better configuration for training, validation, and testing splits 
    # means training has access to events occurring between 70% and 85% of the timeline.
    # the previous configurarion has already seen much more recent interactions than intended
    train_mask = timestamps <= val_time
    val_mask = (timestamps > val_time) & (timestamps <= test_time)
    test_mask = timestamps > test_time

    train_data = EdgeDataset(
        sources[train_mask],
        destinations[train_mask],
        timestamps[train_mask],
        edge_idxs[train_mask],
        labels[train_mask],
    )

    test_data = EdgeDataset(
        sources[test_mask],
        destinations[test_mask],
        timestamps[test_mask],
        edge_idxs[test_mask],
        labels[test_mask],
    )

    graph_train = GraphStorage(
        sources[train_mask],
        destinations[train_mask],
        timestamps[train_mask],
    )
    graph_eval = GraphStorage(
        sources[test_mask],
        destinations[test_mask],
        timestamps[test_mask],
    )

    all_nodes = set(sources) | set(destinations)
    max_node_id = max(all_nodes) if all_nodes else 0
    NUM_NODES = int(max_node_id) + 1
    PAD_NODE = NUM_NODES

    print("Num nodes:", NUM_NODES)
    print("Max : ", max_node_id)

    sampler_train = TemporalNeighborSampler(graph_train, num_neighbors=num_neighbors, pad_node=PAD_NODE)
    sampler_eval = TemporalNeighborSampler(graph_eval, num_neighbors=num_neighbors, pad_node=PAD_NODE)

    test_walks = TemporalWalkLinkDataset(test_data, graph_eval, sampler_eval, num_nodes=NUM_NODES)
    print("test_walks", len(test_walks)) 
    test_loader = DataLoader(
        test_walks,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_temporal_walk_link,
    )
    print("test_loader:", len(test_loader))
    
    experiment = os.path.join(experiment_root, dataset_name)
    os.makedirs(experiment, exist_ok=True)

    for mode in modes:
        '''
        train_dataset = TemporalWalkSupervisionDataset(
            train_data,
            graph_train,
            sampler_train,
            num_nodes=NUM_NODES,
            supervision_mode=mode,
        )
        '''
        trajectory_index = ObservedTrajectoryIndex(
            sources=train_data.sources,
            destinations=train_data.destinations,
            timestamps=train_data.timestamps,
            max_horizon=None,
            max_continuations=20,
        )
        
        train_dataset = TemporalWalkSupervisionDataDriven(
            edge_dataset=train_data,
            graph=graph_train,
            sampler=sampler_train,
            num_nodes=NUM_NODES,
            trajectory_index=trajectory_index,
            supervision_mode="observed_soft",
            smoothing=1e-3,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_temporal_walk_link,
        )

        for lambda_walk, lambda_link in zip(lambda_walks, lambda_links):
            auc_scores = []
            ap_scores = []

            print(f"Model params:")
            print(f"\tlambda_walk: {lambda_walk}, lambda_link: {lambda_link}")
            print(f"\tsupervision_mode: {mode}")

            for run_idx in range(n_runs):
                set_seed(2020 + run_idx)
                print(f"\nRun {run_idx + 1}/{n_runs}...")

                model = TemporalWalkEncoder(
                    num_nodes=NUM_NODES,
                    embedding_dim=32,
                    time_dim=16,
                    pad_node=PAD_NODE,
                    debug=False,
                ).to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
                link_criterion = nn.BCEWithLogitsLoss()

                path_plot = os.path.join(
                    experiment,
                    f"gradient_cosine_walk_{mode}_{lambda_walk}_link_{lambda_link}_run_{run_idx}.png",
                )

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
                    track_gradients=False,
                    gradient_plot_path=path_plot,
                    gradient_every=10,
                    temperature=1.2,
                    label_smoothing=0.05,
                    debug=False,
                )
                metrics = evaluate_link_prediction(model, test_loader, device)
                auc_scores.append(metrics["auc_roc"])
                ap_scores.append(metrics["ap"])

            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores, ddof=1)
            mean_ap = np.mean(ap_scores)
            std_ap = np.std(ap_scores, ddof=1)

            print(f"Summary for {dataset_name} | {mode} | lambda_walk={lambda_walk} | lambda_link={lambda_link}")
            print(f"  AUC mean ± std: {mean_auc:.4f} ± {std_auc:.4f}")
            print(f"  AP  mean ± std: {mean_ap:.4f} ± {std_ap:.4f}")

