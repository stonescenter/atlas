import os
import numpy as np
import pandas as pd
import random

import math
from typing import  Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader
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
from model.time_encoder import TimeEncoder, build_time_encoder
from utils.data_processing import get_data, Graph, TemporalWalkSupervisionDataset, TemporalWalkDataset, TemporalWalkLinkDataset
from utils.data_processing import collate_temporal_walk, collate_temporal_walk_link
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.plots import *
from utils.util import gradient_direction_stats
from utils.md5 import *
from model.atlas import TemporalWalkEncoder


from utils.loader import load_loaders, AtlasLoaders

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
    val_loader,
    device,
    output_directory,
    optimizer,
    link_criterion,
    epochs=20,
    lambda_walk=1.0,  
    lambda_link=1.0,    
    supervision_mode="earliest",
    track_gradients=False,
    gradient_plot_path=None,
    temperature=1.0, 
    label_smoothing=0.05,
    gradient_every=1,
    gradient_clip=1.0,
    debug=False
):
    assert supervision_mode in {
        "earliest",
        "sampled",
        "soft",
    }

    ################################################
    
    output_directory = ensure_directory(output_directory)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    checkpoint_path = output_directory / "atlas_downstream_best.pt"
    best_validation_ap = -float("inf")

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
            optimizer.zero_grad(set_to_none=True)

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
                walk_loss = F.cross_entropy(
                    walk_logits / temperature,
                    target_idx, 
                    label_smoothing=label_smoothing
                )

            # Experiment 3
            elif supervision_mode == "soft":
                # distribution probabilities over neighbors
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

                # q x log(\phi)
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

            # -----------------------------------------
            # Uncertainty-weighted multi-task loss
            # 

            walk_precision = torch.exp(-model.log_var_walk)
            link_precision = torch.exp(-model.log_var_link)

            loss = (
                0.5 * walk_precision * walk_loss +
                0.5 * model.log_var_walk +
                0.5 * link_precision * link_loss +
                0.5 * model.log_var_link
            )

            if track_gradients and global_step % gradient_every == 0:
                grad_stats = gradient_direction_stats(model, walk_loss, link_loss)
                grad_stats["step"] = global_step
                grad_stats["epoch"] = epoch
                gradient_history.append(grad_stats)

            #optimizer.zero_grad()
            loss.backward()

            # avoid exploding gradients by clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
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

        walk_weight = (0.5 * torch.exp(-model.log_var_walk)).item()
        link_weight = (0.5 * torch.exp(-model.log_var_link)).item()
        walk_sigma = torch.exp(0.5 * model.log_var_walk).item()
        link_sigma = torch.exp(0.5 * model.log_var_link).item()

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
            f"| Total {total_loss / n:.4f} "
            f"| Walk {total_walk / n:.4f} "
            f"| Link {total_link / n:.4f} "
            f"| w_walk {walk_weight:.4f} "
            f"| w_link {link_weight:.4f} "
            f"| sigma_walk {walk_sigma:.4f} "
            f"| sigma_link {link_sigma:.4f} "
            f"| AUC {auc:.4f} "
            f"| AP {ap:.4f}"
            f"{grad_msg}"
        )
    
    val_metrics = evaluate_link_prediction(model, val_loader, device)

    scheduler.step(val_metrics["ap"])

    if track_gradients and gradient_plot_path is not None:
        try:
            plot_gradient_directions(gradient_history, gradient_plot_path)
            #print(f"Gradient direction plot saved to {gradient_plot_path}")
        except Exception as exc:
            print(f"Warning: could not save gradient plot to {gradient_plot_path}: {exc}")

    if val_metrics["ap"] > best_validation_ap:
        best_validation_ap = val_metrics["ap"]
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "validation": val_metrics,
                "model_configuration": {
                    "num_nodes": model.embedding.num_embeddings - 1,
                    "embedding_dim": model.embedding.embedding_dim,
                    "time_dim": model.time_encoder.dimension,
                    "hidden_dim": model.walk_head.in_features,
                    "pad_node": model.embedding.padding_idx,
                    "time_encoder_type": time_encoder_type,
                },
            },
            checkpoint_path,
        )

    print(
        "Validation:\n "
        f"| AUC {val_metrics['auc']:.4f} "
        f"| AP {val_metrics['ap']:.4f}"
    )

    return gradient_history, checkpoint_path

@torch.no_grad()
def  evaluate_link_prediction(model, loader, device):
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
        return {"auc": float("nan"), "ap": float("nan")}

    y_score = torch.cat(all_scores).numpy()
    y_true = torch.cat(all_labels).numpy()

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return {
        "auc": float(auc),
        "ap": float(ap),
    }

PATH_DATASET = '/exp-local/steve/datasets/temporal/ml_preprocess/'

random.seed(2020)

batch_size = 64
num_neighbors = 30
embedding_dim = 32 #16 # 32 
time_dim = 16 #8  # 16
hidden_dim = 128 #64 # 128
dropout = 0.1 # 0.2  # 0.1

modes = ["earliest", "sampled", "soft"]
modes = ["soft"]
mode = 'soft'

track_gradients = True
epochs = 10

split_masks = "isolated"
#split_masks = "expanded" 

lambda_walks = [None] # 0.3
lambda_links = [None] # 0.7

learning_rate = 3e-4
time_learning_rate = 1e-4
weight_decay = 1e-4
  
experiment_root = "results/experiments/supervised_loss_improved"
experiment_root = ensure_directory(experiment_root)

optimizer = None
time_encoder_type = "atlas"
testing_mode = False # get 10k observations
beta = 0.001
#beta = -1.0

if testing_mode:
    n_runs = 2
    epochs = 5
    datasets = ['wikipedia']
else:
    n_runs = 5
    epochs = 9
    #datasets = ['wikipedia', 'enron', 'taobao', 'collegemsg', 'mooc', 'reddit']
    #datasets = ['enron', 'collegemsg', 'mooc', 'reddit']
    datasets = ['wikipedia']
    datasets = ['mooc']
        
    
for dataset_name in datasets:
    data_file = os.path.join(PATH_DATASET, f"ml_{dataset_name}.csv")
    if not os.path.exists(data_file):
        print(f"Skipping dataset {dataset_name}: file not found at {data_file}")
        continue

    print(f"\n=== Dataset: {dataset_name} ===")
    #load_data(PATH_DATASET, dataset_name)
    graph_df = pd.read_csv('{}/ml_{}.csv'.format(PATH_DATASET, dataset_name))

    loaders = load_loaders(
        path_file=PATH_DATASET,
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        supervision_mode=mode,
        split_masks=split_masks,
        num_workers=4,
        testing_mode=testing_mode,
        beta=beta
    )

    print(
        f"Loader sizes: train={len(loaders.train_loader.dataset)}, "
        f"validation={len(loaders.validation_loader.dataset)}, "
        f"test={len(loaders.test_loader.dataset)}"
    )

    experiment = os.path.join(experiment_root, dataset_name)
    experiment_ = ensure_directory(experiment)

    for lambda_walk, lambda_link in zip(lambda_walks, lambda_links):
        auc_scores = []
        ap_scores = []

        output = f"Model params:\n"
        output = output + f"\tSupervision_mode: {mode}, beta: {beta}, lambda_walk: {lambda_walk}, lambda_link: {lambda_link}, split_masks: {split_masks}\n"  
        output = output + f"\tModel dim: {embedding_dim}, time_dim: {time_dim}, hidden_dim: {hidden_dim}, dropout: {dropout}\n"            
        output = output + f"\tConfig model epochs: {epochs}, batch_size: {batch_size}, K neighbors: {num_neighbors}\n"
        
        id_md5 = generate_md5_id(output)
        experiment = os.path.join(experiment_, id_md5)
        if os.path.isdir(experiment):
            print(f"Experiment with same parameters already exists: {id_md5}. Skipping...")
            continue

        experiment = ensure_directory(experiment)

        output = output + f"\tDate: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        output = output + f"\tSaved: {experiment}"
        
        print(output)

        with open(f"{experiment}/parameters.txt", "w") as file:
            file.write(output)

        for run_idx in range(n_runs):
            set_seed(2020 + run_idx)
            print(f"\nRun {run_idx + 1}/{n_runs}...")

            time_encoder = build_time_encoder(time_encoder_type, time_dim)

            model = TemporalWalkEncoder(
                num_nodes=loaders.num_nodes,
                embedding_dim=embedding_dim,
                time_dim=time_dim,
                pad_node=loaders.pad_node,
                debug=False,
                time_encoder=time_encoder
            ).to(device)

            time_parameter_ids = {id(p) for p in model.time_encoder.parameters()}
            model_parameters = [
                parameter for parameter in model.parameters()
                if id(parameter) not in time_parameter_ids
            ]

            optimizer = torch.optim.AdamW([
                {
                    "params": model.time_encoder.parameters(),
                    "lr": time_learning_rate
                },
                {
                    "params": model_parameters,
                    "lr": learning_rate
                }
                ],
                weight_decay=weight_decay,
            )

            link_criterion = nn.BCEWithLogitsLoss()

            path_plot = os.path.join(
                experiment / "training",
                f"gradient_cosine_walk_{mode}_{lambda_walk}_link_{lambda_link}_run_{run_idx}.png",
            )

            train_walk_policy(
                model,
                loaders.train_loader,
                loaders.validation_loader,
                device=device,
                output_directory=experiment / "training",
                optimizer=optimizer,
                link_criterion=link_criterion,
                epochs=epochs,
                lambda_walk=lambda_walk,
                lambda_link=lambda_link,
                supervision_mode=mode,
                track_gradients=track_gradients,
                gradient_plot_path=path_plot,
                gradient_every=10,
                temperature=1.2,
                label_smoothing=0.05,
                debug=False,
            )

            metrics = evaluate_link_prediction(model, loaders.test_loader, device)
                            
            auc_scores.append(metrics["auc"])
            ap_scores.append(metrics["ap"])

        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores, ddof=1)
        mean_ap = np.mean(ap_scores)
        std_ap = np.std(ap_scores, ddof=1)

        print(f"Summary for {dataset_name} | {mode} | lambda_walk={lambda_walk} | lambda_link={lambda_link}")
        print(f"Testing:")
        print(f"  AUC mean ± std: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"  AP  mean ± std: {mean_ap:.4f} ± {std_ap:.4f}")

