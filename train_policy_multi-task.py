import argparse
import os
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from model.atlas import TemporalWalkEncoder
from utils.data_processing import EdgeDataset, TemporalWalkLinkDataset, TemporalWalkSupervisionDataset, collate_temporal_walk_link
from utils.graph import GraphStorage
from utils.sampler import TemporalNeighborSampler


if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
    print("Device cuda not available, using cpu")

DEVICE = torch.device(dev)
PATH_DATASET = "/exp-local/steve/datasets/temporal/ml_preprocess/"


def set_seed(seed: int = 2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a temporal walk policy model")
    parser.add_argument("--dataset", type=str, default="wikipedia")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-neighbors", type=int, default=30)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--time-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-walk", type=float, default=0.3)
    parser.add_argument("--lambda-link", type=float, default=0.7)
    parser.add_argument("--supervision-mode", type=str, default="soft", choices=["earliest", "sampled", "soft"])
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--save-path", type=str, default="bin/temporal_walk_policy_model.pt", help="Path to save the trained model checkpoint")
    return parser


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
    temperature=1.0,
    label_smoothing=0.05,
    debug=False,
):
    assert supervision_mode in {"earliest", "sampled", "soft"}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_walk = 0.0
        total_link = 0.0

        for batch in train_loader:
            previous_nodes = batch["src"].long().to(device)
            current_nodes = batch["dst"].long().to(device)
            neg_dst = batch["neg_dst"].long().to(device)

            pos_neighbors = batch["pos_neighbors"].long().to(device)
            pos_deg_current = batch["pos_deg_current"].long().to(device)
            pos_deg_neighbors = batch["pos_deg_neighbors"].long().to(device)
            pos_delta_t = batch["pos_delta_t"].float().to(device)
            pos_mask = batch["pos_mask"].bool().to(device)

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

            if supervision_mode in {"earliest", "sampled"}:
                target_idx = batch["target_idx"].long().to(device)
                walk_loss = F.cross_entropy(walk_logits / temperature, target_idx, label_smoothing=label_smoothing)
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
            else:
                raise ValueError(f"Unsupported supervision mode: {supervision_mode}")

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
            link_loss = link_criterion(pos_score, pos_labels) + link_criterion(neg_score, neg_labels)
            loss = lambda_walk * walk_loss + lambda_link * link_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_walk += walk_loss.item()
            total_link += link_loss.item()

        n = max(len(train_loader), 1)
        print(
            f"Epoch {epoch:03d} "
            f"| Total {total_loss / n:.4f} "
            f"| Walk {total_walk / n:.4f} "
            f"| Link {total_link / n:.4f}"
        )


def main():
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    data_file = os.path.join(PATH_DATASET, f"ml_{args.dataset}.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    graph_df = pd.read_csv(data_file)
    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))
    train_mask = timestamps <= val_time
    test_mask = timestamps > test_time

    train_data = EdgeDataset(
        sources[train_mask],
        destinations[train_mask],
        timestamps[train_mask],
        edge_idxs[train_mask],
        labels[train_mask],
    )

    graph_train = GraphStorage(sources[train_mask], destinations[train_mask], timestamps[train_mask])
    graph_eval = GraphStorage(sources[test_mask], destinations[test_mask], timestamps[test_mask])

    all_nodes = set(sources) | set(destinations)
    max_node_id = max(all_nodes) if all_nodes else 0
    num_nodes = int(max_node_id) + 1
    pad_node = num_nodes

    sampler_train = TemporalNeighborSampler(graph_train, num_neighbors=args.num_neighbors, pad_node=pad_node)
    sampler_eval = TemporalNeighborSampler(graph_eval, num_neighbors=args.num_neighbors, pad_node=pad_node)

    train_dataset = TemporalWalkSupervisionDataset(
        train_data,
        graph_train,
        sampler_train,
        num_nodes=num_nodes,
        supervision_mode=args.supervision_mode,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_temporal_walk_link,
    )

    model = TemporalWalkEncoder(
        num_nodes=num_nodes,
        embedding_dim=args.embedding_dim,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
        pad_node=pad_node,
        debug=False,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    link_criterion = nn.BCEWithLogitsLoss()

    train_walk_policy(
        model,
        train_loader,
        device=DEVICE,
        optimizer=optimizer,
        link_criterion=link_criterion,
        epochs=args.epochs,
        lambda_walk=args.lambda_walk,
        lambda_link=args.lambda_link,
        supervision_mode=args.supervision_mode,
        temperature=args.temperature,
        label_smoothing=args.label_smoothing,
    )

    if args.save_path is not None:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(model.state_dict(), args.save_path)
        print(f"Saved model to {args.save_path}")


if __name__ == "__main__":
    main()
