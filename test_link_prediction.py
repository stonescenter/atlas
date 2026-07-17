import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from model.atlas import TemporalWalkEncoder
from utils.data_processing import EdgeDataset, TemporalWalkLinkDataset, collate_temporal_walk_link
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
    parser = argparse.ArgumentParser(description="Evaluate a trained temporal walk policy model")
    parser.add_argument("--dataset", type=str, default="wikipedia")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-neighbors", type=int, default=30)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--time-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--seed", type=int, default=2020)
    return parser


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

        _, pos_score = model(src, dst, pos_neighbors, pos_deg_current, pos_deg_neighbors, pos_delta_t, pos_mask)
        _, neg_score = model(src, neg_dst, neg_neighbors, neg_deg_current, neg_deg_neighbors, neg_delta_t, neg_mask)

        scores = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)
        scores = torch.sigmoid(scores)

        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())

    if not all_scores or not all_labels:
        return {"auc_roc": float("nan"), "ap": float("nan")}

    y_score = torch.cat(all_scores).numpy()
    y_true = torch.cat(all_labels).numpy()
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    print(f"Test AUC-ROC: {auc:.4f} | AP: {ap:.4f}")
    return {"auc_roc": auc, "ap": ap}


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
    test_mask = timestamps > test_time

    test_data = EdgeDataset(
        sources[test_mask],
        destinations[test_mask],
        timestamps[test_mask],
        edge_idxs[test_mask],
        labels[test_mask],
    )

    graph_eval = GraphStorage(sources[test_mask], destinations[test_mask], timestamps[test_mask])
    all_nodes = set(sources) | set(destinations)
    max_node_id = max(all_nodes) if all_nodes else 0
    num_nodes = int(max_node_id) + 1
    pad_node = num_nodes

    sampler_eval = TemporalNeighborSampler(graph_eval, num_neighbors=args.num_neighbors, pad_node=pad_node)
    test_dataset = TemporalWalkLinkDataset(test_data, graph_eval, sampler_eval, num_nodes=num_nodes)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_temporal_walk_link)

    model = TemporalWalkEncoder(
        num_nodes=num_nodes,
        embedding_dim=args.embedding_dim,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
        pad_node=pad_node,
        debug=False,
    ).to(DEVICE)
    state = torch.load(args.model, map_location=DEVICE)
    if isinstance(state, dict) and any(k.endswith(".weight") or k.endswith(".bias") for k in state.keys()):
        model.load_state_dict(state)
    else:
        model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    evaluate_link_prediction(model, test_loader, DEVICE)


if __name__ == "__main__":
    main()
