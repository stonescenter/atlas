import random
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GraphStorage(object):
    def __init__(self, sources, destinations, timestamps):
        self.sources = np.asarray(sources)
        self.destinations = np.asarray(destinations)
        self.timestamps = np.asarray(timestamps)

        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)

        self.adj_list = defaultdict(list)
        for src, dst, t in zip(self.sources, self.destinations, self.timestamps):
            self.adj_list[src].append((dst, t))
            self.adj_list[dst].append((src, t))

        self.mapping_degrees = self._node_degrees()
        eps = 1e-10
        self.inv_sqrt_degree = {n: 1 / np.sqrt(deg + eps) for n, deg in self.mapping_degrees.items()}

    def get_nodes(self):
        return self.unique_nodes

    def num_nodes(self):
        return self.n_unique_nodes

    def get_neighbors(self, node_id, timestamp=None, is_fordward=True, undirected=True, unique=True):
        if timestamp is not None:
            temporal_mask = self.timestamps > timestamp if is_fordward else self.timestamps < timestamp
        else:
            temporal_mask = np.ones(self.n_interactions, dtype=bool)

        src = self.sources[temporal_mask]
        dst = self.destinations[temporal_mask]

        out_neighbors = dst[src == node_id]
        if undirected:
            in_neighbors = src[dst == node_id]
            neighbors = np.concatenate([out_neighbors, in_neighbors])
        else:
            neighbors = out_neighbors

        if unique:
            neighbors = np.unique(neighbors)
        return neighbors

    def get_neighbors_array(self, node_id, include_edge_weight=False):
        neighbors = []
        times = []
        for neighbor, time in self.adj_list[node_id]:
            neighbors.append(neighbor)
            times.append(time)
        outputs = [np.asarray(neighbors), np.asarray(times)]
        return outputs if include_edge_weight else np.asarray(neighbors)

    def _get_degree(self, node_id, timestamp=None, undirected=True, unique=True):
        neighbors = self.get_neighbors(node_id=node_id, timestamp=timestamp, undirected=undirected, unique=unique)
        return len(neighbors)

    def get_degree(self, node_id):
        if node_id in self.mapping_degrees:
            return self.mapping_degrees[node_id]
        return 1

    def _node_degrees(self):
        return {node: self._get_degree(node) for node in self.get_nodes()}

    def get_degree_neighbors(self, neighbors, time=None):
        if time is None:
            return [self.get_degree(node_id=n) for n in neighbors]
        return [self._get_degree(node_id=n, timestamp=time) for n in neighbors]


class TemporalNeighborSampler:
    def __init__(self, graph, num_neighbors=10, pad_node=0):
        self.graph = graph
        self.num_neighbors = num_neighbors
        self.pad_node = pad_node

    def sample_k(self, node_id, current_time, is_forward=True):
        neighbors, times = self.graph.get_neighbors_array(node_id, include_edge_weight=True)
        neighbors = np.asarray(neighbors)
        times = np.asarray(times)

        if is_forward:
            mask = times > current_time
        else:
            mask = times < current_time

        neighbors = neighbors[mask]
        times = times[mask]

        order = np.argsort(times) if is_forward else np.argsort(-times)
        neighbors = neighbors[order]
        times = times[order]

        K = self.num_neighbors
        neighbors = neighbors[:K]
        times = times[:K]

        n_valid = len(neighbors)
        valid_mask = np.zeros(K, dtype=np.bool_)
        valid_mask[:n_valid] = True

        padded_neighbors = np.full(K, self.pad_node, dtype=np.int64)
        padded_neighbors[:n_valid] = neighbors
        padded_times = np.zeros(K, dtype=np.float32)
        padded_times[:n_valid] = times

        return padded_neighbors, padded_times, valid_mask, n_valid


def temporal_target_distribution(times, mask, current_time, beta=0.1):
    delta = times - current_time
    delta = np.maximum(delta, 0)
    weights = np.exp(-beta * delta).astype(np.float32)
    weights[~mask] = 0.0
    total = weights.sum()
    if total <= 0:
        return None
    return weights / total


def sample_temporal_target(valid_neighbors, valid_times, current_time, lamb=0.1):
    delta = valid_times - current_time
    weights = np.exp(-lamb * delta)
    probs = weights / weights.sum()
    idx = np.random.choice(len(valid_neighbors), p=probs)
    return valid_neighbors[idx]


class ContextBase(Dataset):
    def __init__(self, graph, sampler):
        self.examples = []
        self.graph = graph
        self.sampler = sampler

    def build_context(self, node, ts):
        neighbors, times, mask, n_valid = self.sampler.sample_k(node_id=node, current_time=ts, is_forward=True)
        if n_valid == 0:
            return None
        deg_current = self.graph.get_degree(node)
        deg_neighbors = self.graph.get_degree_neighbors(neighbors)
        delta_t = times - ts
        delta_t[~mask] = 0
        delta_t = np.maximum(delta_t, 0)
        return {
            "deg_current": deg_current,
            "neighbors": neighbors,
            "deg_neighbors": deg_neighbors,
            "delta_t": delta_t,
            "mask": mask,
            "times": times,
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class TemporalWalkSupervisionDataset(ContextBase):
    def __init__(self, edge_dataset, graph, sampler, num_nodes, supervision_mode="earliest", beta=0.1):
        super().__init__(graph, sampler)
        self.supervision_mode = supervision_mode
        self.beta = beta
        self.num_nodes = num_nodes

        assert supervision_mode in {"earliest", "sampled", "soft"}

        for sample in edge_dataset: 
            src = int(sample["src"])
            dst = int(sample["dst"])
            ts = float(sample["ts"])

            pos_context = self.build_context(node=dst, ts=ts)
            if pos_context is None:
                continue

            neg_dst = self.sample_negative_node(src=src, dst=dst, ts=ts)
            neg_context = self.build_context(node=neg_dst, ts=ts)
            if neg_context is None:
                continue

            example = {
                "src": src,
                "dst": dst,
                "neg_dst": neg_dst,
                "pos_neighbors": pos_context["neighbors"],
                "pos_deg_current": pos_context["deg_current"],
                "pos_deg_neighbors": pos_context["deg_neighbors"],
                "pos_delta_t": pos_context["delta_t"],
                "pos_mask": pos_context["mask"],
                "neg_neighbors": neg_context["neighbors"],
                "neg_deg_current": neg_context["deg_current"],
                "neg_deg_neighbors": neg_context["deg_neighbors"],
                "neg_delta_t": neg_context["delta_t"],
                "neg_mask": neg_context["mask"],
            }

            if supervision_mode == "earliest":
                example["target_idx"] = 0
            elif supervision_mode == "sampled":
                probs = temporal_target_distribution(
                    times=pos_context["times"], 
                    mask=pos_context["mask"], 
                    current_time=ts, 
                    beta=beta
                )
                if probs is None:
                    continue
                example["target_idx"] = int(np.random.choice(len(probs), p=probs))
            elif supervision_mode == "soft":
                probs = temporal_target_distribution(
                    times=pos_context["times"], 
                    mask=pos_context["mask"], 
                    current_time=ts, 
                    beta=beta
                )
                if probs is None:
                    continue
                example["target_probs"] = probs.astype(np.float32)

            self.examples.append(example)

    def sample_negative_node(self, src, dst, ts=None, pool_size=16):
        if self.num_nodes <= 0:
            return int(dst)
        seen = {src, dst}
        if ts is None:
            candidate_nodes = set(self.graph.get_neighbors(dst, undirected=True, unique=True))
        else:
            candidate_nodes = set(self.graph.get_neighbors(dst, timestamp=ts, undirected=True, unique=True))
        candidate_nodes = [node for node in candidate_nodes if node not in seen]
        if len(candidate_nodes) > pool_size:
            candidate_nodes = list(np.random.choice(candidate_nodes, size=pool_size, replace=False))
        if not candidate_nodes:
            return int(dst)

        dst_degree = self.graph.get_degree(dst)
        weights = np.ones(len(candidate_nodes), dtype=np.float32)
        for idx, node in enumerate(candidate_nodes):
            degree_gap = abs(self.graph.get_degree(node) - dst_degree)
            weights[idx] = 1.0 + 1.0 / (1.0 + degree_gap)
        weights = np.maximum(weights, 1e-6)
        probs = weights / weights.sum()
        return int(np.random.choice(candidate_nodes, p=probs))


class TemporalWalkLinkDataset(ContextBase):
    def __init__(self, edge_dataset, graph, sampler, num_nodes):
        super().__init__(graph, sampler)
        self.num_nodes = num_nodes
        for sample in edge_dataset:
            src = int(sample["src"])
            dst = int(sample["dst"])
            ts = float(sample["ts"])
            pos = self.build_context(dst, ts)
            if pos is None:
                continue

            neg_dst = np.random.randint(0, num_nodes)
            tries = 0
            while (neg_dst == dst or neg_dst == src) and tries < 20:
                neg_dst = np.random.randint(0, num_nodes)
                tries += 1

            neg = self.build_context(neg_dst, ts)
            if neg is None:
                continue

            self.examples.append({
                "src": src,
                "dst": dst,
                "neg_dst": neg_dst,
                "pos_deg_current": pos["deg_current"],
                "pos_neighbors": pos["neighbors"],
                "pos_deg_neighbors": pos["deg_neighbors"],
                "pos_delta_t": pos["delta_t"],
                "pos_mask": pos["mask"],
                "neg_deg_current": neg["deg_current"],
                "neg_neighbors": neg["neighbors"],
                "neg_deg_neighbors": neg["deg_neighbors"],
                "neg_delta_t": neg["delta_t"],
                "neg_mask": neg["mask"],
                "target_idx": 0,
            })


class EdgeDataset(Dataset):
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(self.sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return {"src": self.sources[idx], "dst": self.destinations[idx], "ts": self.timestamps[idx], "label": self.labels[idx]}


def collate_temporal_walk_link(batch):
    output = {
        "src": torch.tensor([x["src"] for x in batch], dtype=torch.long),
        "dst": torch.tensor([x["dst"] for x in batch], dtype=torch.long),
        "neg_dst": torch.tensor([x["neg_dst"] for x in batch], dtype=torch.long),
        "pos_deg_current": torch.tensor([x["pos_deg_current"] for x in batch], dtype=torch.long),
        "pos_neighbors": torch.tensor(np.stack([x["pos_neighbors"] for x in batch]), dtype=torch.long),
        "pos_deg_neighbors": torch.tensor(np.stack([x["pos_deg_neighbors"] for x in batch]), dtype=torch.long),
        "pos_delta_t": torch.tensor(np.stack([x["pos_delta_t"] for x in batch]), dtype=torch.float),
        "pos_mask": torch.tensor(np.stack([x["pos_mask"] for x in batch]), dtype=torch.bool),
        "neg_deg_current": torch.tensor([x["neg_deg_current"] for x in batch], dtype=torch.long),
        "neg_neighbors": torch.tensor(np.stack([x["neg_neighbors"] for x in batch]), dtype=torch.long),
        "neg_deg_neighbors": torch.tensor(np.stack([x["neg_deg_neighbors"] for x in batch]), dtype=torch.long),
        "neg_delta_t": torch.tensor(np.stack([x["neg_delta_t"] for x in batch]), dtype=torch.float),
        "neg_mask": torch.tensor(np.stack([x["neg_mask"] for x in batch]), dtype=torch.bool),
    }
    if "target_idx" in batch[0]:
        output["target_idx"] = torch.tensor([x["target_idx"] for x in batch], dtype=torch.long)
    elif "target_probs" in batch[0]:
        output["target_probs"] = torch.tensor(np.stack([x["target_probs"] for x in batch]), dtype=torch.float)
    return output


class TimeEncoder(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        freq = 1 / 10 ** np.linspace(0, 9, dimension)
        self.w.weight = nn.Parameter(torch.from_numpy(freq).float().reshape(dimension, 1))
        self.w.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, t):
        t = torch.log1p(t)
        t = t.unsqueeze(-1)
        x = self.w(t)
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def gradient_direction_stats(model, walk_loss, link_loss):
    params = [p for p in model.parameters() if p.requires_grad]
    walk_grads = torch.autograd.grad(walk_loss, params, retain_graph=True, allow_unused=True)
    link_grads = torch.autograd.grad(link_loss, params, retain_graph=True, allow_unused=True)
    walk_parts = []
    link_parts = []
    for walk_grad, link_grad in zip(walk_grads, link_grads):
        if walk_grad is None or link_grad is None:
            continue
        walk_parts.append(walk_grad.detach().reshape(-1))
        link_parts.append(link_grad.detach().reshape(-1))
    if not walk_parts:
        return {"cosine": float("nan"), "walk_norm": 0.0, "link_norm": 0.0}
    walk_vec = torch.cat(walk_parts)
    link_vec = torch.cat(link_parts)
    cosine = F.cosine_similarity(walk_vec, link_vec, dim=0, eps=1e-12).item()
    return {"cosine": cosine, "walk_norm": walk_vec.norm().item(), "link_norm": link_vec.norm().item()}


def plot_gradient_directions(history, output_path):
    if not history:
        return
    steps = [row["step"] for row in history]
    cosines = [row["cosine"] for row in history]
    plt.figure(figsize=(10, 4))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.plot(steps, cosines, linewidth=1.5)
    plt.scatter(steps, cosines, s=10)
    plt.ylim(-1.05, 1.05)
    plt.xlabel("Training step")
    plt.ylabel("cos(grad walk_loss, grad link_loss)")
    plt.title("Gradient Direction Agreement")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


class TemporalWalkModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64, time_dim=32, hidden_dim=128, pad_node=None, dropout=0.1, debug=False):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes + 1, embedding_dim, padding_idx=pad_node)
        self.time_encoder = TimeEncoder(time_dim)
        temporal_dim = 2 * time_dim
        input_dim = 1 + temporal_dim + embedding_dim + embedding_dim + 3 * embedding_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.walk_head = nn.Linear(hidden_dim, 1)
        self.pool_attn = nn.Linear(hidden_dim, 1)
        self.link_head = nn.Sequential(nn.Linear(hidden_dim + 2 * embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.debug = debug

    def structural_score(self, deg_current, deg_neighbors):
        deg_current = deg_current.unsqueeze(1)
        deg_current = torch.clamp(deg_current, min=1.0)
        deg_neighbors = torch.clamp(deg_neighbors, min=1.0)
        score = 1.0 / torch.sqrt(deg_current * deg_neighbors + 1e-8)
        return torch.log(score + 1e-8).unsqueeze(-1)

    def encode_walks(self, previous_nodes, current_nodes, neighbor_nodes, deg_current, deg_neighbors, delta_t, mask=None):
        B, K = neighbor_nodes.shape
        h_prev = self.embedding(previous_nodes)
        h_curr = self.embedding(current_nodes)
        h_next = self.embedding(neighbor_nodes)
        h_prev_exp = h_prev.unsqueeze(1).expand(-1, K, -1)
        h_curr_exp = h_curr.unsqueeze(1).expand(-1, K, -1)
        delta_t = torch.clamp(delta_t, min=0.0)
        structural = self.structural_score(deg_current, deg_neighbors)
        temporal = self.time_encoder(delta_t)
        prev_next = h_prev_exp * h_next
        curr_next = h_curr_exp * h_next
        x = torch.cat([structural, temporal, prev_next, curr_next, h_prev_exp, h_curr_exp, h_next], dim=-1)
        z_walk = self.encoder(x)
        if mask is not None:
            z_walk = z_walk.masked_fill(~mask.unsqueeze(-1), 0.0)
        return z_walk, h_prev, h_curr

    def pool_walks(self, z_walk, mask):
        attn_logits = self.pool_attn(z_walk).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn = torch.softmax(attn_logits, dim=-1)
        z_pool = torch.sum(z_walk * attn.unsqueeze(-1), dim=1)
        return z_pool

    def forward(self, previous_nodes, current_nodes, neighbor_nodes, deg_current, deg_neighbors, delta_t, mask):
        z_walk, h_prev, h_curr = self.encode_walks(previous_nodes, current_nodes, neighbor_nodes, deg_current, deg_neighbors, delta_t, mask)
        walk_logits = self.walk_head(z_walk).squeeze(-1)
        z_pool = self.pool_walks(z_walk, mask)
        edge_repr = torch.cat([z_pool, h_prev, h_curr], dim=-1)
        edge_score = self.link_head(edge_repr).squeeze(-1)
        return walk_logits, edge_score


def train_walk_policy(
        model, train_loader, device,
        optimizer, link_criterion, epochs=20, 
        lambda_walk=0.5, lambda_link=0.5, 
        supervision_mode="earliest", 
        track_gradients=False, 
        gradient_plot_path=None, 
        gradient_every=1, 
        temperature=1.0, 
        label_smoothing=0.05, debug=False):
    assert supervision_mode in {"earliest", "sampled", "soft"}
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

            walk_logits, pos_score = model(previous_nodes, current_nodes, pos_neighbors, pos_deg_current, pos_deg_neighbors, pos_delta_t, pos_mask)
            walk_logits = walk_logits.masked_fill(~pos_mask, -1e9)

            if supervision_mode in {"earliest", "sampled"}:
                target_idx = batch["target_idx"].long().to(device)
                walk_loss = F.cross_entropy(walk_logits / temperature, target_idx, label_smoothing=label_smoothing)
            elif supervision_mode == "soft":
                target_probs = batch["target_probs"].float().to(device)
                log_probs = F.log_softmax(walk_logits / temperature, dim=-1)
                walk_loss = F.kl_div(log_probs, target_probs, reduction="batchmean")

            _, neg_score = model(previous_nodes, neg_dst, neg_neighbors, neg_deg_current, neg_deg_neighbors, neg_delta_t, neg_mask)
            pos_labels = torch.ones_like(pos_score)
            neg_labels = torch.zeros_like(neg_score)
            link_loss = link_criterion(pos_score, pos_labels) + link_criterion(neg_score, neg_labels)
            loss = lambda_walk * walk_loss + lambda_link * link_loss

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
            epoch_cosines = [row["cosine"] for row in gradient_history if row["epoch"] == epoch and not np.isnan(row["cosine"])]
            if epoch_cosines:
                grad_msg = f" | GradCos {np.mean(epoch_cosines):.4f}"
        print(f"Epoch {epoch:03d} | Total {total_loss/n:.4f} | Walk {total_walk/n:.4f} | Link {total_link/n:.4f} | AUC {auc:.4f} | AP {ap:.4f}{grad_msg}")

    if track_gradients and gradient_plot_path is not None:
        plot_gradient_directions(gradient_history, gradient_plot_path)
        print(f"Gradient direction plot saved to {gradient_plot_path}")
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

        _, pos_score = model(src, dst, pos_neighbors, pos_deg_current, pos_deg_neighbors, pos_delta_t, pos_mask)
        _, neg_score = model(src, neg_dst, neg_neighbors, neg_deg_current, neg_deg_neighbors, neg_delta_t, neg_mask)
        scores = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)
        scores = torch.sigmoid(scores)
        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())

    y_score = torch.cat(all_scores).numpy()
    y_true = torch.cat(all_labels).numpy()
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    print(f"Test AUC-ROC: {auc:.4f} | AP: {ap:.4f}")
    return {"auc_roc": auc, "ap": ap}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device available", device)

    PATH_DATASET = "/exp-local/steve/datasets/temporal/ml_preprocess/"
    dataset_name = "wikipedia"

    graph_df = pd.read_csv(f"{PATH_DATASET}/ml_{dataset_name}.csv")
    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    random.seed(2020)

    batch_size = 64
    num_neighbors = 30
    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    train_mask = timestamps <= test_time
    test_mask = timestamps > test_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)

    train_data = EdgeDataset(sources[train_mask], destinations[train_mask], timestamps[train_mask], edge_idxs[train_mask], labels[train_mask])
    val_data = EdgeDataset(sources[val_mask], destinations[val_mask], timestamps[val_mask], edge_idxs[val_mask], labels[val_mask])
    test_data = EdgeDataset(sources[test_mask], destinations[test_mask], timestamps[test_mask], edge_idxs[test_mask], labels[test_mask])

    graph = GraphStorage(sources, destinations, timestamps)
    PAD_NODE = graph.num_nodes()
    max_node_id = max(graph.get_nodes())
    NUM_NODES = graph.num_nodes()
    if max_node_id < NUM_NODES:
        NUM_NODES = max_node_id + 1
    PAD_NODE = NUM_NODES

    sampler = TemporalNeighborSampler(graph, num_neighbors=num_neighbors, pad_node=PAD_NODE)

    test_walks = TemporalWalkLinkDataset(test_data, graph, sampler, num_nodes=NUM_NODES)
    test_loader = DataLoader(test_walks, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_temporal_walk_link)

    epochs = 10
    lambda_walks = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    lambda_links = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]
    modes = ["earliest", "sampled", "soft"]

    optimizer = None
    link_criterion = nn.BCEWithLogitsLoss()

    for mode in modes:
        train_dataset = TemporalWalkSupervisionDataset(train_data, graph, sampler, num_nodes=NUM_NODES, supervision_mode=mode)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_temporal_walk_link)

        model = TemporalWalkModel(num_nodes=NUM_NODES, embedding_dim=32, time_dim=16, pad_node=PAD_NODE, debug=False).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

        for lambda_walk, lambda_link in zip(lambda_walks, lambda_links):
            print(f"Model params:")
            print(f"\tlambda_walk: {lambda_walk}, lambda_link: {lambda_link}")
            print(f"\tsupervision_mode: {mode}")
            print("Training...")
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
                gradient_plot_path=f"gradient_cosine_walk_exp2_{mode}_{lambda_walk}_link_{lambda_link}.png",
                gradient_every=10,
                temperature=1.2,
                label_smoothing=0.05,
                debug=False,
            )
            evaluate_link_prediction(model, test_loader, device)
