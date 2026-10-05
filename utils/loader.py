import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from dataclasses import asdict, dataclass

from .data_processing import EdgeDataset
from .graph import GraphStorage
from .sampler import TemporalNeighborSampler
from .data_processing import TemporalWalkSupervisionDataset
from .data_processing import collate_temporal_walk_link

class TemporalLinkNeighborLoader:

    def __init__(
        self,
        data,
        edge_label_index,
        edge_label_time,
        edge_label=None,
        num_neighbors=10,
        batch_size=32,
        shuffle=True,
    ):

        self.data = data

        self.edge_label_index = edge_label_index
        self.edge_label_time = edge_label_time
        self.edge_label = edge_label

        self.num_neighbors = num_neighbors

        self.batch_size = batch_size

        self.loader = DataLoader(
            range(edge_label_index.size(1)),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def sample_neighbors(self, node_id, query_time):

        edge_index = self.data.edge_index
        #edge_time = self.data.edge_time
        edge_time = self.data.edge_attr


        # Temporal constraint
        mask = edge_time < query_time

        temporal_edges = edge_index[:, mask]

        src, dst = temporal_edges

        # Undirected neighborhood
        node_mask = (src == node_id) | (dst == node_id)

        neighbors = torch.cat([
            dst[src == node_id],
            src[dst == node_id]
        ])

        neighbors = torch.unique(neighbors)

        # Random sample
        if neighbors.numel() > self.num_neighbors:
            perm = torch.randperm(neighbors.numel())
            neighbors = neighbors[
                perm[:self.num_neighbors]
            ]

        return neighbors

    def __iter__(self):

        for batch_ids in self.loader:

            batch_edges = self.edge_label_index[:, batch_ids]
            batch_times = self.edge_label_time[batch_ids]

            batch_labels = None

            if self.edge_label is not None:
                batch_labels = self.edge_label[batch_ids]

            sampled_neighbors = []

            for i in range(batch_edges.size(1)):

                u = batch_edges[0, i]
                v = batch_edges[1, i]
                t = batch_times[i]

                neigh_u = self.sample_neighbors(u, t)
                neigh_v = self.sample_neighbors(v, t)

                sampled_neighbors.append(
                    {
                        "u": u,
                        "v": v,
                        "t": t,
                        "u_neighbors": neigh_u,
                        "v_neighbors": neigh_v,
                    }
                )

            yield {
                "edge_label_index": batch_edges,
                "edge_label_time": batch_times,
                "edge_label": batch_labels,
                "samples": sampled_neighbors,
            } 

# ---------------------------------------------------------------------------
# Dataset construction shared by both experiments
# ---------------------------------------------------------------------------

@dataclass
class AtlasLoaders:
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    num_nodes: int
    pad_node: int

def load_loaders(
    path_file: str,
    dataset_name: str,
    batch_size: int = 128,
    num_neighbors: int = 30,
    supervision_mode: str = "soft",
    split_masks: str = "expanded",
    num_workers: int = 4,
    testing_mode: bool = False,
    beta = 0.001
) -> AtlasLoaders:
    """Build chronological train/validation/test Atlas loaders.

    Graphs are cumulative: validation uses all interactions through the
    validation cutoff; test uses all interactions through the test cutoff.
    This avoids constructing evaluation neighborhoods from future events.
    """
    
    graph_df = pd.read_csv('{}/ml_{}.csv'.format(path_file, dataset_name))
    #edge_features = np.load('{}/ml_{}.npy'.format(path_file, dataset_name))
    #node_features = np.load('{}/ml_{}_node.npy'.format(path_file, dataset_name)) 
        
    required = {"u", "i", "ts", "idx", "label"}
    missing = required.difference(graph_df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
   
    # source = frame["u"].to_numpy()
    # destination = frame["i"].to_numpy()
    # timestamp = frame["ts"].to_numpy()
    # edge_index = frame["idx"].to_numpy()
    # label = frame["label"].to_numpy()

    if testing_mode:
        graph_df = graph_df.head(10000)
        print("Testing mode: using only first 10000 edges for quick testing.")

    source = graph_df.u.values
    destination = graph_df.i.values
    edge_index = graph_df.idx.values
    label = graph_df.label.values
    timestamp = graph_df.ts.values

    validation_time, test_time = np.quantile(timestamp, [0.70, 0.85])

    train_mask = timestamp <= validation_time
    validation_mask = (timestamp > validation_time) & (timestamp <= test_time)
    test_mask = timestamp > test_time

    train_edges = EdgeDataset(
        source[train_mask], destination[train_mask], 
        timestamp[train_mask], edge_index[train_mask], 
        label[train_mask]
    )

    validation_edges = EdgeDataset(
        source[validation_mask], destination[validation_mask],
        timestamp[validation_mask], edge_index[validation_mask],
        label[validation_mask]
    )

    test_edges = EdgeDataset(
        source[test_mask], destination[test_mask], timestamp[test_mask],
        edge_index[test_mask], label[test_mask]
    )

    all_nodes = set(source.tolist()) | set(destination.tolist())
    num_nodes = int(max(all_nodes)) + 1 if all_nodes else 1
    pad_node = num_nodes

    graph_train = GraphStorage(
        source[train_mask],
        destination[train_mask],
        timestamp[train_mask],
        edge_idxs=edge_index[train_mask]
    )

    if split_masks == "expanded":
        graph_validation = GraphStorage(
            source[timestamp <= test_time],
            destination[timestamp <= test_time],
            timestamp[timestamp <= test_time],
            edge_idxs=edge_index[timestamp <= test_time]
        )

        graph_test = GraphStorage(source, destination, timestamp, edge_index)

    elif split_masks == "isolated":
    
        graph_validation = GraphStorage(
            source[validation_mask],
            destination[validation_mask],
            timestamp[validation_mask],
            edge_idxs=edge_index[validation_mask]
        )

        graph_test = GraphStorage(
            source[test_mask],
            destination[test_mask],
            timestamp[test_mask],
            edge_idxs=edge_index[test_mask]
        )
        
    else:
        print("not strategy for masking")

    train_sampler = TemporalNeighborSampler(
        graph_train, num_neighbors=num_neighbors, pad_node=pad_node
    )

    validation_sampler = TemporalNeighborSampler(
        graph_validation, num_neighbors=num_neighbors, pad_node=pad_node
    )

    test_sampler = TemporalNeighborSampler(
        graph_test, num_neighbors=num_neighbors, pad_node=pad_node
    )

    train_dataset = TemporalWalkSupervisionDataset(
        train_edges,
        graph_train,
        train_sampler,
        num_nodes=num_nodes,
        supervision_mode=supervision_mode,
        beta=beta,
        is_fordward=True # future neighbors
    )

    validation_dataset = TemporalWalkSupervisionDataset(
        validation_edges,
        graph_validation,
        validation_sampler,
        num_nodes=num_nodes,
        supervision_mode=supervision_mode,
        beta=beta,
        is_fordward=True
    )

    test_dataset = TemporalWalkSupervisionDataset(
        test_edges,
        graph_test,
        test_sampler,
        num_nodes=num_nodes,
        supervision_mode=supervision_mode,
        beta=beta,
        is_fordward=False
    )

    loader_arguments = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_temporal_walk_link,
    )

    return AtlasLoaders(
        train_loader=DataLoader(
            train_dataset, shuffle=True, **loader_arguments
        ),
        validation_loader=DataLoader(
            validation_dataset, shuffle=False, **loader_arguments
        ),
        test_loader=DataLoader(
            test_dataset, shuffle=False, **loader_arguments
        ),
        num_nodes=num_nodes,
        pad_node=pad_node,
    )

