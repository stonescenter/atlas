import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import DataLoader


from utils.graph import GraphStorage
from utils.sampler import TemporalNeighborSampler
from utils.data_processing import EdgeDataset
from model.time_encoder import TimeEncoder 
from utils.data_processing import get_data, Graph, TemporalWalkDataset, collate_temporal_walk

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

device = 'cpu'

print("Device available ", device)



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
graph_df = graph_df.head(1000)
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

def test():

    for batch in train_loader:

        previous_nodes = batch["src"].to(device)

        current_nodes = batch["dst"].to(device)

        deg_current = batch["deg_current"].float().to(device)

        neighbor_nodes = batch["neighbors"].long().to(device)

        deg_neighbors = batch["deg_neighbors"].float().to(device)

        delta_t = batch["delta_t"].float().to(device)

        mask = batch["mask"].bool().to(device)

        target_idx = batch["target_idx"].long().to(device)

        scores = model(
            previous_nodes,
            current_nodes,
            neighbor_nodes,
            deg_current,
            deg_neighbors,
            delta_t
        )

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


for batch in train_loader:

    previous_nodes = batch["src"].to(device)

    current_nodes = batch["dst"].to(device)

    deg_current = batch["deg_current"].long().to(device)

    neighbor_nodes = batch["neighbors"].long().to(device)

    deg_neighbors = batch["deg_neighbors"].long().to(device)

    delta_t = batch["delta_t"].float().to(device)

    mask = batch["mask"].bool().to(device)

    target_idx = batch["target_idx"].long().to(device)

    print(previous_nodes)