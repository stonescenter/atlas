import torch
from torch.utils.data import DataLoader

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