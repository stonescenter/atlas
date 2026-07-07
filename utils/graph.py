import numpy as np
import pandas as pd
from collections import defaultdict

class GraphStorage(object):

    def __init__(
        self,
        sources,
        destinations,
        timestamps,
    ):

        self.sources = np.asarray(sources)
        self.destinations = np.asarray(destinations)
        self.timestamps = np.asarray(timestamps)

        n = len(sources)

        self.n_interactions = n

        self.unique_nodes = (
            set(sources) | set(destinations)
        )

        self.n_unique_nodes = len(self.unique_nodes)

        self.adj_list = defaultdict(list)

        for src, dst, t in zip(
            self.sources,
            self.destinations,
            self.timestamps,
        ):

            self.adj_list[src].append((dst, t))
            self.adj_list[dst].append((src, t))

        '''
        for node in self.adj_list:
            self.adj_list[node].sort(
                key=lambda x: x[1]
            )
        '''
        self.mapping_degrees = self._node_degrees()
        
        eps = 1e-10
        self.inv_sqrt_degree = {
            n: 1 / np.sqrt(deg + eps) for n, deg in self.mapping_degrees.items()
        }

    def get_nodes(self):
        return self.unique_nodes
    
    def num_nodes(self):
        return self.n_unique_nodes

    def get_neighbors(
        self,
        node_id,
        timestamp=None,
        is_fordward=True,
        undirected=True,
        unique=True,
    ):
        # Temporal filtering
        if timestamp is not None:
            if is_fordward:
                temporal_mask = self.timestamps > timestamp
            else:
                temporal_mask = self.timestamps < timestamp
        else:
            temporal_mask = np.ones(
                self.n_interactions,
                dtype=bool
            )

        src = self.sources[temporal_mask]
        dst = self.destinations[temporal_mask]

        # Outgoing neighbors
        out_neighbors = dst[src == node_id]

        if undirected:
            # Incoming neighbors
            in_neighbors = src[dst == node_id]

            neighbors = np.concatenate(
                [out_neighbors, in_neighbors]
            )
        else:
            neighbors = out_neighbors

        if unique:
            neighbors = np.unique(neighbors)

        return neighbors
    
    def edge_exists_before(self, src, dst, time):
        """
        Check whether an edge between src and dst exists with timestamp < time.

        Args:
            src: Source node.
            dst: Destination node.
            time: Time threshold (exclusive).

        Returns:
            True if the edge exists before the given time, otherwise False.
        """
        for neighbor, edge_time in self.adj_list.get(src, []):
            if neighbor == dst and edge_time < time:
                return True
        return False

    def get_neighbors_array(self, node_id, include_edge_weight=False):

        neighbors = []
        times = []
        
        for neighbor, time in self.adj_list[node_id]:
            neighbors.append(neighbor)
            times.append(time)

        outputs = [
            np.asarray(neighbors),
            np.asarray(times)]
        
        return outputs if include_edge_weight else np.asarray(neighbors)
        
    def _get_degree(
        self,
        node_id,
        timestamp=None,
        undirected=True,
        unique=True,
    ):
        neighbors = self.get_neighbors(
            node_id=node_id,
            timestamp=timestamp,
            undirected=undirected,
            unique=unique,
        )

        return len(neighbors)
    
    def get_degree(self, node_id):
        if node_id in self.mapping_degrees:
            return self.mapping_degrees[node_id]
        else:
            #return 0
            return 1 # previende explosoes numericas com 0
        
    def _node_degrees(self):
        mapping = {node: self._get_degree(node) for node in self.get_nodes()}
        return mapping
    
    def get_degree_neighbors(self, neighbors, time=None):
        '''
            retorna o grau de uma lista de nos
        '''
        result = []
        if time is None:
            result = [self.get_degree(node_id=n) for n in neighbors]
        else:
            result = [self._get_degree(node_id=n, timestamp=time) for n in neighbors]

        return result