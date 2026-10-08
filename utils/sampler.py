import numpy as np

class TemporalNeighborSampler:

    def __init__(
        self,
        graph,
        num_neighbors=10, # fixed neighbors
        pad_node=0
    ):

        self.graph = graph
        self.num_neighbors = num_neighbors
        self.pad_node = pad_node
    '''
    def sample(
        self,
        node_id,
        current_time,
    ):

        neighbors = []
        times = []

        for neigh, ts in self.graph.adj_list[node_id]:
            if ts > current_time:
                neighbors.append(neigh)
                times.append(ts)

        neighbors = np.asarray(neighbors)
        times = np.asarray(times)

        return neighbors, times
    '''

    def sample(self, node_id, current_time, is_forward=True):
        
      neighbors, times = self.graph.get_neighbors_array(node_id, include_edge_weight=True)
      
      if is_forward:
        mask = times > current_time # future neighbors
      else:
        mask = times < current_time # future neighbors
      
      neighbors = np.asarray(neighbors)[mask]
      times = np.asarray(times)[mask]

      if len(times) == 0:
          return np.array([]), np.array([])

      if is_forward:
          order = np.argsort(times)
      else:
          order = np.argsort(-times)

      neighbors = neighbors[order]
      times = times[order]

      K = self.num_neighbors

      neighbors = neighbors[:K]
      times = times[:K]

      return neighbors, times

    def sample_k(self, node_id, current_time, is_forward=True, return_edge_idxs=False):
        '''
            Returns the K nearest neighbors of a node in the temporal graph,
            along with their timestamps > , < current_time and a validity mask.
        '''
        if return_edge_idxs:
            neighbors, times, edge_idxs = self.graph.get_neighbors_array(
                node_id,
                include_edge_weight=True,
                include_edge_index=True,
            )
        else:
            neighbors, times = self.graph.get_neighbors_array(
                node_id,
                include_edge_weight=True,
            )

        neighbors = np.asarray(neighbors)
        times = np.asarray(times)

        if is_forward:
            mask = times > current_time
        else:
            mask = times < current_time

        neighbors = neighbors[mask]
        times = times[mask]
        if return_edge_idxs:
            edge_idxs = np.asarray(edge_idxs)[mask]

        # sort by temporal distance
        if is_forward:
            order = np.argsort(times)
        else:
            order = np.argsort(-times)

        neighbors = neighbors[order]
        times = times[order]
        if return_edge_idxs:
            edge_idxs = edge_idxs[order]

        K = self.num_neighbors

        # keep first K
        neighbors = neighbors[:K]
        times = times[:K]
        if return_edge_idxs:
            edge_idxs = edge_idxs[:K]

        n_valid = len(neighbors)

        # Build validity mask
        valid_mask = np.zeros(K, dtype=np.bool_)
        valid_mask[:n_valid] = True

        # Pad neighbors
        padded_neighbors = np.full(K, self.pad_node, dtype=np.int64)
        padded_neighbors[:n_valid] = neighbors
        # Pad times
        padded_times = np.zeros(K, dtype=np.float32)

        padded_times[:n_valid] = times

        if return_edge_idxs:
            padded_edge_idxs = np.full(K, -1, dtype=np.int64)
            padded_edge_idxs[:n_valid] = edge_idxs
            return (
                padded_neighbors,
                padded_times,
                padded_edge_idxs,
                valid_mask,
                n_valid,
            )

        return (
            padded_neighbors,
            padded_times,
            valid_mask,
            n_valid,
        )


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]
  