import numpy as np
import random
import pandas as pd
import torch
from bisect import bisect_right
from collections import defaultdict
from torch.utils.data import Dataset
import torch.nn.functional as F

'''
https://pytorch-geometric.readthedocs.io/en/2.5.2/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
Data Parameters:
  x (torch.Tensor, optional) : Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
  edge_index (LongTensor, optional) : Graph connectivity in COO format with shape [2, num_edges]. (default: None)
  edge_attr (torch.Tensor, optional) : Edge feature matrix with shape [num_edges, num_edge_features]. (default: None)
  y (torch.Tensor, optional) : Graph-level or node-level ground-truth labels with arbitrary shape. (default: None)
  pos (torch.Tensor, optional) : Node position matrix with shape [num_nodes, num_dimensions]. (default: None)
  time (torch.Tensor, optional) : The timestamps for each event with shape [num_edges] or [num_nodes]. (default: None)
'''


'''
the strongest version is to sample one observed continuation according to a temporal distribution.
prefer temporally plausible continuations 
    q(x∣v,t)~exp(−lambda(t_x​−t))
So closer future interactions are more likely, but not the only valid target
'''
def sample_temporal_target(valid_neighbors, valid_times, current_time, lamb=0.1):
    delta = valid_times - current_time
    weights = np.exp(-lamb * delta)
    probs = weights / weights.sum()

    idx = np.random.choice(len(valid_neighbors), p=probs)

    return valid_neighbors[idx]


def temporal_target_distribution(times, mask, current_time, beta=0.001):
    delta = times - current_time
    delta = np.maximum(delta, 0)

    weights = np.exp(-beta * delta).astype(np.float32)
    weights[~mask] = 0.0

    total = weights.sum()
    if total <= 0:
        return None
    probs = weights / total
    return probs


def _valid_target_index(mask, probs=None, supervision_mode="earliest"):
    valid_positions = np.flatnonzero(mask)
    if len(valid_positions) == 0:
        return None

    if supervision_mode == "earliest":
        return int(valid_positions[0])

    if supervision_mode == "sampled":
        if probs is None:
            return int(valid_positions[0])
        valid_probs = probs[valid_positions].astype(np.float32)
        total = valid_probs.sum()
        if total <= 0:
            return int(valid_positions[0])
        valid_probs = valid_probs / total
        return int(valid_positions[np.random.choice(len(valid_positions), p=valid_probs)])

    return None

class ObservedTrajectoryIndex:
    """
    Data-driven supervision class
        We used new
            q_{obs}(x_i \mid u,v) = \frac{N(u,v,x_i)}{\sum_{x_j \in \mathcal{C}(v,t)} N(u,v,x_j)},
        Instead of exponencial decay assumption

    Builds empirical continuation counts for temporal triples:

        u --t1--> v --t2--> x,  with t2 > t1

    The resulting counts estimate q_obs(x | u, v).
    """

    def __init__(
        self,
        sources,
        destinations,
        timestamps,
        max_horizon=None,
        max_continuations=None,
    ):
        
        self.counts = defaultdict(lambda: defaultdict(int))

        # A stable NumPy sort avoids materializing and sorting one Python tuple
        # per event. Stability preserves the input order of equal timestamps.
        sources = np.asarray(sources)
        destinations = np.asarray(destinations)
        timestamps = np.asarray(timestamps)
        event_order = np.argsort(timestamps, kind="stable")

        # Store (timestamp, neighbor), so binary search can jump directly past
        # the node's history instead of re-scanning it for every edge.
        incident = defaultdict(list)

        for event_idx in event_order:
            src = int(sources[event_idx])
            dst = int(destinations[event_idx])
            ts = float(timestamps[event_idx])
            incident[src].append((ts, dst))
            incident[dst].append((ts, src))

        counts = self.counts
        max_timestamp_neighbor = float("inf")

        for event_idx in event_order:
            u = int(sources[event_idx])
            v = int(destinations[event_idx])
            t_uv = float(timestamps[event_idx])
            v_incident = incident[v]

            n_added = 0

            # Equal-time events are skipped because continuations must be
            # strictly later than the observed (u, v) event.
            first_future = bisect_right(
                v_incident,
                (t_uv, max_timestamp_neighbor),
            )

            for incident_idx in range(first_future, len(v_incident)):
                t_vx, x = v_incident[incident_idx]

                if max_horizon is not None:
                    if t_vx - t_uv > max_horizon:
                        break

                # Optional: avoid immediately returning to u.
                if x == u:
                    continue

                counts[(u, v)][x] += 1
                n_added += 1

                if (
                    max_continuations is not None
                    and n_added >= max_continuations
                ):
                    break

    def get_distribution(
        self,
        previous_node,
        current_node,
        candidate_nodes,
        mask,
        smoothing=0.0,
    ):
        """
        Returns q_obs over the padded candidate set.

        When the observed continuation table has no support for this context,
        fall back to a uniform distribution over the valid candidates instead of
        returning an all-zero target. Otherwise the KL objective becomes
        degenerate and the walk head receives almost no signal.
        """

        candidate_nodes = np.asarray(candidate_nodes)
        mask = np.asarray(mask, dtype=bool)

        if candidate_nodes.size == 0 or not mask.any():
            return None

        probs = np.zeros(
            len(candidate_nodes),
            dtype=np.float32,
        )

        continuation_counts = self.counts.get(
            (int(previous_node), int(current_node)),
            {},
        )

        for idx, candidate in enumerate(candidate_nodes):
            if not mask[idx]:
                continue

            probs[idx] = float(
                continuation_counts.get(int(candidate), 0)
            )

        if smoothing > 0:
            probs[mask] += smoothing

        valid_total = probs[mask].sum()
        if valid_total <= 0:
            probs[mask] = 1.0
            valid_total = probs[mask].sum()

        return probs / valid_total

class ContextBase(Dataset):
    def __init__(self, graph, sampler):
        self.examples = []
        self.graph = graph
        self.sampler = sampler

    def build_context(self, node, ts, is_forward=False):
        '''
            Return the K neighbors context for a node, it also return
            information about the current node degree and the neighbors degree
        ''' 
        neighbors, times, edge_idxs, mask, n_valid = self.sampler.sample_k(
            node_id=node,
            current_time=ts,
            is_forward=is_forward,
            return_edge_idxs=True,
        )

        if is_forward:
            # Future continuation: t_vx - t
            delta_t = times - ts
        else:
            # Historical interaction: t - t_vx
            delta_t = ts - times

        if n_valid == 0:
            return None


        deg_current = self.graph.get_degree(node)
        deg_neighbors = self.graph.get_degree_neighbors(neighbors)

        delta_t[~mask] = 0
        delta_t = np.maximum(delta_t, 0)

        return {
            "deg_current": deg_current,
            "neighbors": neighbors, ## content 
            "neighbors_deg": deg_neighbors,
            "neighbors_times": times,
            "neighbors_delta_t": delta_t,
            "neighbors_masks": mask,
            "neighbors_edge_idxs": edge_idxs,
        }
        
    def get_last_iteraction(self, src, dst, t):
            
        last_time = self.graph.get_last_interaction(
            src=src,
            dst=dst,
            before_time=t,
        )

        if last_time is None:
            previous_edge_delta_t = 0.0
        else:
            previous_edge_delta_t = t - last_time

        return previous_edge_delta_t

    def _make_zero_edge_feature(self, size):
        feature_dim = getattr(self, "edge_feature_dim", 0)
        return np.zeros((size, feature_dim), dtype=np.float32)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

#class TemporalWalkLinkDataset(torch.utils.data.Dataset):
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

            # se nao tenho contexto continuo
            neg = self.build_context(neg_dst, ts)
            if neg is None:
                continue

            self.examples.append({
                "src": src,
                "dst": dst,
                "neg_dst": neg_dst,

                "pos_deg_current": pos["deg_current"],
                "pos_neighbors": pos["neighbors"],
                "pos_deg_neighbors": pos["neighbors_deg"],
                "pos_delta_t": pos["neighbors_delta_t"],
                "pos_mask": pos["neighbors_masks"],

                "neg_deg_current": neg["deg_current"],
                "neg_neighbors": neg["neighbors"],
                "neg_deg_neighbors": neg["neighbors_deg"],
                "neg_delta_t": neg["neighbors_delta_t"],
                "neg_mask": neg["neighbors_masks"],

                "target_idx": 0,   # earliest future neighbor
            })


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class TemporalWalkSupervisionDataset(ContextBase):
    """
    supervision_mode:
        "earliest" -> Experiment 1
        "sampled"  -> Experiment 2
        "soft"     -> Experiment 3
    """

    def __init__(
        self,
        edge_dataset,
        graph,
        sampler,
        num_nodes,
        supervision_mode="earliest",
        beta=0.1,
        is_fordward=True
    ):
        super().__init__(graph, sampler)
        self.supervision_mode = supervision_mode
        self.beta = beta
        self.num_nodes = num_nodes

        assert supervision_mode in {
            "earliest",
            "sampled",
            "soft",
        }

        for sample in edge_dataset:
            src = int(sample["src"])
            dst = int(sample["dst"])
            ts = float(sample["ts"])

            # is_fordward = True for positive future neighbors
            pos_context = self.build_context(node=dst, ts=ts, is_forward=is_fordward)
            if pos_context is None:
                continue

            neg_dst = self.sample_negative_node(src=src, dst=dst, ts=ts)
            if neg_dst is None:
                continue

            neg_context = self.build_context(node=neg_dst, ts=ts)
            if neg_context is None:
                continue

            example = {
                "src": src,
                "dst": dst,
                "neg_dst": neg_dst,
                "pos_neighbors": pos_context["neighbors"],
                "pos_deg_current": pos_context["deg_current"],
                "pos_deg_neighbors": pos_context["neighbors_deg"],
                "pos_delta_t": pos_context["neighbors_delta_t"],
                "pos_mask": pos_context["neighbors_masks"],
                "neg_neighbors": neg_context["neighbors"],
                "neg_deg_current": neg_context["deg_current"],
                "neg_deg_neighbors": neg_context["neighbors_deg"],
                "neg_delta_t": neg_context["neighbors_delta_t"],
                "neg_mask": neg_context["neighbors_masks"],
            }

            # ---------------------------------------
            # Experiment 1: earliest future neighbor
            # ---------------------------------------
            if supervision_mode == "earliest":
                target_idx = _valid_target_index(
                    pos_context["neighbors_masks"],
                    supervision_mode="earliest",
                )
                if target_idx is None:
                    continue
                example["target_idx"] = target_idx

            # ---------------------------------------
            # Experiment 2: sampled temporal target
            # q(x) proportional to exp(-beta * delta_t)
            # ---------------------------------------
            elif supervision_mode == "sampled":
                probs = temporal_target_distribution(
                    times=pos_context["neighbors_times"], 
                    mask=pos_context["neighbors_masks"], 
                    current_time=ts, 
                    beta=beta
                )

                if probs is None:
                    continue

                target_idx = _valid_target_index(
                    pos_context["neighbors_masks"],
                    probs=probs,
                    supervision_mode="sampled",
                )
                if target_idx is None:
                    continue
                example["target_probs"] = probs.astype(np.float32)
                example["target_idx"] = int(target_idx)

            # ---------------------------------------
            # Experiment 3: soft-label supervision
            # target_probs = q(x)
            # ---------------------------------------
            elif supervision_mode == "soft":
                # the target distribution is :q_i = exp(-\beta\delta t)
                # exp(-\beta*median_future_delta) receives the half the weight of immediate neighbor
                # This has three advantages:

                # Scale adaptation: beta changes with each dataset’s timestamp units.
                # Interpretability: the median future gap is the decay half-life.
                # Robustness: the median is less affected by very large temporal gaps than the mean
                
                if beta==-1:
                    valid_deltas = pos_context["neighbors_delta_t"][
                        pos_context["neighbors_masks"]
                    ]
                    valid_deltas = valid_deltas[valid_deltas > 0]
                    if valid_deltas.size == 0:
                        continue
                    median_delta = np.median(valid_deltas)
                    beta = np.log(2.0) / median_delta

                    
                probs = temporal_target_distribution(
                    times=pos_context["neighbors_times"], 
                    mask=pos_context["neighbors_masks"], 
                    current_time=ts, 
                    beta=beta
                )

                if probs is None:
                    continue

                example["target_probs"] = probs.astype(np.float32)

            self.examples.append(example)

    def sample_negative_node(self, src, dst, ts=None, pool_size=16):
        """
            the negative node is likely to be:
                - A neighbor of the true destination dst;
                - Not equal to src or dst, but degree-similar to dst.

            Conceptually, it creates a degree-aware hard negative.
        """
        if self.num_nodes <= 0:
            return int(dst)
        seen = {src, dst}

        # get neighbors of dst
        if ts is None:
            candidate_nodes = set(self.graph.get_neighbors(dst, undirected=True, unique=True))
        else:    
            candidate_nodes = set(self.graph.get_neighbors(
                dst,
                timestamp=ts,
                is_fordward=False,
                undirected=True,
                unique=True,
            ))

        # instead of random negatives, it samples nodes structurally close to dst.
        # candidate_nodes = [node for node in candidate_nodes if node not in seen]
        # we check if the edge exists before the current timestamp,
        # and if it does, we exclude it from the candidate pool.
        candidate_nodes = [node 
                           for node in candidate_nodes 
                           if node not in seen and not self.graph.edge_exists_before(src, node, ts)]
        
        #It limits the candidate pool to at most 16 nodes.
        if len(candidate_nodes) > pool_size:
            candidate_nodes = list(np.random.choice(candidate_nodes, size=pool_size, replace=False))
        if not candidate_nodes:
            return None

        # Nodes with degree similar to dst receive higher probability.
        dst_degree = self.graph.get_degree(dst)
        weights = np.ones(len(candidate_nodes), dtype=np.float32)
        for idx, node in enumerate(candidate_nodes):
            degree_gap = abs(self.graph.get_degree(node) - dst_degree)
            weights[idx] = 1.0 + 1.0 / (1.0 + degree_gap)
        weights = np.maximum(weights, 1e-6)
        probs = weights / weights.sum()
        return int(np.random.choice(candidate_nodes, p=probs))

    
    def __getitem__(self, idx):
        '''
        In every epoch get different sample  
        '''    
        example = self.examples[idx]

        if self.supervision_mode == "sampled":
            probs = example["target_probs"]
            mask = example["pos_mask"]
            target = _valid_target_index(
                mask,
                probs=probs,
                supervision_mode="sampled",
            )
            if target is not None:
                example["target_idx"] = target

        return example
    
class TemporalWalkSupervisionDatasetFeat(ContextBase):
    """
    supervision_mode:
        "earliest" -> Experiment 1
        "sampled"  -> Experiment 2
        "soft"     -> Experiment 3
    """

    def __init__(
        self,
        edge_dataset,
        graph,
        sampler,
        num_nodes,
        node_features,
        edge_features,
        pad_node=None,
        supervision_mode="soft",
        beta=0.1,
        is_fordward=True

    ):
        super().__init__(graph, sampler)
        self.supervision_mode = supervision_mode
        self.beta = beta
        self.num_nodes = num_nodes
        self.node_features = np.asarray(node_features)
        self.edge_features = np.asarray(edge_features)
        self.pad_node = pad_node

        self.node_feature_dim = 0
        if self.node_features is not None:
            self.node_feature_dim = self.node_features.shape[1]

        self.edge_feature_dim = 0
        if self.edge_features is not None:
            self.edge_feature_dim = self.edge_features.shape[1]

        assert supervision_mode in {
            "earliest",
            "sampled",
            "soft",
        }

        for sample in edge_dataset:
            src = int(sample["src"])
            dst = int(sample["dst"])
            ts = float(sample["ts"])
            edge_idx = int(sample.get("idx", -1))

            # Historical context: positive link prediction
            # we get the K neighbors context for the node dst : 
            # -------------------------------------------------
            # Nomenclature: 
            #   src -> dst -> x_i |
            #   prev -> current -> neighbors
            #   u -> v -> x_i
            # Positive context is: x_i
            # -------------------------------------------------
            pos_context = self.build_context(node=dst, ts=ts, is_forward=is_fordward)
            if pos_context is None: 
                continue

            # -------------------------------------------------
            # Negative queried edge: u -> v_neg
            # -------------------------------------------------
            neg_dst = self.sample_negative_node(src=src, dst=dst, ts=ts)
            if neg_dst is None:
                continue

            # Historical context is available for the negative candidate at ts.
            neg_context = self.build_context(node=neg_dst, ts=ts, is_forward=False)
            if neg_context is None:
                continue
   
            # [f_u | f_v | f_x | e_uv | e_vx]
            # [f_prev | f_curr | f_neighbors | e_current | e_neighbors]
            previous_node_feat = self._get_node_features(src)
            current_node_feat = self._get_node_features(dst)
            neighbors_node_feat = self._get_node_features(pos_context["neighbors"])
            # e_uv
            previous_edge_feat = self._get_edge_features(edge_idx)
            
            # e_vx_i it is array of arrays
            neighbors_edge_feat = self._get_edge_features(pos_context["neighbors_edge_idxs"])
            #previous_time = np.array([ts], dtype=np.float32) 
            previous_edge_delta_t = self.get_last_iteraction(src, dst, ts)
            neighbors_times = pos_context["neighbors_times"].astype(np.float32)

            neg_current_node_feat = self._get_node_features(neg_dst)
            neg_neighbor_node_feat = self._get_node_features(neg_context["neighbors"])
            #neg_previous_edge_feat = self._make_zero_edge_feature(1)[0]
            neg_neighbors_edge_feat = self._get_edge_features(neg_context["neighbors_edge_idxs"])

            # neg_previous_node_feat = self._safe_node_features(src)
            # neg_current_node_feat = self._safe_node_features(neg_dst)
            # neg_neighbor_node_feat = self._safe_node_features(neg_context["neighbors"])
            # neg_previous_edge_feat = self._make_zero_edge_feature(1)
            # neg_neighbor_edge_feat = self._safe_edge_features(neg_context["neighbor_edge_idxs"])
            # neg_previous_edge_delta = np.zeros((1,), dtype=np.float32)
            # neg_neighbor_edge_delta = neg_context["delta_t"].astype(np.float32)

            example = {
                "src": src,
                "dst": dst,
                "neg_dst": neg_dst,

                # positive nodes neighbors of dst
                "pos_neighbors": pos_context["neighbors"],
                "pos_deg_current": pos_context["deg_current"],
                "pos_deg_neighbors": pos_context["neighbors_deg"],
                "pos_delta_t": pos_context["neighbors_delta_t"],
                "pos_mask": pos_context["neighbors_masks"],

                # negative similar nodes
                "neg_neighbors": neg_context["neighbors"],
                "neg_deg_current": neg_context["deg_current"],
                "neg_deg_neighbors": neg_context["neighbors_deg"],
                "neg_delta_t": neg_context["neighbors_delta_t"],
                "neg_mask": neg_context["neighbors_masks"],

                # node and edge features 
                "previous_node_feat": previous_node_feat.astype(np.float32),
                "current_node_feat": current_node_feat.astype(np.float32),
                "neighbor_node_feat": neighbors_node_feat.astype(np.float32),
                "previous_edge_feat": previous_edge_feat.astype(np.float32),  # u-v
                "neighbors_edge_feat": neighbors_edge_feat.astype(np.float32), # v-x_i
                # times
                "previous_time": previous_edge_delta_t,
                "neighbors_times": neighbors_times,
                "neighbors_delta_t": pos_context["neighbors_delta_t"],

                "neg_current_node_feat": neg_current_node_feat.astype(np.float32),
                "neg_neighbor_node_feat": neg_neighbor_node_feat.astype(np.float32),
                #"neg_previous_edge_feat": neg_previous_edge_feat.astype(np.float32),
                "neg_neighbors_edge_feat": neg_neighbors_edge_feat.astype(np.float32),
                "neg_neighbors_delta_t": neg_context["neighbors_delta_t"].astype(np.float32)
            }

            # ---------------------------------------
            # Experiment 1: earliest future neighbor
            # ---------------------------------------
            if supervision_mode == "earliest":
                target_idx = _valid_target_index(
                    pos_context["neighbors_masks"],
                    supervision_mode="earliest",
                )
                if target_idx is None:
                    continue
                example["target_idx"] = target_idx

            # ---------------------------------------
            # Experiment 2: sampled temporal target
            # q(x) proportional to exp(-beta * delta_t)
            # ---------------------------------------
            elif supervision_mode == "sampled":
                probs = temporal_target_distribution(
                    times=pos_context["neighbors_times"],
                    mask=pos_context["neighbors_masks"],
                    current_time=ts,
                    beta=beta,
                )

                if probs is None:
                    continue

                target_idx = _valid_target_index(
                    pos_context["neighbors_masks"],
                    probs=probs,
                    supervision_mode="sampled",
                )
                if target_idx is None:
                    continue
                example["target_probs"] = probs.astype(np.float32)
                example["target_idx"] = int(target_idx)

            # ---------------------------------------
            # Experiment 3: soft-label supervision
            # target_probs = q(x)
            # ---------------------------------------
            elif supervision_mode == "soft":
                probs = temporal_target_distribution(
                    times=pos_context["neighbors_times"],
                    mask=pos_context["neighbors_masks"],
                    current_time=ts,
                    beta=beta,
                )

                if probs is None:
                    continue

                example["target_probs"] = probs.astype(np.float32)

            self.examples.append(example)

    def sample_negative_node(self, src, dst, ts=None, pool_size=16):
        """
            the negative node is likely to be:

            a neighbor of the true destination dst;
            not equal to src or dst;
            degree-similar to dst.

            Conceptually, it creates a degree-aware hard negative.
        """
        if self.num_nodes <= 0:
            return int(dst)
        seen = {src, dst}
        if ts is None:
            candidate_nodes = set(self.graph.get_neighbors(dst, undirected=True, unique=True))
        else:
            # sampling from neighbors of dst
            candidate_nodes = set(
                self.graph.get_neighbors(
                    dst,
                    timestamp=ts,
                    is_fordward=False,
                    undirected=True,
                    unique=True,
                )
            )

        # instead of random negatives, it samples nodes structurally close to dst.
        # candidate_nodes = [node for node in candidate_nodes if node not in seen]
        # we check if the edge exists before the current timestamp, and if it does, we exclude it from the candidate pool.
        candidate_nodes = [
            node for node in candidate_nodes 
            if node not in seen and 
                    not self.graph.edge_exists_before(src, node, ts)
        ]
        
        #It limits the candidate pool to at most 16 nodes.
        if len(candidate_nodes) > pool_size:
            candidate_nodes = list(np.random.choice(candidate_nodes, size=pool_size, replace=False))
        if not candidate_nodes:
            #return int(dst)
            return None

        # Nodes with degree similar to dst receive higher probability.
        dst_degree = self.graph.get_degree(dst)
        weights = np.ones(len(candidate_nodes), dtype=np.float32)
        for idx, node in enumerate(candidate_nodes):
            degree_gap = abs(self.graph.get_degree(node) - dst_degree)
            weights[idx] = 1.0 + 1.0 / (1.0 + degree_gap)
        weights = np.maximum(weights, 1e-6)
        probs = weights / weights.sum()
        return int(np.random.choice(candidate_nodes, p=probs))

    
    def __getitem__(self, idx):
        '''
        In every epoch get different sample  
        '''    
        example = self.examples[idx]

        if self.supervision_mode == "sampled":
            probs = example["target_probs"]
            mask = example["pos_mask"]
            target = _valid_target_index(
                mask,
                probs=probs,
                supervision_mode="sampled",
            )
            if target is not None:
                example["target_idx"] = target

        return example
    
    def _get_node_features(self, node):
        if self.node_features is None:
            return np.zeros((1, self.node_feature_dim), dtype=np.float32)

        node_ids = np.atleast_1d(np.asarray(node, dtype=np.int64))
        out = np.zeros((len(node_ids), self.node_feature_dim), dtype=np.float32)

        for i, node_id in enumerate(node_ids):
            if 0 <= node_id < len(self.node_features):
                out[i] = self.node_features[node_id]
        return out

    def _get_edge_features(self, edge_idx):
        if self.edge_features is None:
            return np.zeros((1, self.edge_feature_dim), dtype=np.float32)

        edge_ids = np.atleast_1d(np.asarray(edge_idx, dtype=np.int64))
        out = np.zeros((len(edge_ids), self.edge_feature_dim), dtype=np.float32)

        for i, edge_id in enumerate(edge_ids):
            if 0 <= edge_id < len(self.edge_features):
                out[i] = self.edge_features[edge_id]

        return out
    
class TemporalWalkSupervisionDataDriven(ContextBase):
    def __init__(
        self,
        edge_dataset,
        graph,
        sampler,
        num_nodes,
        trajectory_index,
        supervision_mode="observed_hard",
        smoothing=0.0,
    ):
        super().__init__(graph, sampler)

        self.num_nodes = num_nodes
        self.trajectory_index = trajectory_index
        self.supervision_mode = supervision_mode
        self.smoothing = smoothing

        assert supervision_mode in {
            "observed_hard",
            "observed_sampled",
            "observed_soft",
        }

        for sample in edge_dataset:
            src = int(sample["src"])
            dst = int(sample["dst"])
            ts = float(sample["ts"])


            '''
                return {
                    "deg_current": deg_current,
                    "neighbors": neighbors, ## content 
                    "neighbors_deg": deg_neighbors,
                    "neighbors_times": times,
                    "neighbors_delta_t": delta_t,
                    "neighbors_masks": mask,
                    "neighbors_edge_idxs": edge_idxs,
                }
            '''

            pos_context = self.build_context(
                node=dst,
                ts=ts,
            )

            if pos_context is None:
                continue

            target_probs = trajectory_index.get_distribution(
                previous_node=src,
                current_node=dst,
                candidate_nodes=pos_context["neighbors"],
                mask=pos_context["neighbors_masks"],
                smoothing=smoothing,
            )

            # If there is no empirical support, fall back to a uniform
            # distribution over the valid candidates so the walk loss remains
            # meaningful instead of collapsing to zero mass.
            if target_probs is None:
                valid_positions = np.flatnonzero(pos_context["neighbors_masks"])
                if len(valid_positions) == 0:
                    continue
                target_probs = np.zeros_like(pos_context["neighbors"], dtype=np.float32)
                target_probs[valid_positions] = 1.0 / len(valid_positions)

            neg_dst = self.sample_negative_node(
                src=src,
                dst=dst,
                ts=ts,
            )

            neg_context = self.build_context(
                node=neg_dst,
                ts=ts,
            )

            if neg_context is None:
                continue

            '''
                return {
                    "deg_current": deg_current,
                    "neighbors": neighbors, ## content 
                    "neighbors_deg": deg_neighbors,
                    "neighbors_times": times,
                    "neighbors_delta_t": delta_t,
                    "neighbors_masks": mask,
                    "neighbors_edge_idxs": edge_idxs,
                }
            '''
            example = {
                "src": src,
                "dst": dst,
                "neg_dst": neg_dst,

                "pos_neighbors": pos_context["neighbors"],
                "pos_deg_current": pos_context["deg_current"],
                "pos_deg_neighbors": pos_context["neighbors_deg"],
                "pos_delta_t": pos_context["neighbors_delta_t"],
                "pos_mask": pos_context["neighbors_masks"],

                "neg_neighbors": neg_context["neighbors"],
                "neg_deg_current": neg_context["deg_current"],
                "neg_deg_neighbors": neg_context["neighbors_deg"],
                "neg_delta_t": neg_context["neighbors_delta_t"],
                "neg_mask": neg_context["neighbors_masks"],

                "target_probs": target_probs.astype(np.float32),
            }

            if supervision_mode == "observed_hard":
                example["target_idx"] = int(
                    np.argmax(target_probs)
                )

            elif supervision_mode == "observed_sampled":
                example["target_idx"] = int(
                    np.random.choice(
                        len(target_probs),
                        p=target_probs,
                    )
                )

            self.examples.append(example)

    def sample_negative_node(
        self,
        src,
        dst,
        ts=None,
        pool_size=16,
    ):
        seen = {src, dst}

        candidate_nodes = set(
            self.graph.get_neighbors(
                dst,
                timestamp=ts,
                undirected=True,
                unique=True,
            )
        )

        candidate_nodes = [
            node
            for node in candidate_nodes
            if node not in seen
            and not self.graph.edge_exists_before(src, node, ts)
        ]

        if len(candidate_nodes) > pool_size:
            candidate_nodes = list(
                np.random.choice(
                    candidate_nodes,
                    size=pool_size,
                    replace=False,
                )
            )

        if not candidate_nodes:
            return int(dst)

        return int(np.random.choice(candidate_nodes))

    def __getitem__(self, idx):
        example = self.examples[idx].copy()

        if self.supervision_mode == "observed_sampled":
            probs = example["target_probs"]

            example["target_idx"] = int(
                np.random.choice(
                    len(probs),
                    p=probs,
                )
            )

        return example
      
class TemporalWalkContinuationDataset(torch.utils.data.Dataset):

    def __init__(self, edge_dataset, graph, sampler):
        self.examples = []

        for sample in edge_dataset:
            u = int(sample["src"])
            v = int(sample["dst"])
            t = float(sample["ts"])

            neighbors, times, mask, n_valid = sampler.sample(
                node_id=v,
                current_time=t,
                is_forward=True,
            )

            if n_valid == 0:
                continue

            # valid candidates only
            valid_neighbors = neighbors[mask]
            valid_times = times[mask]

            # observed continuation: earliest edge from v after t
            # BUT target_idx is now derived from the actual candidate node
            next_idx_valid = np.argmin(valid_times)
            x_true = valid_neighbors[next_idx_valid]

            # find x_true inside full padded candidate array
            target_positions = np.where(neighbors == x_true)[0]

            if len(target_positions) == 0:
                continue

            #This is still close to current version, but it makes the target explicit.
            target_idx = int(target_positions[0])

            x_true = sample_temporal_target(
                valid_neighbors,
                valid_times,
                t,
                beta=0.1,
            )

            deg_current = graph.get_degree(v)
            deg_neighbors = graph.get_degree_neighbors(neighbors)

            delta_t = times - t
            delta_t[~mask] = 0

            self.examples.append({
                "src": u,
                "dst": v,
                "neighbors": neighbors,
                "deg_current": deg_current,
                "deg_neighbors": deg_neighbors,
                "delta_t": delta_t,
                "mask": mask,
                "target_idx": target_idx,
                "target_node": int(x_true),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    
'''
 Precomputed dst, neighbors, graph.get_degree()
'''
class TemporalWalkDataset(Dataset):

    def __init__( 
        self,
        edge_dataset,
        graph,
        sampler,
        num_nodes
    ):

        self.examples = []
        self.num_nodes = num_nodes

        for sample in edge_dataset:
            src = sample["src"]
            dst = sample["dst"]
            ts  = sample["ts"]

            neighbors, times, mask, n_valid = sampler.sample_k(
                node_id=dst,
                current_time=ts,
                is_forward=True
            )

            if n_valid == 0:
                continue

            deg_current = graph.get_degree(dst)
            deg_neighbors = graph.get_degree_neighbors(neighbors)

            delta_t = times - ts
            
            neg_dst = np.random.randint(0,  num_nodes)

            while neg_dst == dst:
                neg_dst = np.random.randint(0, num_nodes)

            self.examples.append(
                {
                    "src": src,
                    "dst": dst,
                    "deg_current": deg_current,
                    "neighbors": neighbors,
                    "deg_neighbors": deg_neighbors,
                    "delta_t": delta_t,
                    "mask": mask,
                    "target_idx": 0,
                    "neg_dst": neg_dst
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
    
class EdgeDataset(Dataset):

    def __init__(
        self,
        sources,
        destinations,
        timestamps,
        edge_idxs,
        labels,
    ):

        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        
        n = len(self.sources)
        self.n_interactions = n

        self.unique_nodes = (
            set(sources) | set(destinations)
        )

        self.n_unique_nodes = len(self.unique_nodes)

    def __len__(self):

        return len(self.sources)

    def __getitem__(self, idx):

        return {
            "src": self.sources[idx],
            "dst": self.destinations[idx],
            "ts": self.timestamps[idx],
            "label": self.labels[idx],
        }


class Graph(Dataset):

    def __init__(
        self,
        sources,
        destinations,
        timestamps,
        edge_idxs,
        labels,
    ):

        n = len(sources)

        if not (
            len(destinations) == n and
            len(timestamps) == n and
            len(edge_idxs) == n and
            len(labels) == n
        ):
            raise ValueError(
                "All inputs must have same length."
            )

        self.sources = np.array(sources)
        self.destinations = np.array(destinations)
        self.timestamps = np.array(timestamps)
        self.edge_idxs = np.array(edge_idxs)
        self.labels = np.array(labels)
        self.index = 0

        self.n_interactions = n

        self.unique_nodes = (
            set(sources) | set(destinations)
        )

        self.n_unique_nodes = len(self.unique_nodes)

        self.adj_list = defaultdict(list)

        for src, dst, ts, eidx, lbl in zip(
            self.sources,
            self.destinations,
            self.timestamps,
            self.edge_idxs,
            self.labels,
        ):

            self.adj_list[src].append((dst, ts, eidx, lbl))
            self.adj_list[dst].append((src, ts, eidx, lbl))

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

    def __len__(self):
        return self.n_interactions

    def __getitem__(self, idx):

        return {
            "src": self.sources[idx],
            "dst": self.destinations[idx],
            "ts": self.timestamps[idx],
            "idx": self.edge_idxs[idx],
            "label": self.labels[idx],
        }
            
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index < self.n_interactions:
            result = (
                  self.sources[self.index],
                  self.destinations[self.index],
                  self.timestamps[self.index])
            
            self.index+=1
            return result
        else:
            raise StopIteration

    def get_nodes(self):
        return self.unique_nodes
    
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
    
    def get_neighbors_array(self, node_id, include_edge_weight=False):

        neighbors = []
        times = []
        
        for (neighbor, time, _, _) in self.adj_list[node_id]:
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
       return self.mapping_degrees[node_id]
    
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
    
class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)
    

def load_data(path_file, dataset_name):
    ### Load data and train val test split
    graph_df = pd.read_csv('{}/ml_{}.csv'.format(path_file, dataset_name))
    edge_features = np.load('{}/ml_{}.npy'.format(path_file, dataset_name))
    node_features = np.load('{}/ml_{}_node.npy'.format(path_file, dataset_name)) 

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    random.seed(2020)

    #train_mask = timestamps <= test_time
    #test_mask = timestamps > test_time
    #val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)

    # much better configuration for training, validation, and testing splits 
    # means training has access to events occurring between 70% and 85% of the timeline.
    # the previous configurarion has already seen much more recent interactions than intended
    train_mask = timestamps <= val_time
    val_mask = (timestamps > val_time) & (timestamps <= test_time)
    test_mask = timestamps > test_time

    train_edge = EdgeDataset(
        sources[train_mask],
        destinations[train_mask],
        timestamps[train_mask],
        edge_idxs[train_mask],
        labels[train_mask],
    )

    test_edge = EdgeDataset(
        sources[test_mask],
        destinations[test_mask],
        timestamps[test_mask],
        edge_idxs[test_mask],
        labels[test_mask],
    )

    val_edge = EdgeDataset(
        sources[val_mask], 
        destinations[val_mask],
        timestamps[val_mask],
        edge_idxs[val_mask],
        labels[val_mask]
    )

    full_edge = EdgeDataset(sources, destinations, timestamps, edge_idxs, labels)

    return full_edge, node_features, edge_features, train_edge, val_edge, test_edge

def get_data(path_file, dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
  ### Load data and train val test split
  graph_df = pd.read_csv('{}/ml_{}.csv'.format(path_file, dataset_name))
  edge_features = np.load('{}/ml_{}.npy'.format(path_file, dataset_name))
  node_features = np.load('{}/ml_{}_node.npy'.format(path_file, dataset_name)) 
  
  if randomize_features:
    node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  full_data = EdgeDataset(sources, destinations, timestamps, edge_idxs, labels)

  random.seed(2020)

  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)

  # Compute nodes which appear at test time
  test_node_set = set(sources[timestamps > val_time]).union(
    set(destinations[timestamps > val_time]))
  # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
  # their edges from training
  new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

  # Mask saying for each source and destination whether they are new test nodes
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  # Mask which is true for edges with both destination and source not being new test nodes (because
  # we want to remove all edges involving any new test node)
  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

  # For train we keep edges happening before the validation time which do not involve any new node
  # used for inductiveness
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

  train_data = EdgeDataset(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  # define the new nodes sets for testing inductiveness of the model
  train_node_set = set(train_data.sources).union(train_data.destinations)
  assert len(train_node_set & new_test_node_set) == 0
  new_node_set = node_set - train_node_set

  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time

  if different_new_nodes_between_val_and_test:
    n_new_nodes = len(new_test_node_set) // 2
    val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
    test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

    edge_contains_new_val_node_mask = np.array(
      [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
    edge_contains_new_test_node_mask = np.array(
      [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


  else:
    edge_contains_new_node_mask = np.array(
      [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  # validation and test with all edges
  val_data = EdgeDataset(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = EdgeDataset(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  # validation and test with edges that at least has one new node (not in training set)
  new_node_val_data = EdgeDataset(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])

  new_node_test_data = EdgeDataset(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])

  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
  print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
  print("The new node test dataset has {} interactions, involving {} different nodes".format(
    new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
  print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
    len(new_test_node_set)))

  return node_features, edge_features, full_data, train_data, val_data, test_data, \
         new_node_val_data, new_node_test_data


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst



def collate_temporal_walk(batch):
    output = {
        "src": torch.tensor([x["src"] for x in batch], dtype=torch.long),
        "dst": torch.tensor([x["dst"] for x in batch], dtype=torch.long),
        "deg_current": torch.tensor([x["deg_current"] for x in batch], dtype=torch.long),
        "neighbors": torch.tensor(np.stack([x["neighbors"] for x in batch]), dtype=torch.long),
        "deg_neighbors": torch.tensor(np.stack([x["deg_neighbors"] for x in batch]), dtype=torch.long),
        "delta_t": torch.tensor(np.stack([x["delta_t"] for x in batch]), dtype=torch.float),
        "mask": torch.tensor(np.stack([x["mask"] for x in batch]), dtype=torch.bool),
        "neg_dst": torch.tensor([x["neg_dst"] for x in batch], dtype=torch.long),
    }

    if "target_idx" in batch[0]:
        output["target_idx"] = torch.tensor([x["target_idx"] for x in batch], dtype=torch.long)
    elif "target_probs" in batch[0]:
        output["target_probs"] = torch.tensor(np.stack([x["target_probs"] for x in batch]), dtype=torch.float)

    return output

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

def collate_temporal_walk_link_feat(batch):
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

    if "previous_node_feat" in batch[0]:
        feature_keys = (
            "previous_node_feat", "current_node_feat", "neighbor_node_feat",
            "neighbors_edge_feat", "neighbors_times", "neighbors_delta_t",
            "neg_current_node_feat",
            "neg_neighbor_node_feat", "neg_neighbors_edge_feat",
            "neg_neighbors_delta_t",
        )
        output.update({
            key: torch.tensor(np.stack([x[key] for x in batch]), dtype=torch.float)
            for key in feature_keys
        })

    if "target_idx" in batch[0]:
        output["target_idx"] = torch.tensor([x["target_idx"] for x in batch], dtype=torch.long)
    elif "target_probs" in batch[0]:
        output["target_probs"] = torch.tensor(np.stack([x["target_probs"] for x in batch]), dtype=torch.float)
    return output