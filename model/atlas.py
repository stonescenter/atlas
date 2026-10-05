import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .time_encoder import TimeEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()

        # Produces one attention score for each input embedding
        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor, mask: Union[torch.Tensor, None] = None):
        """
        x:
            Shape [batch_size, num_items, embedding_dim]

        mask:
            Shape [batch_size, num_items]
            True for valid items and False for padding.
        """

        # [B, K, D] -> [B, K]
        scores = self.attention(x).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        # Attention weights over the K items
        weights = F.softmax(scores, dim=-1)

        # Weighted sum: [B, K, 1] * [B, K, D] -> [B, D]
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)

        return pooled, weights
    
# ============================================================
# Temporal Attention Layer
# ============================================================

class TemporalAttentionLayer(nn.Module):

    def __init__(
        self,
        embedding_dim=64,
        time_dim=32,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.time_encoder = TimeEncoder(dimension=time_dim)

        temporal_dim = 2 * time_dim

        input_dim = (
            1                    # structural score
            + temporal_dim
            + 2 * embedding_dim
        )

        self.proj = nn.Linear(
            input_dim,
            embedding_dim
        )

        self.attn = nn.Linear(
            embedding_dim,
            1
        )

    # --------------------------------------------------------
    # Structural normalization
    # --------------------------------------------------------

    def structural_score(self, deg_u, deg_v):
        score = (
            1.0 /
            torch.sqrt(
                deg_u * deg_v + 1e-8
            )
        )

        return score.unsqueeze(-1)

    # --------------------------------------------------------
    # Attention score
    # --------------------------------------------------------

    def forward(
        self,
        h_u,
        h_v,
        deg_u,
        deg_v,
        delta_t,
    ):

        s_uv = self.structural_score(deg_u, deg_v)

        phi_t = self.time_encoder(delta_t) 
        
        x = torch.cat([s_uv, phi_t, h_u, h_v], dim=-1)
        # --------------------------------------------
        # Attention network
        # --------------------------------------------

        z = F.leaky_relu(self.proj(x), negative_slope=0.2)
            
        e_uv = self.attn(z).squeeze(-1)

        return e_uv




# ============================================================
# Temporal Link Predictor
# ============================================================

class TemporalLinkPredictor(nn.Module):

    def __init__(
        self,
        num_nodes,
        embedding_dim=64,
        time_dim=32,
    ):
        super().__init__()

        self.node_embedding = nn.Embedding(
            num_nodes,
            embedding_dim
        )

        nn.init.xavier_uniform_(
            self.node_embedding.weight
        )

        self.temporal_attention = TemporalAttentionLayer(
            embedding_dim=embedding_dim,
            time_dim=time_dim,
        )

    def forward(
        self,
        u,
        v,
        deg_u,
        deg_v,
        delta_t,
    ):

        h_u = self.node_embedding(u)

        h_v = self.node_embedding(v)

        score = self.temporal_attention(
            h_u=h_u,
            h_v=h_v,
            deg_u=deg_u,
            deg_v=deg_v,
            delta_t=delta_t,
        )

        return score

 
class TemporalWalkEncoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        embedding_dim=64,
        time_dim=32,
        hidden_dim=128,
        pad_node=None,
        dropout=0.1,
        debug=False,
        time_encoder=None,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_nodes + 1,
            embedding_dim,
            padding_idx=pad_node,
        )

        #self.time_encoder = TimeEncoder(time_dim)
        self.time_encoder = time_encoder if time_encoder is not None else TimeEncoder(time_dim)

        #with torch.no_grad():
        #   temporal_dim = int(self.time_encoder(torch.zeros(1, 1)).shape[-1])
        
        temporal_dim = 2 * time_dim

        input_dim = (
            1 +                 # structural score
            temporal_dim +      # time encoding
            3 * embedding_dim + # trajectory interaction features
            3 * embedding_dim   # h_prev, h_curr, h_next
        )

        # learns the semantic interaction among the structural, temporal and embedding features.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Walk policy head: predicts next candidate among K neighbors
        self.walk_head = nn.Linear(hidden_dim, 1)

        # Attention pooling for link prediction
        self.pool_attn = nn.Linear(hidden_dim, 1)
        #self.pool_attn = AttentionPooling(hidden_dim)

        # Link prediction head: predicts whether edge exists
        self.link_head = nn.Sequential(
            nn.Linear(hidden_dim + 2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.log_var_walk = nn.Parameter(torch.zeros(1))
        self.log_var_link = nn.Parameter(torch.zeros(1))

        self.debug = debug

    def structural_score(self, deg_current, deg_neighbors):
        """
        deg_current   : [B]
        deg_neighbors : [B,K]

        returns:
            structural : [B,K,1]
        """

        deg_current = deg_current.unsqueeze(1)  # [B,1]

        deg_current = torch.clamp(deg_current, min=1.0)
        deg_neighbors = torch.clamp(deg_neighbors, min=1.0)

        score = 1.0 / torch.sqrt(
            deg_current * deg_neighbors + 1e-8
        )

        return torch.log(score + 1e-8).unsqueeze(-1)

    def encode_walks(
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
        mask           : [B,K]

        returns:
            z_walk      : [B,K,H]
        """

        B, K = neighbor_nodes.shape

        h_prev = self.embedding(previous_nodes)   #u   # [B,D]
        h_curr = self.embedding(current_nodes)    #v   # [B,D]
        h_next = self.embedding(neighbor_nodes)   #x   # [B,K,D]

        h_prev_exp = h_prev.unsqueeze(1).expand(-1, K, -1)
        h_curr_exp = h_curr.unsqueeze(1).expand(-1, K, -1)

        # Avoid invalid temporal values for padded entries
        delta_t = torch.clamp(delta_t, min=0.0)

        structural = self.structural_score(
            deg_current,
            deg_neighbors,
        )                                            # [B,K,1]

        temporal = self.time_encoder(delta_t)        # [B,K,2*time_dim]

        # \pi(x|u,v,t) 
        # \phi = u * v
        # \phi = v * x 
        # interaction features
        
        #prev_next = h_prev_exp * h_next # u * x
        #curr_next = h_curr_exp * h_next # v * x
        
        # u->v, v-x, u->x interaction features
        trajetory  = torch.cat([h_prev_exp * h_curr_exp, h_curr_exp * h_next, h_prev_exp * h_next], dim=-1) 
        
        x = torch.cat(
            [
                structural,
                temporal,
                trajetory,
                h_prev_exp,
                h_curr_exp,
                h_next,
            ],
            dim=-1,
        )

        # learn with a MLP
        z_walk = self.encoder(x)                     # [B,K,H]

        if mask is not None:
            z_walk = z_walk.masked_fill(
                ~mask.unsqueeze(-1),
                0.0,
            )

        if self.debug:
            print("h_prev:", h_prev.shape)
            print("h_curr:", h_curr.shape)
            print("h_next:", h_next.shape)
            print("structural:", structural.shape)
            print("temporal:", temporal.shape)
            print("z_walk:", z_walk.shape)

        return z_walk, h_prev, h_curr

    def pool_walks(self, z_walk, mask):
        """
        z_walk : [B,K,H]
        mask   : [B,K]

        returns:
            z_pool : [B,H]
        """

        attn_logits = self.pool_attn(z_walk).squeeze(-1)  # [B,K]

        attn_logits = attn_logits.masked_fill(
            ~mask,
            -1e9,
        )

        attn = torch.softmax(attn_logits, dim=-1)         # [B,K]

        z_pool = torch.sum(
            z_walk * attn.unsqueeze(-1),
            dim=1,
        )                                                 # [B,H]

        return z_pool

    def forward(
        self,
        previous_nodes,
        current_nodes,
        neighbor_nodes,
        deg_current,
        deg_neighbors,
        delta_t,
        mask,
    ):
        """
        returns:
            walk_logits : [B,K]
            edge_score  : [B]
        """

        z_walk, h_prev, h_curr = self.encode_walks(
            previous_nodes,
            current_nodes,
            neighbor_nodes,
            deg_current,
            deg_neighbors,
            delta_t,
            mask,
        )

        # Output 1: temporal walk policy
        walk_logits = self.walk_head(z_walk).squeeze(-1)  # [B,K]

        # Output 2: temporal edge prediction
        z_pool = self.pool_walks(z_walk, mask)            # [B,H]
        #z_pool = self.pool_attn(z_walk, mask)            # [B,H]

        edge_repr = torch.cat([z_pool, h_prev, h_curr], dim=-1,)  # [B,H+2*D])

        edge_score = self.link_head(edge_repr).squeeze(-1)  # [B]

        return walk_logits, edge_score


   
class EdgeFeatureEncoder(nn.Module):    
    '''
        Encodes edge features and temporal information into a single laten representation.
    '''
    def __init__(
        self,
        time_dim,
        features_dim,
        output_dim
    ):
        super().__init__()

        self.time_encoder = TimeEncoder(dimension=time_dim)
        self.feature_encoder = nn.Linear(features_dim + 2 * time_dim, output_dim)

    def forward(self, delta_t, edge_features):

        time_enc = self.time_encoder(delta_t)
        if time_enc.ndim == edge_features.ndim + 1 and time_enc.shape[-2] == 1:
            time_enc = time_enc.squeeze(-2)
        x = torch.cat([time_enc, edge_features], dim=-1)
        return self.feature_encoder(x)

class NodeFeatureEncoder(nn.Module):
    def __init__(
            self,
            features_dim: int,
            output_dim: int,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(features_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, node_features):
        return self.encoder(node_features)
    
class TemporalWalkEncoderFeatures(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        node_hidden_dim: int = 64,
        edge_hidden_dim: int = 64,
        time_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        debug: bool = False,
    ):
        super().__init__()

        # Node features embedding layer.
        self.node_encoder = NodeFeatureEncoder(
            features_dim=node_feature_dim,
            output_dim=node_hidden_dim,
            dropout=dropout,
        )

        # Shared encoder for the observed and candidate edges.
        self.edge_encoder = EdgeFeatureEncoder(
            time_dim=time_dim,
            features_dim=edge_feature_dim,
            output_dim=edge_hidden_dim,
        )

        # structural feature
        structural_dim = 3
        input_dim = (
            structural_dim +                       # structural feature
            3 * node_hidden_dim  + # f_u, f_v, f_x
            edge_hidden_dim   # e_uv, e_vx
        )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.walk_head = nn.Linear(hidden_dim, 1)

        self.pool_attn = nn.Linear(hidden_dim, 1)

        # Current edge representation:
        # pooled future + node u + node v + observed edge uv
        # link_input_dim = (
        #     hidden_dim
        #     + 2 * node_hidden_dim
        #     + edge_hidden_dim
        # )

        #link_input_dim = hidden_dim + edge_hidden_dim
        link_input_dim = hidden_dim

        self.link_head = nn.Sequential(
            nn.Linear(link_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.log_var_walk = nn.Parameter(torch.zeros(1))
        self.log_var_link = nn.Parameter(torch.zeros(1))

        self.debug = debug

    @staticmethod
    def structural_info(deg_current, deg_neighbors):

        """
        Calculate the structural feature vector for every candidate edge (v, x_i)
        deg_current   : [B]
        deg_neighbors : [B,K]

        returns:
            structural : [B,K,1]
        """

        deg_current = deg_current.unsqueeze(1)  # [B,1]

        deg_current = torch.clamp(deg_current, min=1.0)
        deg_neighbors = torch.clamp(deg_neighbors, min=1.0)

        normalized_score = 1.0 / torch.sqrt(
            deg_current * deg_neighbors + 1e-8
        )

        # Individual degree information
        # new structural features: 
        # \phi(u, x_i)= [log(1-d(u))| log(1-d(x_i))| log(1/sqrt(d(u)*d(x_i)))]
        log_deg_current = torch.log1p(deg_current).expand_as(deg_neighbors)
        log_deg_neighbors = torch.log1p(deg_neighbors)
        log_normalized_score = torch.log(normalized_score  + 1e-8)

        structural = torch.stack([
            log_deg_current,
            log_deg_neighbors,
            log_normalized_score,
        ], dim=-1)

        return structural  # [B,K,3]

    def pool_walks(self, z_walk, mask):
        """
            For each candidate x_i the encoder produce:
                z_i = z_uvx_i  
            pool_walks calculate:
                a = Wz_i + b
                scores = softmax(a)
                z_pool = sum(scores_i*z_i)
        returns:
            z_pool : [B,H]
        """

        attn_logits = self.pool_attn(z_walk).squeeze(-1)  # [B,K]
        attn_logits = attn_logits.masked_fill(~mask, -1e9)

        attn = torch.softmax(attn_logits, dim=-1)         # [B,K]

        z_pool = torch.sum(
            z_walk * attn.unsqueeze(-1),
            dim=1,
        )                                                 # [B,H]
 
        return z_pool
    
    def compute_policy(
        self,
        z_walk: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
    ):
        """
        z_walk : [B, K, H]
        mask   : [B, K]

        returns:
            walk_logits : [B, K]
            policy      : [B, K]
        """

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        walk_logits = self.walk_head(z_walk).squeeze(-1)
        walk_logits = walk_logits.masked_fill(
            ~mask,
            torch.finfo(walk_logits.dtype).min,
        )

        policy = torch.softmax(walk_logits / temperature, dim=-1  )

        # Optional numerical safeguard
        policy = policy * mask.float()

        #policy = policy / (policy.sum(dim=-1, keepdim=True) + 1e-12 )

        return walk_logits, policy

    
    def encode_walks(
        self,
        previous_nodes,
        current_nodes,
        neighbor_nodes, # next_neighbors or candidates
        #previous_edge_feat,
        neighbors_edge_feat,
        neighbors_times,          
        deg_current,
        deg_neighbors, 
        mask=None
    ):
        """
        prev_node  : [B,F_n]
        curr_node   : [B,F_n]
        candidate_node_features : [B,K,F_n]

        previous_edge_features  : [B,F_e]
        candidate_edge_features : [B,K,F_e]

        previous_edge_delta_t   : [B]
        candidate_edge_delta_t  : [B,K]

        deg_current             : [B]
        deg_neighbors           : [B,K]
        mask                    : [B,K]
        """

        B, K, _ = neighbor_nodes.shape
        if previous_nodes.ndim == 3 and previous_nodes.shape[1] == 1:
            previous_nodes = previous_nodes.squeeze(1)
        if current_nodes.ndim == 3 and current_nodes.shape[1] == 1:
            current_nodes = current_nodes.squeeze(1)
        #if previous_edge_feat.ndim == 3 and previous_edge_feat.shape[1] == 1:
        #    previous_edge_feat = previous_edge_feat.squeeze(1)

        # Node representations
        f_prev = self.node_encoder(previous_nodes) # [B,D_n]
        f_curr = self.node_encoder(current_nodes) # [B,D_n]
        f_next = self.node_encoder(neighbor_nodes)

        # expand vectors
        f_prev = f_prev.unsqueeze(1).expand(-1, K, -1)
        f_curr = f_curr.unsqueeze(1).expand(-1, K, -1)

        
        # Encode edge representations previous and neighbors
        # [cos(time)+sin(time)| features]
        #e_prev = self.edge_encoder(previous_time, previous_edge_feat)  # u-v                                          
        e_next = self.edge_encoder(neighbors_times, neighbors_edge_feat) # v-x_i

        #e_prev_expanded = e_prev.unsqueeze(1).expand(-1, K, -1)

        structural = self.structural_info(
            deg_current,
            deg_neighbors,
        )                                           # [B,K,1]
        '''
        # u-v-x_i trajectory features:
        # Z = [f_u | f_v | f_x | e_uv | e_vx]
        trajectory_features = torch.cat([
            f_prev,     # previous node u
            f_curr,     # current node v
            f_next,     # candidate node x_i
            structural, # 
            e_prev_expanded,  # observed edge (u,v)
            e_next,     # candidate edge (v,x_i)
        ], dim=-1 )

        # Z = [f_u | e_uv | f_v | e_vx | f_x | structural]
        trajectory_features = torch.cat(
            [
                f_prev,          # node u
                e_prev_expanded, # edge u-v
                f_curr,          # node v
                e_next,          # edge v-x_i
                f_next,          # candidate x_i
                structural,      # structural relation v-x_i
            ],
            dim=-1,
        )
        '''
        trajectory_features = torch.cat(
            [
                f_prev,          # node u
                f_curr,          # node v
                f_next,          # candidate x_i
                structural,      # structural relation v-x_i
                e_next,          # edge v-x_i

            ],
            dim=-1,
        )

        z_walk = self.encoder(trajectory_features) # [B,K,H]

        if mask is not None:
            z_walk = z_walk.masked_fill(
                ~mask.unsqueeze(-1),
                0.0,
            )

        if self.debug:
            print("f_prev:", f_prev.shape)
            print("f_curr:", f_curr.shape)
            print("f_next:", f_next.shape)
            #print("e_prev:", e_prev.shape)
            print("e_next:", e_next.shape)
            print("trajectory:", trajectory_features.shape)
            print("z_walk:", z_walk.shape)

        return z_walk

    def forward(
        self,
        previous_node_feat,
        current_node_feat,
        neighbor_node_feat,
        #previous_edge_feat,
        neighbors_edge_feat,
        neighbors_times_delta,
        deg_current,
        deg_neighbors,
        mask):

        z_walk = self.encode_walks(
            previous_node_feat,
            current_node_feat,
            neighbor_node_feat,
            #previous_edge_feat,
            neighbors_edge_feat,
            neighbors_times_delta,
            deg_current,
            deg_neighbors,
            mask,
        )

        # Distribution 1: Auxiliary temporal walk-policy task
        # walks_logits = \phi (x_i | u, v, t) = softmax(z_walk)
        #walk_logits = self.walk_head(z_walk).squeeze(-1)
        walk_logits, policy = self.compute_policy(z_walk, mask=mask)

        # Distribution 2: Main link-prediction task Pooling Attention
        #z_pool = self.pool_walks(z_walk, mask)
        
        z_pool = torch.sum(
            z_walk * policy.unsqueeze(-1),
            dim=1,
        )   

        # Current edge representation: [z_pool | f_u | f_v | e_prev]
        #edge_repr = torch.cat([z_pool, f_prev[:, 0], f_curr[:, 0], e_prev], dim=-1)
        #edge_repr = torch.cat([z_pool, e_prev], dim=-1)
        edge_repr = z_pool

        edge_score = self.link_head(edge_repr).squeeze(-1)

        return walk_logits, edge_score
    