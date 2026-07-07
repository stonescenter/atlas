import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .time_encoder import TimeEncoder

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
    

class TemporalWalkModel(nn.Module):
    def __init__(
        self,
        num_nodes,
        embedding_dim=64,
        time_dim=32,
        hidden_dim=128,
        pad_node=None,
        dropout=0.1,
        debug=False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_nodes + 1,
            embedding_dim,
            padding_idx=pad_node,
        )

        self.time_encoder = TimeEncoder(time_dim)

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

        # Link prediction head: predicts whether edge exists
        self.link_head = nn.Sequential(
            nn.Linear(hidden_dim + 2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

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
                #prev_next,
                #curr_next,
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

        edge_repr = torch.cat(
            [
                z_pool,
                h_prev,
                h_curr,
            ],
            dim=-1,
        )

        edge_score = self.link_head(edge_repr).squeeze(-1)  # [B]

        return walk_logits, edge_score

   