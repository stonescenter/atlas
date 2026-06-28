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
    

