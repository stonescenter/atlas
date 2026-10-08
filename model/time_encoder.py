import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimeEncoding(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.freq = nn.Parameter(torch.randn(dim))

    def forward(self, t):

        t = t.unsqueeze(-1)

        return torch.sin(t * self.freq)


class TimeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)

    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                       .float().reshape(dimension, -1))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    t = t.unsqueeze(dim=2)

    # output has shape [batch_size, seq_len, dimension]
    output = torch.cos(self.w(t))

    return output
 
class TimeEncoder(nn.Module):
    # Improved version Time Encoding proposed by TGAT
    def __init__(self, dimension):

        super().__init__()

        self.dimension = dimension
        self.w = nn.Linear(1, dimension)

        freq = 1 / 10 ** np.linspace(0, 9, self.dimension, dtype=np.float32)

        #self.w.weight = nn.Parameter(torch.from_numpy(freq).float() .reshape(dimension, 1))
        self.w.weight = nn.Parameter(torch.from_numpy(freq).float() .reshape(dimension, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dimension))

    def forward(self, t):
        #if torch.isnan(t).any():
        #    print("NaN before encoding")

        #print(
        #    "time min:", t.min().item(),
        #    "time max:", t.max().item()
        #)

        t = torch.log1p(t)
        t = t.unsqueeze(-1)
        x = self.w(t)

        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    
# ============================================================
# Harmonic Time Encoding
# ============================================================

class HarmonicTimeEncoder(nn.Module):
    """
    TGAT-style harmonic temporal encoding.

    phi(dt) =
        [ cos(w_i * dt + b_i),
          sin(w_i * dt + b_i) ]
    """

    def __init__(
        self,
        time_dim=32,
        max_period=10000.0,
    ):
        super().__init__()

        self.time_dim = time_dim

        # --------------------------------------------
        # Frequencies initialized log-uniformly
        # --------------------------------------------

        freq = torch.logspace(
            start=0,
            end=math.log10(max_period),
            steps=time_dim,
        )

        self.freq = nn.Parameter(
            1.0 / freq
        )

        self.phase = nn.Parameter(
            torch.zeros(time_dim)
        )

    def forward(self, delta_t):
        """
        delta_t:
            shape [N]

        returns:
            shape [N, 2*time_dim]
        """

        # --------------------------------------------
        # Log normalization
        # --------------------------------------------

        delta_t = torch.log1p(delta_t)

        # [N,1]
        delta_t = delta_t.unsqueeze(-1)

        # --------------------------------------------
        # Harmonic projection
        # --------------------------------------------

        angles = (
            delta_t * self.freq
            + self.phase
        )

        harmonic = torch.cat(
            [
                torch.cos(angles),
                torch.sin(angles),
            ],
            dim=-1
        )

        return harmonic

"""
Module: GraphMixer Time-encoder
"""

class GrapMixerTimeEncoder(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim):
        super(GrapMixerTimeEncoder, self).__init__()
        self.dim = dim
        self.dimension = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.float().unsqueeze(-1)))
        return output

    
class RawLearnableTimeEncoder(nn.Module):
    """TGAT-style learnable harmonic encoder without log1p normalization."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        frequency = 1.0 / (10.0 ** np.linspace(0, 9, dimension))
        self.w.weight = nn.Parameter(
            torch.from_numpy(frequency).float().reshape(dimension, 1)
        )
        self.w.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        delta_t = delta_t.float().clamp_min(0.0).unsqueeze(-1)
        angle = self.w(delta_t)
        return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)


class FixedLogTimeEncoder(nn.Module):
    """Fixed harmonic frequencies applied after log1p time normalization."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        frequency = 1.0 / (10.0 ** torch.linspace(0, 9, dimension))
        self.register_buffer("frequency", frequency)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        tau = torch.log1p(delta_t.float().clamp_min(0.0)).unsqueeze(-1)
        angle = tau * self.frequency
        return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)


class HybridTimeEncoder(nn.Module):
    """Stable fixed basis plus a bounded learnable residual."""

    def __init__(self, dimension: int, residual_init: float = 0.01):
        super().__init__()
        self.dimension = dimension
        frequency = 1.0 / (10.0 ** torch.linspace(0, 9, dimension))
        self.register_buffer("fixed_frequency", frequency)
        self.frequency_delta = nn.Parameter(torch.zeros(dimension))
        self.phase = nn.Parameter(torch.zeros(dimension))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        tau = torch.log1p(delta_t.float().clamp_min(0.0)).unsqueeze(-1)
        fixed_angle = tau * self.fixed_frequency
        # Multiplicative adjustment keeps frequencies positive and near init.
        learned_frequency = self.fixed_frequency * torch.exp(
            self.frequency_delta.clamp(-3.0, 3.0)
        )
        learned_angle = tau * learned_frequency + self.phase

        fixed = torch.cat(
            [torch.cos(fixed_angle), torch.sin(fixed_angle)], dim=-1
        )
        residual = torch.cat(
            [torch.cos(learned_angle), torch.sin(learned_angle)], dim=-1
        )
        return fixed + torch.tanh(self.residual_scale) * residual


def build_time_encoder(name: str, dimension: int) -> nn.Module:
    name = name.lower()
    if name == "atlas":
        return TimeEncoder(dimension)
    if name == "raw_learnable":
        return RawLearnableTimeEncoder(dimension)
    if name == "fixed_log":
        return FixedLogTimeEncoder(dimension)
    if name == "hybrid":
        return HybridTimeEncoder(dimension)
    if name == "graphmixer":
        return GrapMixerTimeEncoder(dimension)
    raise ValueError(
        f"Unknown encoder '{name}'. Expected atlas, raw_learnable, "
        "fixed_log, hybrid or graphmixer."
    )
