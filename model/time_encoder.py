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
