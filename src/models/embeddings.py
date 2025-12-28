"""
File: src/models/embeddings.py
Unit Test: tests/test_models.py::TestTimeEmbedding

Time embedding module for diffusion models.
Transforms scalar timesteps into rich vector representations.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    """
    Produces time embedding for each timestep in the diffusion process.
    
    Maps a scalar timestep to a higher-dimensional representation that
    can be injected into the U-Net via residual blocks.
    
    Args:
        n_embed (int): Base embedding dimension (default: 320)
        
    Forward:
        x: (batch_size, n_embed) - Input timestep embeddings
        Returns: (batch_size, 4*n_embed) - Transformed time embeddings
    """
    def __init__(self, n_embed: int = 320):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform timestep embedding through MLP with SiLU activation.
        
        Args:
            x: (B, n_embed) timestep features
            
        Returns:
            (B, 4*n_embed) transformed embeddings
        """
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x