"""
File: src/models/diffusion.py
Unit Tests: tests/test_models.py::TestFinalLayer, TestDiffusion

Main diffusion model combining all components.
Entry point for training and sampling.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .embeddings import TimeEmbedding
from .unet import UNet


class FinalLayer(nn.Module):
    """
    Final output layer for the diffusion model.
    
    Converts U-Net features to noise prediction in latent space.
    
    Args:
        in_channels (int): Input feature channels (default: 320)
        out_channels (int): Output channels (default: 4 for VAE latent)
        
    Forward:
        x: (B, in_channels, H, W)
        Returns: (B, out_channels, H, W) - Predicted noise
    """
    def __init__(self, in_channels: int = 320, out_channels: int = 4):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization, activation, and final convolution.
        
        Args:
            x: (B, C_in, H, W) U-Net output features
            
        Returns:
            (B, C_out, H, W) predicted noise
        """
        x = self.group_norm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x


class Diffusion(nn.Module):
    """
    Complete diffusion model for noise prediction.
    
    Combines:
    1. TimeEmbedding: Converts timesteps to embeddings
    2. UNet: Processes latent + context with time conditioning
    3. FinalLayer: Converts features to noise prediction
    
    This is the main model used during training and sampling.
    
    Forward:
        latent: (B, 4, 64, 64) - VAE-encoded image
        context: (B, 77, 768) - CLIP text embeddings
        time: (B, 320) - Raw timestep features (NOT embedded yet)
        Returns: (B, 4, 64, 64) - Predicted noise ε_φ(x_t, t)
    """
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = FinalLayer(320, 4)

    def forward(
        self, 
        latent: torch.Tensor, 
        context: torch.Tensor, 
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise given noisy latent, text context, and timestep.
        
        This implements ε_φ(x_t, c, t) from DDPM.
        
        Args:
            latent: (B, 4, H, W) noisy latent at timestep t
            context: (B, seq_len, 768) text conditioning
            time: (B, 320) timestep features
            
        Returns:
            (B, 4, H, W) predicted noise ε
        """
        # Embed timestep: (B, 320) -> (B, 1280)
        time = self.time_embedding(time)
        
        # Process through U-Net: (B, 4, H, W) -> (B, 320, H, W)
        output = self.unet(latent, context, time)
        
        # Final layer: (B, 320, H, W) -> (B, 4, H, W)
        output = self.final(output)
        
        return output