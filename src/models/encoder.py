"""
File: src/models/encoder.py
Unit Test: tests/test_models.py::TestEncoder

VAE Encoder for Stable Diffusion.
Compresses images from pixel space (512x512x3) to latent space (64x64x4).

Architecture:
- Progressive downsampling: 512 → 256 → 128 → 64
- ResidualBlocks for feature extraction
- AttentionBlock at bottleneck
- Outputs latent distribution (mean, log_variance)
"""
import torch
from torch import nn
from torch.nn import functional as F
from .blocks import ResidualBlock


class EncoderAttentionBlock(nn.Module):
    """
    Attention block for encoder (simpler than U-Net attention).
    
    Uses self-attention only (no cross-attention with text).
    
    Args:
        channels: Number of input/output channels
        
    Forward:
        x: (B, C, H, W) spatial features
        Returns: (B, C, H, W) attended features
    """
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=1,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention to spatial features.
        
        Args:
            x: (B, C, H, W) input features
            
        Returns:
            (B, C, H, W) attended features
        """
        residue = x
        B, C, H, W = x.shape
        
        # Normalize
        x = self.groupnorm(x)
        
        # Reshape to sequence: (B, C, H, W) → (B, H*W, C)
        x = x.view(B, C, H * W)
        x = x.transpose(1, 2)  # (B, H*W, C)
        
        # Self-attention
        x, _ = self.attention(x, x, x, need_weights=False)
        
        # Reshape back: (B, H*W, C) → (B, C, H, W)
        x = x.transpose(1, 2)
        x = x.view(B, C, H, W)
        
        return x + residue


class Encoder(nn.Module):
    """
    VAE Encoder for Stable Diffusion.
    
    Compresses RGB images to latent representations:
    - Input: (B, 3, 512, 512) RGB images in [-1, 1]
    - Output: (B, 4, 64, 64) latent codes
    
    Architecture:
    1. Initial conv: 3 → 128 channels
    2. Level 1 (512x512): ResBlocks + downsample → 256x256
    3. Level 2 (256x256): ResBlocks + downsample → 128x128
    4. Level 3 (128x128): ResBlocks + downsample → 64x64
    5. Bottleneck (64x64): ResBlocks + Attention
    6. Output: Conv to 8 channels (mean + log_var)
    7. Reparameterization: Sample from N(mean, var)
    
    The encoder learns a probabilistic encoding (VAE):
    - First 4 channels: mean (μ)
    - Last 4 channels: log variance (log σ²)
    - Sample: z = μ + σ * ε, where ε ~ N(0, I)
    
    Example:
        >>> encoder = Encoder()
        >>> img = torch.randn(2, 3, 512, 512)  # 2 images
        >>> noise = torch.randn(2, 4, 64, 64)  # For reparameterization
        >>> latent = encoder(img, noise)
        >>> latent.shape
        torch.Size([2, 4, 64, 64])
    """
    
    def __init__(self):
        super().__init__()
        
        # Build encoder layers
        self.layers = nn.Sequential(
            # Initial convolution: (B, 3, 512, 512) → (B, 128, 512, 512)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # Level 1: 512x512, 128 channels
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            
            # Downsample to 256x256 (asymmetric padding handled in forward)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # Level 2: 256x256, 256 channels
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            
            # Downsample to 128x128
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            
            # Level 3: 128x128, 512 channels
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            
            # Downsample to 64x64
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
            # Bottleneck: 64x64, 512 channels
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            
            # Self-attention at bottleneck
            EncoderAttentionBlock(512),
            
            # Final residual block
            ResidualBlock(512, 512),
            
            # Normalization and activation
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            
            # Output projection: 512 → 8 channels (mean + log_var)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            
            # Final 1x1 conv (no change in spatial dims)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
        
        # Scaling factor for KL regularization
        # This matches Stable Diffusion's scaling
        self.scaling_factor = 0.18215
    
    def forward(
        self,
        x: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode image to latent representation.
        
        Args:
            x: (B, 3, H, W) RGB image in [-1, 1], typically (B, 3, 512, 512)
            noise: (B, 4, H/8, W/8) Gaussian noise for reparameterization
            
        Returns:
            (B, 4, H/8, W/8) latent representation, typically (B, 4, 64, 64)
            
        Process:
        1. Progressive downsampling through conv layers
        2. Split output into mean and log_variance
        3. Reparameterization trick: z = μ + σ * ε
        4. Scale by 0.18215 (Stable Diffusion convention)
        """
        # Process through encoder layers
        for module in self.layers:
            # Apply asymmetric padding for stride=2 convs
            if getattr(module, 'stride', None) == (2, 2):
                # Pad (left, right, top, bottom) = (0, 1, 0, 1)
                # This ensures correct output size after stride=2
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        
        # x shape: (B, 8, 64, 64)
        
        # Split into mean and log_variance
        # First 4 channels: mean (μ)
        # Last 4 channels: log variance (log σ²)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        # Clamp log_variance for numerical stability
        # Prevents extreme values that could cause NaN
        log_variance = torch.clamp(log_variance, -30, 20)
        
        # Compute standard deviation: σ = exp(log σ² / 2) = exp(log σ²)^0.5
        variance = log_variance.exp()
        std_dev = variance.sqrt()
        
        # Reparameterization trick: z = μ + σ * ε
        # This allows backprop through sampling
        x = mean + std_dev * noise
        
        # Scale by constant factor
        # This is empirically chosen for Stable Diffusion
        x = x * self.scaling_factor
        
        return x
    
    def encode_without_sampling(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image and return mean and log_variance.
        
        Useful for deterministic encoding (just use mean).
        
        Args:
            x: (B, 3, H, W) RGB image
            
        Returns:
            Tuple of (mean, log_variance), each (B, 4, H/8, W/8)
        """
        # Process through encoder
        for module in self.layers:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # Split and clamp
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        
        # Scale mean
        mean = mean * self.scaling_factor
        
        return mean, log_variance