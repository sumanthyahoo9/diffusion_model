"""
File: src/models/unet.py
Unit Tests: tests/test_models.py::TestUpsample, TestSwitchSequential, TestUNet

U-Net architecture for diffusion models.
Implements encoder-decoder with skip connections and multi-scale processing.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .blocks import ResidualBlock, AttentionBlock


class Upsample(nn.Module):
    """
    Upsampling block using nearest-neighbor interpolation + convolution.
    
    Doubles spatial resolution (H, W) -> (2H, 2W).
    
    Args:
        channels (int): Number of input/output channels
        
    Forward:
        x: (B, channels, H, W)
        Returns: (B, channels, 2H, 2W)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample by 2x using nearest-neighbor interpolation."""
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    """
    Sequential module with conditional routing based on layer type.
    
    Routes inputs (x, context, time) to appropriate layers:
    - AttentionBlock: receives (x, context)
    - ResidualBlock: receives (x, time)
    - Other layers: receive (x)
    
    This allows flexible composition of different block types.
    """
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditional routing.
        
        Args:
            x: (B, C, H, W) spatial features
            context: (B, seq_len, d_context) conditioning
            time: (B, n_time) time embeddings
            
        Returns:
            (B, C', H', W') processed features
        """
        for layer in self:
            if isinstance(layer, AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNet(nn.Module):
    """
    U-Net architecture for noise prediction in diffusion models.
    
    Architecture:
    - Encoder: 12 blocks with progressive downsampling (64x64 -> 8x8)
    - Bottleneck: 3 blocks at lowest resolution
    - Decoder: 12 blocks with progressive upsampling (8x8 -> 64x64)
    - Skip connections between encoder and decoder
    
    Input/Output:
        latent: (B, 4, 64, 64) - Latent representation (VAE-encoded)
        context: (B, 77, 768) - Text embeddings (CLIP)
        time: (B, 1280) - Time embeddings
        Returns: (B, 320, 64, 64) - Noise prediction
    """
    def __init__(self):
        super().__init__()
        
        # Encoder: Progressive downsampling with attention
        self.encoders = nn.ModuleList([
            # Level 1: 64x64, 320 channels
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(320, 320), AttentionBlock(8, 40)),
            
            # Downsample to 32x32
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            # Level 2: 32x32, 640 channels
            SwitchSequential(ResidualBlock(320, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(640, 640), AttentionBlock(8, 80)),
            
            # Downsample to 16x16
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            # Level 3: 16x16, 1280 channels
            SwitchSequential(ResidualBlock(640, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(1280, 1280), AttentionBlock(8, 160)),
            
            # Downsample to 8x8
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            # Level 4: 8x8, 1280 channels (no attention)
            SwitchSequential(ResidualBlock(1280, 1280)),
            SwitchSequential(ResidualBlock(1280, 1280)),
        ])

        # Bottleneck: Lowest resolution processing
        self.bottleneck = SwitchSequential(
            ResidualBlock(1280, 1280),
            AttentionBlock(8, 160),
            ResidualBlock(1280, 1280),
        )

        # Decoder: Progressive upsampling with skip connections
        self.decoders = nn.ModuleList([
            # Level 4: 8x8, concatenate skip connections (1280 + 1280 = 2560)
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280)),
            SwitchSequential(ResidualBlock(2560, 1280), Upsample(1280)),
            
            # Level 3: 16x16, with attention
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(2560, 1280), AttentionBlock(8, 160)),
            SwitchSequential(ResidualBlock(1920, 1280), AttentionBlock(8, 160), Upsample(1280)),
            
            # Level 2: 32x32
            SwitchSequential(ResidualBlock(1920, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(1280, 640), AttentionBlock(8, 80)),
            SwitchSequential(ResidualBlock(960, 640), AttentionBlock(8, 80), Upsample(640)),
            
            # Level 1: 64x64
            SwitchSequential(ResidualBlock(960, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
            SwitchSequential(ResidualBlock(640, 320), AttentionBlock(8, 40)),
        ])

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        U-Net forward pass with skip connections.
        
        Args:
            x: (B, 4, H, W) latent input
            context: (B, seq_len, 768) text conditioning
            time: (B, 1280) time embeddings
            
        Returns:
            (B, 320, H, W) processed features for final layer
        """
        # Encoder with skip connection storage
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x, context, time)

        # Decoder with skip connections (LIFO order)
        for layers in self.decoders:
            # Concatenate with corresponding encoder output
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x