"""
File: src/models/blocks.py
Unit Tests: tests/test_models.py::TestResidualBlock, TestAttentionBlock

Core building blocks for the U-Net architecture.
Contains ResidualBlock (ResNet-style) and AttentionBlock (cross-attention).
"""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with time conditioning.
    
    Combines spatial features with time embeddings using GroupNorm,
    convolutions, and residual connections.
    
    Args:
        in_channels (int): Input feature channels
        out_channels (int): Output feature channels
        n_time (int): Time embedding dimension (default: 1280)
        
    Forward:
        feature: (B, in_channels, H, W) - Spatial features
        time: (B, n_time) - Time embeddings
        Returns: (B, out_channels, H, W) - Processed features
    """
    def __init__(self, in_channels: int, out_channels: int, n_time: int = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Process spatial features conditioned on time.
        
        Args:
            feature: (B, C_in, H, W) spatial features
            time: (B, n_time) time embeddings
            
        Returns:
            (B, C_out, H, W) processed features with residual connection
        """
        residue = feature

        # Process spatial features
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        # Process time embeddings
        time = F.silu(time)
        time = self.linear_time(time)

        # Merge: broadcast time across spatial dimensions
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class AttentionBlock(nn.Module):
    """
    Attention block with self-attention and cross-attention.
    
    Implements:
    1. Self-attention on spatial features
    2. Cross-attention with context (e.g., text embeddings)
    3. Feed-forward network with GEGLU activation
    
    Args:
        n_heads (int): Number of attention heads
        n_embed (int): Embedding dimension per head
        d_context (int): Context dimension (default: 768 for CLIP)
        
    Forward:
        x: (B, channels, H, W) - Spatial features where channels = n_heads * n_embed
        context: (B, seq_len, d_context) - Context embeddings (e.g., text)
        Returns: (B, channels, H, W) - Attended features
    """
    def __init__(self, n_heads: int, n_embed: int, d_context: int = 768):
        super().__init__()
        channels = n_heads * n_embed

        self.group_norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layer_norm_1 = nn.LayerNorm(channels)
        # NOTE: Assumes SelfAttention and CrossAttention are defined in attention.py
        # We'll need to import these
        from attention import SelfAttention, CrossAttention
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layer_norm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_heads, channels, d_context, in_proj_bias=False)
        self.layer_norm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention, cross-attention, and FFN.
        
        Args:
            x: (B, C, H, W) spatial features
            context: (B, seq_len, d_context) conditioning context
            
        Returns:
            (B, C, H, W) attended features with residual connection
        """
        residue_long = x

        # Prepare input
        x = self.group_norm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape
        x = x.view((n, c, h * w))  # (B, C, H*W)
        x = x.transpose(-1, -2)     # (B, H*W, C)

        # Self-attention
        residue_short = x
        x = self.layer_norm_1(x)
        x = self.attention_1(x)
        x += residue_short

        # Cross-attention
        residue_short = x
        x = self.layer_norm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        # Feed-forward with GEGLU
        residue_short = x
        x = self.layer_norm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        # Reshape back to spatial
        x = x.transpose(-1, -2)     # (B, C, H*W)
        x = x.view((n, c, h, w))    # (B, C, H, W)

        return self.conv_output(x) + residue_long