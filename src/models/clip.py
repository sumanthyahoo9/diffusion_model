"""
File: src/models/clip.py
Unit Test: tests/test_models.py::TestCLIP

CLIP (Contrastive Language-Image Pre-training) text encoder.
Used to encode text prompts into embeddings for diffusion guidance.

Based on OpenAI's CLIP architecture (Radford et al. 2021).
"""
import torch
from torch import nn
from .attention import SelfAttention


class CLIPEmbedding(nn.Module):
    """
    CLIP text embedding layer.
    
    Combines token embeddings with learned positional embeddings.
    
    Args:
        n_vocab: Vocabulary size (default: 49408 for CLIP)
        n_embed: Embedding dimension (default: 768)
        n_tokens: Maximum sequence length (default: 77 for CLIP)
        
    Forward:
        tokens: (B, seq_len) token IDs
        Returns: (B, seq_len, n_embed) embedded tokens
    """
    def __init__(
        self,
        n_vocab: int = 49408,
        n_embed: int = 768,
        n_tokens: int = 77
    ):
        super().__init__()
        # Token embeddings (learned from vocabulary)
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        
        # Positional embeddings (learned, not sinusoidal)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Embed tokens with positional information.
        
        Args:
            tokens: (B, seq_len) token indices
            
        Returns:
            (B, seq_len, n_embed) embedded tokens
        """
        # Get token embeddings
        x = self.token_embedding(tokens)  # (B, seq_len, n_embed)
        
        # Add positional embeddings (broadcast across batch)
        x = x + self.position_embedding  # (B, seq_len, n_embed)
        
        return x


class CLIPLayer(nn.Module):
    """
    Single CLIP transformer layer.
    
    Architecture:
    1. Layer norm + self-attention (with causal mask)
    2. Residual connection
    3. Layer norm + feed-forward network (with QuickGELU)
    4. Residual connection
    
    Args:
        n_heads: Number of attention heads
        n_embed: Embedding dimension
        
    Forward:
        x: (B, seq_len, n_embed) input
        Returns: (B, seq_len, n_embed) output
    """
    def __init__(self, n_heads: int, n_embed: int):
        super().__init__()
        # Pre-norm architecture (norm before attention/FFN)
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_heads, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        
        # Feed-forward network (expand 4x then project back)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through transformer layer.
        
        Args:
            x: (B, seq_len, n_embed) input features
            
        Returns:
            (B, seq_len, n_embed) processed features
        """
        # Self-attention block with residual
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)  # Causal for autoregressive
        x = x + residue
        
        # Feed-forward block with residual
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        
        # QuickGELU activation: x * σ(1.702 * x)
        # Faster approximation of GELU used in CLIP
        x = x * torch.sigmoid(1.702 * x)
        
        x = self.linear_2(x)
        x = x + residue
        
        return x


class CLIP(nn.Module):
    """
    CLIP text encoder (transformer-based).
    
    Architecture:
    - Embedding layer (token + positional)
    - 12 transformer layers
    - Final layer norm
    
    Input: Text token IDs
    Output: Text embeddings for diffusion conditioning
    
    Default config matches OpenAI's CLIP ViT-L/14 text encoder:
    - Vocab size: 49408
    - Embedding dim: 768
    - Layers: 12
    - Heads: 12
    - Max length: 77 tokens
    
    Example:
        >>> clip = CLIP()
        >>> tokens = torch.randint(0, 49408, (2, 77))  # 2 prompts
        >>> embeddings = clip(tokens)  # (2, 77, 768)
    """
    def __init__(
        self,
        n_vocab: int = 49408,
        n_embed: int = 768,
        n_tokens: int = 77,
        n_layers: int = 12,
        n_heads: int = 12
    ):
        super().__init__()
        
        # Embedding layer
        self.embedding = CLIPEmbedding(n_vocab, n_embed, n_tokens)
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            CLIPLayer(n_heads, n_embed) for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.layernorm = nn.LayerNorm(n_embed)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Encode text tokens to embeddings.
        
        Args:
            tokens: (B, seq_len) token IDs, typically (B, 77)
            
        Returns:
            (B, seq_len, n_embed) text embeddings, typically (B, 77, 768)
            
        Note:
            For Stable Diffusion, the output is used as cross-attention
            context in the U-Net. The model attends to all token positions.
        """
        # Ensure tokens are long integers
        tokens = tokens.type(torch.long)
        
        # Embed tokens
        state = self.embedding(tokens)  # (B, seq_len, n_embed)
        
        # Process through transformer layers
        for layer in self.layers:
            state = layer(state)
        
        # Final normalization
        output = self.layernorm(state)
        
        return output
    
    def encode_text(self, text_tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Convenience method for text encoding.
        
        Alias for forward() with clearer naming.
        """
        return self.forward(text_tokens)