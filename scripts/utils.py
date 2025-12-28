"""
File: scripts/utils.py
Unit Test: tests/test_scripts.py::TestScriptUtils

Utility functions for inference and training pipelines.
Includes image processing, timestep embeddings, and tensor operations.
"""
import os
from typing import Tuple
import torch
import numpy as np
from PIL import Image


def rescale(
    tensor: torch.Tensor,
    old_range: Tuple[float, float],
    new_range: Tuple[float, float],
    clamp: bool = False
) -> torch.Tensor:
    """
    Rescale tensor values from old range to new range.
    
    Common use cases:
    - Image preprocessing: [0, 255] → [-1, 1]
    - Image postprocessing: [-1, 1] → [0, 255]
    
    Args:
        tensor: Input tensor
        old_range: (min, max) of current range
        new_range: (min, max) of target range
        clamp: Whether to clamp values to new_range
        
    Returns:
        Rescaled tensor
        
    Example:
        >>> img = torch.randint(0, 256, (3, 224, 224))
        >>> img_normalized = rescale(img, (0, 255), (-1, 1))
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    
    # Normalize to [0, 1]
    tensor = (tensor - old_min) / (old_max - old_min)
    
    # Scale to new range
    tensor = tensor * (new_max - new_min) + new_min
    
    if clamp:
        tensor = torch.clamp(tensor, new_min, new_max)
    
    return tensor


def move_channel(
    tensor: torch.Tensor,
    to: str = "first"
) -> torch.Tensor:
    """
    Move channel dimension in image tensor.
    
    PyTorch convention: (B, C, H, W)
    NumPy/PIL convention: (B, H, W, C)
    
    Args:
        tensor: Image tensor
        to: "first" for (B,H,W,C)→(B,C,H,W) or "last" for reverse
        
    Returns:
        Tensor with moved channels
        
    Example:
        >>> img_hwc = torch.randn(1, 224, 224, 3)  # NumPy format
        >>> img_chw = move_channel(img_hwc, to="first")  # PyTorch format
        >>> img_chw.shape
        torch.Size([1, 3, 224, 224])
    """
    if to == "first":
        # (B, H, W, C) → (B, C, H, W)
        if tensor.ndim == 4:
            return tensor.permute(0, 3, 1, 2)
        elif tensor.ndim == 3:
            return tensor.permute(2, 0, 1)
    elif to == "last":
        # (B, C, H, W) → (B, H, W, C)
        if tensor.ndim == 4:
            return tensor.permute(0, 2, 3, 1)
        elif tensor.ndim == 3:
            return tensor.permute(1, 2, 0)
    else:
        raise ValueError(f"to must be 'first' or 'last', got {to}")
    
    return tensor


def get_time_embedding(
    timestep: int,
    embedding_dim: int = 320,
    dtype: torch.dtype = torch.float32,
    max_period: int = 10000
) -> torch.Tensor:
    """
    Create sinusoidal timestep embedding.
    
    Standard positional encoding from "Attention is All You Need".
    Used to inject timestep information into diffusion models.
    
    Args:
        timestep: Timestep value
        embedding_dim: Embedding dimension (default: 320)
        dtype: Data type for embedding
        max_period: Maximum period for sinusoids
        
    Returns:
        (embedding_dim,) timestep embedding
        
    Example:
        >>> t_emb = get_time_embedding(timestep=500, embedding_dim=320)
        >>> t_emb.shape
        torch.Size([320])
    """
    half_dim = embedding_dim // 2
    
    # Compute frequencies
    freqs = torch.exp(
        -np.log(max_period) *
        torch.arange(0, half_dim, dtype=torch.float32) / half_dim
    )
    
    # Compute sin/cos embeddings
    args = timestep * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)])
    
    # Handle odd embedding_dim
    if embedding_dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros(1)])
    
    return embedding.to(dtype)


def get_alphas_cumprod(
    n_training_steps: int = 1000,
    beta_start: float = 0.00085,
    beta_end: float = 0.012
) -> np.ndarray:
    """
    Compute cumulative product of alphas for noise schedule.
    
    Used by samplers for sigma computation.
    
    Args:
        n_training_steps: Number of diffusion steps
        beta_start: Starting noise variance
        beta_end: Ending noise variance
        
    Returns:
        (n_training_steps,) array of ᾱ_t values
    """
    # Linear beta schedule
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return alphas_cumprod


def prepare_image_tensor(
    image,
    width: int,
    height: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Prepare PIL Image or path for model input.
    
    Steps:
    1. Load image (if path provided)
    2. Resize to (width, height)
    3. Convert to tensor
    4. Normalize to [-1, 1]
    5. Convert to (C, H, W) format
    
    Args:
        image: PIL Image or path string
        width: Target width
        height: Target height
        dtype: Tensor dtype
        device: Target device
        
    Returns:
        (C, H, W) normalized image tensor
    """
    
    # Load if path
    if isinstance(image, str):
        image = Image.open(image)
    
    # Resize
    image = image.resize((width, height))
    
    # Convert to array
    image = np.array(image)
    
    # To tensor
    image = torch.tensor(image, dtype=dtype, device=device)
    
    # Normalize [0, 255] → [-1, 1]
    image = rescale(image, (0, 255), (-1, 1))
    
    # Move channels to first: (H, W, C) → (C, H, W)
    image = move_channel(image, to="first")
    
    return image


def save_images(
    images: torch.Tensor,
    output_dir: str = "outputs",
    prefix: str = "sample"
):
    """
    Save batch of images to disk.
    
    Args:
        images: (B, C, H, W) tensor in [-1, 1] range
        output_dir: Directory to save images
        prefix: Filename prefix
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Rescale to [0, 255]
    images = rescale(images, (-1, 1), (0, 255), clamp=True)
    
    # Move channels: (B, C, H, W) → (B, H, W, C)
    images = move_channel(images, to="last")
    
    # Convert to numpy
    images = images.cpu().to(torch.uint8).numpy()
    
    # Save each image
    for i, img in enumerate(images):
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(output_dir, f"{prefix}_{i:04d}.png"))
    
    print(f"Saved {len(images)} images to {output_dir}/")