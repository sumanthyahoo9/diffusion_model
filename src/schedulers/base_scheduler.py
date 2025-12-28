"""
File: src/schedulers/base_scheduler.py
Unit Test: tests/test_schedulers.py::TestBaseScheduler

Abstract base class for noise schedulers in diffusion models.
Provides common interface and utilities for DDPM, DDIM, and other schedulers.
"""
from abc import ABC, abstractmethod
import torch
import numpy as np


class BaseScheduler(ABC):
    """
    Abstract base class for diffusion schedulers.
    
    A scheduler defines:
    1. Noise schedule (how much noise at each timestep)
    2. Forward process (adding noise)
    3. Reverse process (removing noise)
    
    Subclasses must implement:
    - step(): Single denoising step
    - _get_variance_schedule(): Define β_t schedule
    
    Common notation:
    - β_t: Noise schedule (variance at timestep t)
    - α_t = 1 - β_t
    - ᾱ_t = ∏_{i=1}^t α_i (cumulative product)
    - σ_t = √(1 - ᾱ_t²) (noise std at timestep t)
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        device: str = "cpu"
    ):
        """
        Initialize base scheduler.
        
        Args:
            num_train_timesteps: Number of diffusion steps (T)
            beta_start: Starting noise variance
            beta_end: Ending noise variance
            beta_schedule: "linear" or "cosine"
            device: Device to store tensors
        """
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.device = device
        
        # Compute noise schedule
        self.betas = self._get_variance_schedule()
        
        # Compute α_t = 1 - β_t
        self.alphas = 1.0 - self.betas
        
        # Compute ᾱ_t = ∏_{i=1}^t α_i (cumulative product)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Useful for various computations
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device), 
            self.alphas_cumprod[:-1]
        ])
        
        # Square root of ᾱ_t (for forward process)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        
        # Square root of (1 - ᾱ_t) (noise scale)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For reverse process variance computation
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance: β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _get_variance_schedule(self) -> torch.Tensor:
        """
        Compute the β_t schedule.
        
        Returns:
            (num_train_timesteps,) tensor of β values
        """
        if self.beta_schedule == "linear":
            return self._linear_beta_schedule()
        elif self.beta_schedule == "cosine":
            return self._cosine_beta_schedule()
        else:
            raise ValueError(f"Unknown beta_schedule: {self.beta_schedule}")
    
    def _linear_beta_schedule(self) -> torch.Tensor:
        """
        Linear noise schedule: β_t increases linearly from β_start to β_end.
        
        Original DDPM uses this schedule.
        
        Returns:
            (T,) tensor of β values
        """
        return torch.linspace(
            self.beta_start,
            self.beta_end,
            self.num_train_timesteps,
            device=self.device
        )
    
    def _cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        """
        Cosine noise schedule (Improved DDPM, Nichol & Dhariwal 2021).
        
        Provides better sample quality by having smaller noise at endpoints.
        
        Formula:
            ᾱ_t = cos²((t/T + s)/(1 + s) * π/2)
            β_t = 1 - α_t = 1 - ᾱ_t/ᾱ_{t-1}
        
        Args:
            s: Small offset to prevent β_t from being too small near t=0
            
        Returns:
            (T,) tensor of β values
        """
        timesteps = self.num_train_timesteps
        t = torch.linspace(0, timesteps, timesteps + 1, device=self.device)
        
        # Compute ᾱ_t using cosine schedule
        alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize
        
        # Compute β_t from ᾱ_t
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        # Clip to prevent numerical issues
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to samples (forward diffusion process).
        
        Implements: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        
        This is the "direct perturbation kernel" from DDPM - we can
        jump directly to any timestep without iterating.
        
        Args:
            original_samples: (B, C, H, W) clean images/latents
            noise: (B, C, H, W) Gaussian noise ε ~ N(0, I)
            timesteps: (B,) timestep indices [0, T-1]
            
        Returns:
            (B, C, H, W) noisy samples at specified timesteps
        """
        # Extract ᾱ_t for each timestep in batch
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting: (B,) -> (B, 1, 1, 1)
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        # x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        noisy_samples = (
            sqrt_alpha_prod * original_samples +
            sqrt_one_minus_alpha_prod * noise
        )
        
        return noisy_samples
    
    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity for flow matching perspective.
        
        Velocity: v_t = √ᾱ_t * ε - √(1-ᾱ_t) * x_0
        
        This is used in some recent formulations (e.g., v-prediction).
        
        Args:
            sample: (B, C, H, W) clean samples x_0
            noise: (B, C, H, W) noise ε
            timesteps: (B,) timesteps
            
        Returns:
            (B, C, H, W) velocity targets
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity
    
    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Perform one denoising step (reverse diffusion).
        
        This is the core sampling method that must be implemented by subclasses.
        
        Args:
            model_output: Predicted noise ε_θ(x_t, t) from model
            timestep: Current timestep t
            sample: Current noisy sample x_t
            **kwargs: Scheduler-specific arguments
            
        Returns:
            Denoised sample x_{t-1}
        """
        pass
    
    def __len__(self) -> int:
        """Return number of timesteps."""
        return self.num_train_timesteps