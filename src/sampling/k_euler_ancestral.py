"""
File: src/sampling/k_euler_ancestral.py
Unit Test: tests/test_sampling.py::TestKEulerAncestralSampler

K-Euler-Ancestral: Stochastic variant of K-Euler sampler.
From Karras et al. 2022.

Properties:
- Stochastic (adds noise at each step, like DDPM)
- Better sample diversity than deterministic K-Euler
- Slightly slower convergence but higher quality
"""
from typing import Optional
import torch
import numpy as np
from .base_sampler import BaseSampler
from .utils import get_alphas_cumprod


class KEulerAncestralSampler(BaseSampler):
    """
    Euler-Ancestral method: Stochastic ODE solver with noise injection.
    
    This combines Euler ODE solving with ancestral sampling (noise addition),
    similar to DDPM but using the sigma parameterization.
    
    Update rule (two steps):
    1. Euler step: x' = x + (σ_down - σ_from) * D(x, σ)
    2. Add noise:  x_new = x' + σ_up * ε
    
    Where:
    - σ_down = σ_to² / σ_from (denoising component)
    - σ_up = σ_to * √(1 - σ_to²/σ_from²) (noise component)
    - ε ~ N(0, I)
    
    Args:
        n_inference_steps: Number of sampling steps
        n_training_steps: Number of training timesteps
        generator: Random generator for reproducible noise
        
    Example:
        >>> gen = torch.Generator().manual_seed(42)
        >>> sampler = KEulerAncestralSampler(n_inference_steps=50, generator=gen)
        >>> latents = torch.randn(1, 4, 64, 64) * sampler.initial_scale
        >>> for _ in range(len(sampler)):
        ...     scaled = latents * sampler.get_input_scale()
        ...     model_output = model(scaled, ...)
        ...     latents = sampler.step(latents, model_output)
    """
    
    def __init__(
        self,
        n_inference_steps: int = 50,
        n_training_steps: int = 1000,
        generator: Optional[torch.Generator] = None
    ):
        super().__init__(
            n_inference_steps=n_inference_steps,
            n_training_steps=n_training_steps,
            generator=generator
        )
        
        # Create timestep schedule
        timesteps = np.linspace(n_training_steps - 1, 0, n_inference_steps)
        
        # Get cumulative alphas and convert to sigmas
        alphas_cumprod = get_alphas_cumprod(n_training_steps=n_training_steps)
        sigmas = np.sqrt((1 - alphas_cumprod) / alphas_cumprod)
        
        # Interpolate sigmas
        log_sigmas = np.log(sigmas)
        log_sigmas = np.interp(timesteps, range(n_training_steps), log_sigmas)
        sigmas = np.exp(log_sigmas)
        sigmas = np.append(sigmas, 0)
        
        self.sigmas = sigmas
        self.initial_scale = sigmas.max()
        self.timesteps = timesteps
    
    def step(
        self,
        latents: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform one Euler-Ancestral step (denoise + add noise).
        
        Two-stage update:
        1. Denoise using Euler step
        2. Add fresh noise (ancestral sampling)
        
        This balances deterministic denoising with stochastic exploration.
        
        Args:
            latents: (B, C, H, W) current noisy latents
            output: (B, C, H, W) model output (denoiser)
            
        Returns:
            (B, C, H, W) updated latents
        """
        t = self.step_count
        self.step_count += 1
        
        sigma_from = self.sigmas[t]      # Current sigma
        sigma_to = self.sigmas[t + 1]    # Target sigma
        
        # Compute sigma schedule for ancestral sampling
        # σ_up: Amount of noise to add
        # σ_down: Effective denoising sigma
        sigma_up = sigma_to * np.sqrt(1 - (sigma_to ** 2 / sigma_from ** 2))
        sigma_down = np.sqrt(sigma_to ** 2 - sigma_up ** 2)
        
        # Step 1: Euler denoising step
        # Move from sigma_from to sigma_down
        latents = latents + output * (sigma_down - sigma_from)
        
        # Step 2: Add noise (ancestral sampling)
        # This maintains sample diversity
        noise = torch.randn(
            latents.shape,
            generator=self.generator,
            device=latents.device,
            dtype=latents.dtype
        )
        latents = latents + noise * sigma_up
        
        return latents
    
    def __repr__(self) -> str:
        return (
            f"KEulerAncestralSampler("
            f"n_inference_steps={self.n_inference_steps}, "
            f"stochastic=True, "
            f"sigma_range=[{self.sigmas.min():.4f}, {self.sigmas.max():.4f}])"
        )