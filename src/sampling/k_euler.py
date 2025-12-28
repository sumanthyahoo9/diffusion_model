"""
File: src/sampling/k_euler.py
Unit Test: tests/test_sampling.py::TestKEulerSampler

K-Euler sampler: Deterministic ODE solver for diffusion models.
From Karras et al. 2022: "Elucidating the Design Space of Diffusion-Based Generative Models"

Properties:
- Deterministic (same noise → same output)
- Simple first-order ODE solver
- Fast but less accurate than higher-order methods
"""
import torch
import numpy as np
from .base_sampler import BaseSampler
from .utils import get_alphas_cumprod


class KEulerSampler(BaseSampler):
    """
    Euler method ODE solver for diffusion sampling.
    
    The Euler method is the simplest first-order numerical ODE solver:
        x_{t+Δt} = x_t + Δt * f(x_t, t)
    
    For diffusion ODEs:
        dx/dt = (x - D(x,σ)) / σ
    
    Where D(x,σ) is the denoiser (model output).
    
    Update rule:
        x_{i+1} = x_i + (σ_{i+1} - σ_i) * output_i
    
    Args:
        n_inference_steps: Number of sampling steps (default: 50)
        n_training_steps: Number of training timesteps (default: 1000)
        
    Example:
        >>> sampler = KEulerSampler(n_inference_steps=50)
        >>> latents = torch.randn(1, 4, 64, 64) * sampler.initial_scale
        >>> for _ in range(len(sampler)):
        ...     model_output = model(latents * sampler.get_input_scale(), ...)
        ...     latents = sampler.step(latents, model_output)
    """
    
    def __init__(
        self,
        n_inference_steps: int = 50,
        n_training_steps: int = 1000
    ):
        super().__init__(
            n_inference_steps=n_inference_steps,
            n_training_steps=n_training_steps,
            generator=None  # Deterministic, no generator needed
        )
        
        # Create timestep schedule
        timesteps = np.linspace(n_training_steps - 1, 0, n_inference_steps)
        
        # Get cumulative alphas
        alphas_cumprod = get_alphas_cumprod(n_training_steps=n_training_steps)
        
        # Convert to sigma parameterization: σ = √((1-ᾱ)/ᾱ)
        sigmas = np.sqrt((1 - alphas_cumprod) / alphas_cumprod)
        
        # Interpolate sigmas to match our timestep schedule
        log_sigmas = np.log(sigmas)
        log_sigmas = np.interp(timesteps, range(n_training_steps), log_sigmas)
        sigmas = np.exp(log_sigmas)
        
        # Append 0 for final step (completely denoised)
        sigmas = np.append(sigmas, 0)
        
        self.sigmas = sigmas
        self.initial_scale = sigmas.max()  # Starting noise level
        self.timesteps = timesteps
    
    def step(
        self,
        latents: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform one Euler step.
        
        Update: x_{i+1} = x_i + (σ_{i+1} - σ_i) * D_i
        
        Where:
        - x_i: Current latents
        - σ_i: Current noise level
        - D_i: Denoiser output (model prediction)
        - (σ_{i+1} - σ_i): Step size (negative, moving toward less noise)
        
        Args:
            latents: (B, C, H, W) current noisy latents
            output: (B, C, H, W) model output (denoiser prediction)
            
        Returns:
            (B, C, H, W) updated latents with reduced noise
        """
        t = self.step_count
        self.step_count += 1
        
        sigma_from = self.sigmas[t]      # Current sigma
        sigma_to = self.sigmas[t + 1]    # Next sigma (lower)
        
        # Euler step: move from sigma_from to sigma_to
        # dx = (sigma_to - sigma_from) * output
        latents = latents + output * (sigma_to - sigma_from)
        
        return latents
    
    def __repr__(self) -> str:
        return (
            f"KEulerSampler(n_inference_steps={self.n_inference_steps}, "
            f"n_training_steps={self.n_training_steps}, "
            f"sigma_range=[{self.sigmas.min():.4f}, {self.sigmas.max():.4f}])"
        )