"""
File: src/sampling/k_lms.py
Unit Test: tests/test_sampling.py::TestKLMSSampler

K-LMS: Linear Multi-Step sampler for diffusion models.
From Karras et al. 2022.

Properties:
- Higher-order method (uses history of past outputs)
- More accurate than Euler with same number of steps
- Default order=4 (uses last 4 outputs)
- Deterministic
"""
from typing import List
import torch
import numpy as np
from .base_sampler import BaseSampler
from .utils import get_alphas_cumprod


class KLMSSampler(BaseSampler):
    """
    Linear Multi-Step (LMS) sampler.
    
    LMS methods use polynomial interpolation of past derivative estimates
    to achieve higher accuracy than simple Euler methods.
    
    Instead of: x_{i+1} = x_i + h * f(x_i, t_i)  [Euler]
    
    LMS uses: x_{i+1} = x_i + ∫ P(t) dt
    
    Where P(t) is a polynomial interpolating f(x_j, t_j) for j = i, i-1, ..., i-k+1
    
    Order 1 = Euler
    Order 2 = Adams-Bashforth 2
    Order 4 = Standard LMS (good accuracy/speed tradeoff)
    
    Args:
        n_inference_steps: Number of sampling steps
        n_training_steps: Number of training timesteps
        lms_order: Order of LMS method (default: 4)
        
    Example:
        >>> sampler = KLMSSampler(n_inference_steps=50, lms_order=4)
        >>> latents = torch.randn(1, 4, 64, 64) * sampler.initial_scale
        >>> for _ in range(len(sampler)):
        ...     model_output = model(latents * sampler.get_input_scale(), ...)
        ...     latents = sampler.step(latents, model_output)
    """
    
    def __init__(
        self,
        n_inference_steps: int = 50,
        n_training_steps: int = 1000,
        lms_order: int = 4
    ):
        super().__init__(
            n_inference_steps=n_inference_steps,
            n_training_steps=n_training_steps,
            generator=None  # Deterministic
        )
        
        self.lms_order = lms_order
        self.outputs: List[torch.Tensor] = []  # Store past model outputs
        
        # Create timestep and sigma schedule (same as K-Euler)
        timesteps = np.linspace(n_training_steps - 1, 0, n_inference_steps)
        
        alphas_cumprod = get_alphas_cumprod(n_training_steps=n_training_steps)
        sigmas = np.sqrt((1 - alphas_cumprod) / alphas_cumprod)
        
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
        Perform one LMS step using polynomial interpolation.
        
        Algorithm:
        1. Store current output in history
        2. For each past output, compute Lagrange polynomial coefficient
        3. Integrate polynomial over [σ_i, σ_{i+1}] using trapezoidal rule
        4. Sum weighted contributions
        
        The order increases from 1 to lms_order as we accumulate history.
        
        Args:
            latents: (B, C, H, W) current latents
            output: (B, C, H, W) model output (denoiser prediction)
            
        Returns:
            (B, C, H, W) updated latents
        """
        t = self.step_count
        self.step_count += 1
        
        # Add current output to history (most recent first)
        self.outputs = [output] + self.outputs[:self.lms_order - 1]
        order = len(self.outputs)  # Effective order (starts at 1, grows to lms_order)
        
        # Integrate using polynomial interpolation
        # We'll accumulate contributions from each output in history
        result = torch.zeros_like(latents)
        
        for i, past_output in enumerate(self.outputs):
            # Compute Lagrange polynomial coefficient for this output
            # L_i(σ) = ∏_{j≠i} (σ - σ_j) / (σ_i - σ_j)
            
            # Create integration points (81 points for trapezoidal rule)
            sigma_range = np.linspace(self.sigmas[t], self.sigmas[t + 1], 81)
            
            # Evaluate Lagrange basis polynomial at integration points
            lagrange_poly = np.ones(81)
            for j in range(order):
                if i == j:
                    continue
                # Multiply by (σ - σ_j) / (σ_i - σ_j)
                lagrange_poly *= (sigma_range - self.sigmas[t - j])
                lagrange_poly /= (self.sigmas[t - i] - self.sigmas[t - j])
            
            # Integrate using trapezoidal rule: ∫ L_i(σ) dσ
            lms_coeff = np.trapz(y=lagrange_poly, x=sigma_range)
            
            # Add weighted contribution
            result += lms_coeff * past_output
        
        latents = result
        
        return latents
    
    def reset(self):
        """Reset sampler state including output history."""
        super().reset()
        self.outputs = []
    
    def __repr__(self) -> str:
        return (
            f"KLMSSampler("
            f"n_inference_steps={self.n_inference_steps}, "
            f"lms_order={self.lms_order}, "
            f"history_size={len(self.outputs)})"
        )