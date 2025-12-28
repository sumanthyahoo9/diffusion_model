"""
File: src/sampling/utils.py
Unit Test: tests/test_sampling.py::TestSamplingUtils

Utility functions for sampling and sigma schedules.
Used by K-Euler and K-LMS samplers.
"""
import numpy as np


def get_alphas_cumprod(
    n_training_steps: int = 1000,
    beta_start: float = 0.00085,
    beta_end: float = 0.012
) -> np.ndarray:
    """
    Compute cumulative product of alphas (ᾱ_t) for noise schedule.
    
    This is the same as in BaseScheduler but returns numpy array
    for compatibility with K-samplers.
    
    Args:
        n_training_steps: Number of training timesteps (T)
        beta_start: Starting noise variance
        beta_end: Ending noise variance
        
    Returns:
        (T,) array of ᾱ_t values
    """
    # Linear beta schedule
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps) ** 2
    
    # Compute alphas: α_t = 1 - β_t
    alphas = 1.0 - betas
    
    # Cumulative product: ᾱ_t = ∏_{i=1}^t α_i
    alphas_cumprod = np.cumprod(alphas, axis=0)
    
    return alphas_cumprod


def timestep_to_sigma(
    timestep: int,
    n_training_steps: int = 1000
) -> float:
    """
    Convert timestep t to sigma (noise level).
    
    Sigma parameterization: σ_t = √((1-ᾱ_t)/ᾱ_t)
    
    This is used in Karras et al. 2022 formulation.
    
    Args:
        timestep: Timestep index [0, T-1]
        n_training_steps: Total training steps
        
    Returns:
        Sigma value (noise level)
    """
    alphas_cumprod = get_alphas_cumprod(n_training_steps)
    alpha_t = alphas_cumprod[timestep]
    sigma = np.sqrt((1 - alpha_t) / alpha_t)
    return sigma


def sigma_to_timestep(
    sigma: float,
    n_training_steps: int = 1000
) -> int:
    """
    Convert sigma back to timestep (inverse of timestep_to_sigma).
    
    Args:
        sigma: Noise level
        n_training_steps: Total training steps
        
    Returns:
        Approximate timestep index
    """
    alphas_cumprod = get_alphas_cumprod(n_training_steps)
    
    # From σ = √((1-ᾱ)/ᾱ), solve for ᾱ:
    # σ² = (1-ᾱ)/ᾱ
    # σ²·ᾱ = 1-ᾱ
    # ᾱ(σ²+1) = 1
    # ᾱ = 1/(σ²+1)
    target_alpha = 1 / (sigma ** 2 + 1)
    
    # Find closest timestep
    timestep = np.abs(alphas_cumprod - target_alpha).argmin()
    
    return int(timestep)


def get_sigmas_karras(
    n_steps: int,
    sigma_min: float = 0.1,
    sigma_max: float = 10.0,
    rho: float = 7.0
) -> np.ndarray:
    """
    Generate sigma schedule using Karras et al. 2022 method.
    
    This provides better spacing of noise levels for ODE solvers.
    
    Formula: σ_i = (σ_max^(1/ρ) + i/(n-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
    
    Args:
        n_steps: Number of sampling steps
        sigma_min: Minimum sigma (low noise)
        sigma_max: Maximum sigma (high noise)
        rho: Scheduling parameter (7.0 works well)
        
    Returns:
        (n_steps+1,) array of sigma values (includes 0 at end)
    """
    ramp = np.linspace(0, 1, n_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas = np.append(sigmas, 0.0)  # Add final 0
    return sigmas