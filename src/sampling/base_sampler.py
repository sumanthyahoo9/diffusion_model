"""
File: src/sampling/base_sampler.py
Unit Test: tests/test_sampling.py::TestBaseSampler

Abstract base class for ODE/SDE samplers.
Provides common interface for K-Euler, K-LMS, and other sampling methods.
"""
from abc import ABC, abstractmethod
from typing import Optional
import torch
import numpy as np


class BaseSampler(ABC):
    """
    Base class for diffusion samplers.
    
    Samplers solve the reverse diffusion process using various
    numerical methods (Euler, LMS, Heun, DPM-Solver, etc.).
    
    Common attributes:
        sigmas: Noise schedule in sigma parameterization
        timesteps: Corresponding timesteps
        step_count: Current step in sampling
        initial_scale: Starting noise level
    
    Subclasses must implement:
        step(): Single sampling step
    """
    
    def __init__(
        self,
        n_inference_steps: int = 50,
        n_training_steps: int = 1000,
        generator: Optional[torch.Generator] = None
    ):
        """
        Initialize base sampler.
        
        Args:
            n_inference_steps: Number of sampling steps
            n_training_steps: Number of training timesteps
            generator: Random generator for reproducibility
        """
        self.n_inference_steps = n_inference_steps
        self.n_training_steps = n_training_steps
        self.generator = generator
        self.step_count = 0
        
        # Will be set by subclasses
        self.sigmas = None
        self.timesteps = None
        self.initial_scale = None
    
    @abstractmethod
    def step(
        self,
        latents: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform one sampling step.
        
        Args:
            latents: (B, C, H, W) current latents x_t
            output: (B, C, H, W) model output (usually noise prediction)
            
        Returns:
            (B, C, H, W) updated latents x_{t-1}
        """
        pass
    
    def get_input_scale(self, step_count: Optional[int] = None) -> float:
        """
        Get scaling factor for model input at given step.
        
        For sigma parameterization: scale = 1/√(σ²+1)
        
        This normalizes input to unit variance.
        
        Args:
            step_count: Step index (uses self.step_count if None)
            
        Returns:
            Scaling factor
        """
        if step_count is None:
            step_count = self.step_count
        sigma = self.sigmas[step_count]
        return 1 / (sigma ** 2 + 1) ** 0.5
    
    def set_strength(self, strength: float = 1.0):
        """
        Set denoising strength for img2img tasks.
        
        strength=1.0: Full denoising (img2img with maximum change)
        strength=0.5: Partial denoising (preserve more original)
        strength=0.0: No denoising (return original)
        
        This adjusts the starting timestep.
        
        Args:
            strength: Denoising strength in [0, 1]
        """
        start_step = self.n_inference_steps - int(self.n_inference_steps * strength)
        self.timesteps = np.linspace(
            self.n_training_steps - 1, 0, self.n_inference_steps
        )
        self.timesteps = self.timesteps[start_step:]
        self.initial_scale = self.sigmas[start_step]
        self.step_count = start_step
    
    def reset(self):
        """Reset sampler to initial state."""
        self.step_count = 0
    
    def __len__(self) -> int:
        """Return number of sampling steps."""
        return self.n_inference_steps
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_inference_steps={self.n_inference_steps}, "
            f"n_training_steps={self.n_training_steps})"
        )