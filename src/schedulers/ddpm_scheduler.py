"""
File: src/schedulers/ddpm_scheduler.py
Unit Test: tests/test_schedulers.py::TestDDPMScheduler

DDPM (Denoising Diffusion Probabilistic Models) scheduler.
Original algorithm from Ho et al. 2020.

Key properties:
- Stochastic sampling (adds noise at each step)
- Typically uses 1000 timesteps
- High quality but slow generation
"""
from typing import Optional
import torch
from .base_scheduler import BaseScheduler


class DDPMScheduler(BaseScheduler):
    """
    DDPM scheduler for training and sampling.
    
    Training:
        1. Sample timestep t ~ Uniform(0, T-1)
        2. Add noise: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        3. Predict noise: ε_θ(x_t, t)
        4. Loss: ||ε_θ(x_t, t) - ε||²
    
    Sampling (reverse diffusion):
        Start from x_T ~ N(0, I)
        For t = T-1, ..., 0:
            ε_θ = model(x_t, t)
            x_{t-1} = (1/√α_t)[x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ] + σ_t * z
        Return x_0
    
    Args:
        num_train_timesteps: Number of diffusion steps (default: 1000)
        beta_start: Starting noise variance (default: 0.0001)
        beta_end: Ending noise variance (default: 0.02)
        beta_schedule: "linear" or "cosine"
        clip_sample: Whether to clip x_0 prediction to [-1, 1]
        device: Device for tensors
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = True,
        device: str = "cpu"
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            device=device
        )
        self.clip_sample = clip_sample
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Perform one DDPM reverse diffusion step: x_t → x_{t-1}.
        
        Algorithm (from DDPM paper):
        1. Predict x_0 from x_t and ε_θ
        2. Compute mean μ_θ(x_t, t)
        3. Add noise: x_{t-1} = μ_θ + σ_t * z  (if t > 0)
        
        Args:
            model_output: (B, C, H, W) predicted noise ε_θ(x_t, t)
            timestep: Current timestep t
            sample: (B, C, H, W) current noisy sample x_t
            generator: Optional random generator for reproducibility
            
        Returns:
            (B, C, H, W) previous sample x_{t-1}
        """
        t = timestep
        
        # Extract coefficients for this timestep
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod_prev[t] if t > 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = self.alphas[t]
        current_beta_t = self.betas[t]
        
        # Predict x_0 from x_t and noise prediction
        # From: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        # Solve: x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
        pred_original_sample = (
            sample - beta_prod_t.sqrt() * model_output
        ) / alpha_prod_t.sqrt()
        
        # Clip x_0 prediction (common in practice)
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute coefficients for μ_θ(x_t, x_0, t)
        # μ_θ = (√ᾱ_{t-1} * β_t)/(1 - ᾱ_t) * x_0 + (√α_t * (1-ᾱ_{t-1}))/(1-ᾱ_t) * x_t
        pred_original_sample_coeff = (
            alpha_prod_t_prev.sqrt() * current_beta_t
        ) / beta_prod_t
        
        current_sample_coeff = (
            current_alpha_t.sqrt() * beta_prod_t_prev
        ) / beta_prod_t
        
        # Compute predicted mean μ_θ
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample +
            current_sample_coeff * sample
        )
        
        # Add noise (stochastic sampling)
        variance = 0
        if t > 0:
            # Use posterior variance: σ_t² = β_t * (1-ᾱ_{t-1})/(1-ᾱ_t)
            variance = self.posterior_variance[t]
            
            # Sample noise
            noise = torch.randn(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype
            )
            
            # x_{t-1} = μ_θ + σ_t * z
            pred_prev_sample = pred_prev_sample + variance.sqrt() * noise
        
        return pred_prev_sample
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to samples (forward process).
        
        Wrapper around BaseScheduler.add_noise for training.
        
        Args:
            original_samples: (B, C, H, W) clean samples x_0
            noise: (B, C, H, W) Gaussian noise ε ~ N(0, I)
            timesteps: (B,) timesteps to sample at
            
        Returns:
            (B, C, H, W) noisy samples x_t
        """
        return super().add_noise(original_samples, noise, timesteps)
    
    def sample(
        self,
        model: torch.nn.Module,
        shape: tuple,
        context: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Full sampling loop: x_T → x_0.
        
        Generates samples by iteratively denoising pure noise.
        
        Args:
            model: Diffusion model (predicts noise)
            shape: Output shape (B, C, H, W)
            context: Optional conditioning (e.g., text embeddings)
            generator: Random generator for reproducibility
            device: Device for computation
            
        Returns:
            (B, C, H, W) generated samples
        """
        # Start from pure noise x_T ~ N(0, I)
        sample = torch.randn(shape, generator=generator, device=device)
        
        # Iteratively denoise
        for t in reversed(range(self.num_train_timesteps)):
            # Create timestep tensor
            timestep = torch.full(
                (shape[0],), t, device=device, dtype=torch.long
            )
            
            # Create time embedding (this is model-specific)
            # For Stable Diffusion, we need sinusoidal embedding
            time_emb = self._get_timestep_embedding(timestep, embedding_dim=320)
            
            # Predict noise
            with torch.no_grad():
                if context is not None:
                    noise_pred = model(sample, context, time_emb)
                else:
                    noise_pred = model(sample, time_emb)
            
            # Denoise one step
            sample = self.step(noise_pred, t, sample, generator)
        
        return sample
    
    def _get_timestep_embedding(
        self,
        timesteps: torch.Tensor,
        embedding_dim: int,
        max_period: int = 10000
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Standard positional encoding from "Attention is All You Need".
        
        Args:
            timesteps: (B,) timestep indices
            embedding_dim: Embedding dimension
            max_period: Maximum period for sinusoids
            
        Returns:
            (B, embedding_dim) timestep embeddings
        """
        half_dim = embedding_dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(max_period)) *
            torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device) /
            half_dim
        )
        
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if embedding_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return embedding
    
    def __repr__(self) -> str:
        return (
            f"DDPMScheduler(num_timesteps={self.num_train_timesteps}, "
            f"beta_schedule={self.beta_schedule}, "
            f"beta_range=[{self.beta_start:.6f}, {self.beta_end:.6f}])"
        )