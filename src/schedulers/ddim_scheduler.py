"""
File: src/schedulers/ddim_scheduler.py
Unit Test: tests/test_schedulers.py::TestDDIMScheduler

DDIM (Denoising Diffusion Implicit Models) scheduler.
From Song et al. 2021.

Key properties:
- Deterministic sampling (no noise added)
- Can skip timesteps (10-100 steps vs 1000)
- 10-50x faster than DDPM with comparable quality
- Enables interpolation in latent space
"""
import torch
from typing import Optional
from .base_scheduler import BaseScheduler


class DDIMScheduler(BaseScheduler):
    """
    DDIM scheduler for fast deterministic sampling.
    
    DDIM discovers that DDPM's stochastic sampling is NOT necessary.
    By using a deterministic update rule, we can:
    1. Skip most timesteps (use 50 instead of 1000)
    2. Get deterministic outputs (same noise → same image)
    3. Enable latent space interpolation
    
    Update rule:
        x_{t-1} = √ᾱ_{t-1} * x̂_0 + √(1-ᾱ_{t-1}) * ε_θ(x_t, t)
    
    Where x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t is predicted clean sample.
    
    Args:
        num_train_timesteps: Training timesteps (default: 1000)
        num_inference_steps: Sampling timesteps (default: 50)
        beta_start: Starting noise variance
        beta_end: Ending noise variance
        beta_schedule: "linear" or "cosine"
        clip_sample: Clip x_0 prediction to [-1, 1]
        eta: Stochasticity parameter (0=deterministic, 1=DDPM)
        device: Device for tensors
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = True,
        eta: float = 0.0,
        device: str = "cpu"
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            device=device
        )
        self.num_inference_steps = num_inference_steps
        self.clip_sample = clip_sample
        self.eta = eta  # 0 = deterministic, 1 = stochastic like DDPM
        
        # Set timesteps for inference (subset of training timesteps)
        self.set_timesteps(num_inference_steps)
    
    def set_timesteps(self, num_inference_steps: int):
        """
        Set timesteps for inference (which subset to use).
        
        Args:
            num_inference_steps: Number of steps for sampling
        """
        self.num_inference_steps = num_inference_steps
        
        # Create evenly spaced timesteps
        # E.g., if num_train=1000, num_inference=50, use [0, 20, 40, ..., 980]
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(
            0, self.num_train_timesteps, step_ratio, device=self.device
        ).long()
        
        # Reverse order for sampling (high noise → low noise)
        self.timesteps = torch.flip(self.timesteps, dims=[0])
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: Optional[float] = None,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Perform one DDIM denoising step: x_t → x_{t-Δt}.
        
        DDIM update (deterministic when eta=0):
        1. Predict x̂_0: x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
        2. Predict x_{t-Δt}: x_{t-Δt} = √ᾱ_{t-Δt} * x̂_0 + √(1-ᾱ_{t-Δt}) * ε_θ
        
        Args:
            model_output: (B, C, H, W) predicted noise ε_θ(x_t, t)
            timestep: Current timestep index (not value!)
            sample: (B, C, H, W) current noisy sample x_t
            eta: Override stochasticity (None = use self.eta)
            generator: Random generator for noise
            
        Returns:
            (B, C, H, W) denoised sample x_{t-Δt}
        """
        if eta is None:
            eta = self.eta
        
        # Get current and previous timestep values
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        # Extract ᾱ values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else torch.tensor(1.0, device=self.device)
        )
        
        beta_prod_t = 1 - alpha_prod_t
        
        # Predict x_0 (clean sample)
        # x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
        pred_original_sample = (
            sample - beta_prod_t.sqrt() * model_output
        ) / alpha_prod_t.sqrt()
        
        # Clip predicted x_0
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute "direction pointing to x_t"
        # This is the noise component that points from x̂_0 toward x_t
        pred_sample_direction = (1 - alpha_prod_t_prev).sqrt() * model_output
        
        # Compute predicted previous sample
        # x_{t-Δt} = √ᾱ_{t-Δt} * x̂_0 + √(1-ᾱ_{t-Δt}) * ε_θ
        pred_prev_sample = (
            alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
        )
        
        # Add noise for stochastic sampling (if eta > 0)
        if eta > 0:
            # Compute variance
            variance = self._get_variance(timestep, prev_timestep)
            std_dev_t = eta * variance.sqrt()
            
            # Sample noise
            noise = torch.randn(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype
            )
            
            pred_prev_sample = pred_prev_sample + std_dev_t * noise
        
        return pred_prev_sample
    
    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.Tensor:
        """
        Compute variance for stochastic DDIM (when eta > 0).
        
        Variance: σ_t² = η² * β̃_t
        Where β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t
        
        Args:
            timestep: Current timestep
            prev_timestep: Previous timestep
            
        Returns:
            Scalar variance
        """
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else torch.tensor(1.0, device=self.device)
        )
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        
        return variance
    
    def sample(
        self,
        model: torch.nn.Module,
        shape: tuple,
        context: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Full DDIM sampling loop: x_T → x_0 (fast).
        
        Much faster than DDPM due to skipping most timesteps.
        
        Args:
            model: Diffusion model
            shape: Output shape (B, C, H, W)
            context: Optional conditioning
            generator: Random generator
            device: Device for computation
            
        Returns:
            (B, C, H, W) generated samples
        """
        # Start from pure noise
        sample = torch.randn(shape, generator=generator, device=device)
        
        # Iteratively denoise (only num_inference_steps iterations!)
        for i, t in enumerate(self.timesteps):
            # Get timestep value
            timestep_value = t.item()
            
            # Create time embedding
            time_emb = self._get_timestep_embedding(
                torch.tensor([timestep_value] * shape[0], device=device),
                embedding_dim=320
            )
            
            # Predict noise
            with torch.no_grad():
                if context is not None:
                    noise_pred = model(sample, context, time_emb)
                else:
                    noise_pred = model(sample, time_emb)
            
            # Denoise one step (using timestep index i, not value t)
            sample = self.step(noise_pred, timestep_value, sample, generator=generator)
        
        return sample
    
    def _get_timestep_embedding(
        self,
        timesteps: torch.Tensor,
        embedding_dim: int,
        max_period: int = 10000
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Same as DDPM scheduler.
        
        Args:
            timesteps: (B,) timestep values
            embedding_dim: Embedding dimension
            max_period: Maximum period
            
        Returns:
            (B, embedding_dim) embeddings
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
            f"DDIMScheduler(num_train_timesteps={self.num_train_timesteps}, "
            f"num_inference_steps={self.num_inference_steps}, "
            f"eta={self.eta}, "
            f"beta_schedule={self.beta_schedule})"
        )