"""
File: tests/test_schedulers.py

Comprehensive unit tests for scheduler components.
Tests cover:
- BaseScheduler (noise schedules, forward process)
- DDPMScheduler (stochastic sampling)
- DDIMScheduler (deterministic fast sampling)
"""
import sys
import pytest
import torch
sys.path.append('../src/schedulers')
from src.schedulers.ddpm_scheduler import DDPMScheduler
from src.schedulers.ddim_scheduler import DDIMScheduler


class TestBaseScheduler:
    """Tests for BaseScheduler abstract class."""
    
    def test_linear_schedule(self):
        """Test linear beta schedule is monotonically increasing."""
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        betas = scheduler.betas
        assert len(betas) == 1000
        assert betas[0].item() == pytest.approx(0.0001, abs=1e-6)
        assert betas[-1].item() == pytest.approx(0.02, abs=1e-6)
        
        # Check monotonically increasing
        assert torch.all(betas[1:] >= betas[:-1])
    
    def test_cosine_schedule(self):
        """Test cosine beta schedule properties."""
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="cosine"
        )
        
        betas = scheduler.betas
        assert len(betas) == 1000
        
        # Cosine schedule should have smaller betas at endpoints
        assert betas[0] < betas[500]
        assert torch.all(betas > 0)
        assert torch.all(betas < 1)
    
    def test_alphas_cumprod(self):
        """Test cumulative product of alphas."""
        scheduler = DDPMScheduler(num_train_timesteps=100)
        
        # ᾱ_t should be monotonically decreasing
        alphas_cumprod = scheduler.alphas_cumprod
        assert torch.all(alphas_cumprod[1:] <= alphas_cumprod[:-1])
        
        # ᾱ_0 should be close to 1 (minimal noise)
        assert alphas_cumprod[0] > 0.99
        
        # ᾱ_T should be close to 0 (pure noise)
        assert alphas_cumprod[-1] < 0.01
    
    def test_add_noise_shape(self):
        """Test add_noise preserves shape."""
        scheduler = DDPMScheduler()
        
        batch_size = 4
        original = torch.randn(batch_size, 4, 64, 64)
        noise = torch.randn(batch_size, 4, 64, 64)
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        noisy = scheduler.add_noise(original, noise, timesteps)
        
        assert noisy.shape == original.shape
        assert not torch.isnan(noisy).any()
    
    def test_add_noise_properties(self):
        """Test forward process noise addition."""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        original = torch.randn(2, 4, 64, 64)
        noise = torch.randn(2, 4, 64, 64)
        
        # At t=0, should be mostly original
        timesteps_early = torch.tensor([0, 0])
        noisy_early = scheduler.add_noise(original, noise, timesteps_early)
        assert torch.allclose(noisy_early, original, atol=0.1)
        
        # At t=T-1, should be mostly noise
        timesteps_late = torch.tensor([999, 999])
        noisy_late = scheduler.add_noise(original, noise, timesteps_late)
        # Correlation with noise should be high
        correlation = torch.corrcoef(
            torch.stack([noisy_late.flatten(), noise.flatten()])
        )[0, 1]
        assert correlation > 0.9
    
    def test_velocity_computation(self):
        """Test velocity target computation."""
        scheduler = DDPMScheduler()
        
        sample = torch.randn(2, 4, 64, 64)
        noise = torch.randn(2, 4, 64, 64)
        timesteps = torch.tensor([500, 500])
        
        velocity = scheduler.get_velocity(sample, noise, timesteps)
        
        assert velocity.shape == sample.shape
        assert not torch.isnan(velocity).any()


class TestDDPMScheduler:
    """Tests for DDPM scheduler."""
    
    def test_initialization(self):
        """Test DDPM scheduler initializes correctly."""
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        assert scheduler.num_train_timesteps == 1000
        assert scheduler.beta_start == 0.0001
        assert scheduler.beta_end == 0.02
        assert len(scheduler) == 1000
    
    def test_step_reduces_noise(self):
        """Test that step() progressively denoises."""
        scheduler = DDPMScheduler()
        
        # Create noisy sample
        clean = torch.randn(1, 4, 8, 8)
        noise = torch.randn(1, 4, 8, 8)
        
        # High noise timestep
        t_high = 800
        timesteps = torch.tensor([t_high])
        noisy = scheduler.add_noise(clean, noise, timesteps)
        
        # Denoise one step (predict noise perfectly)
        denoised = scheduler.step(noise, t_high, noisy)
        
        # After denoising, should be closer to clean
        dist_before = torch.dist(noisy, clean)
        dist_after = torch.dist(denoised, clean)
        assert dist_after < dist_before
    
    def test_step_output_shape(self):
        """Test step produces correct output shape."""
        scheduler = DDPMScheduler()
        
        sample = torch.randn(2, 4, 64, 64)
        noise_pred = torch.randn(2, 4, 64, 64)
        timestep = 500
        
        output = scheduler.step(noise_pred, timestep, sample)
        
        assert output.shape == sample.shape
        assert not torch.isnan(output).any()
    
    def test_step_final_timestep(self):
        """Test step at t=0 produces no additional noise."""
        scheduler = DDPMScheduler()
        
        sample = torch.randn(1, 4, 8, 8)
        noise_pred = torch.randn(1, 4, 8, 8)
        
        # At t=0, no noise should be added
        output = scheduler.step(noise_pred, 0, sample)
        
        # Output should be deterministic (no noise added)
        output2 = scheduler.step(noise_pred, 0, sample)
        assert torch.allclose(output, output2)
    
    def test_clip_sample(self):
        """Test sample clipping functionality."""
        scheduler_clip = DDPMScheduler(clip_sample=True)
        scheduler_no_clip = DDPMScheduler(clip_sample=False)
        
        # Create sample with extreme values
        sample = torch.randn(1, 4, 8, 8) * 10
        noise_pred = torch.randn(1, 4, 8, 8)
        timestep = 500
        
        output_clip = scheduler_clip.step(noise_pred, timestep, sample)
        output_no_clip = scheduler_no_clip.step(noise_pred, timestep, sample)
        
        # Clipped version should be in [-1, 1] range
        assert output_clip.min() >= -1.1  # Some tolerance
        assert output_clip.max() <= 1.1
    
    def test_timestep_embedding(self):
        """Test sinusoidal timestep embedding generation."""
        scheduler = DDPMScheduler()
        
        timesteps = torch.tensor([0, 100, 500, 999])
        embeddings = scheduler._get_timestep_embedding(timesteps, embedding_dim=320)
        
        assert embeddings.shape == (4, 320)
        assert not torch.isnan(embeddings).any()
        
        # Different timesteps should produce different embeddings
        assert not torch.allclose(embeddings[0], embeddings[1])
    
    def test_reproducibility_with_generator(self):
        """Test sampling is reproducible with same generator."""
        scheduler = DDPMScheduler()
        
        sample = torch.randn(1, 4, 8, 8)
        noise_pred = torch.randn(1, 4, 8, 8)
        timestep = 500
        
        generator1 = torch.Generator().manual_seed(42)
        generator2 = torch.Generator().manual_seed(42)
        
        output1 = scheduler.step(noise_pred, timestep, sample, generator1)
        output2 = scheduler.step(noise_pred, timestep, sample, generator2)
        
        assert torch.allclose(output1, output2)


class TestDDIMScheduler:
    """Tests for DDIM scheduler."""
    
    def test_initialization(self):
        """Test DDIM scheduler initializes correctly."""
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            num_inference_steps=50
        )
        
        assert scheduler.num_train_timesteps == 1000
        assert scheduler.num_inference_steps == 50
        assert len(scheduler.timesteps) == 50
    
    def test_set_timesteps(self):
        """Test timestep selection for inference."""
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        
        scheduler.set_timesteps(50)
        assert len(scheduler.timesteps) == 50
        
        # Timesteps should be evenly spaced
        assert scheduler.timesteps[0] == 980  # Reversed order
        assert scheduler.timesteps[-1] == 0
        
        # Check spacing
        step_size = 1000 // 50
        expected_spacing = step_size
        actual_spacing = (scheduler.timesteps[0] - scheduler.timesteps[1]).item()
        assert abs(actual_spacing - expected_spacing) <= 1
    
    def test_deterministic_sampling(self):
        """Test DDIM is deterministic when eta=0."""
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            num_inference_steps=50,
            eta=0.0  # Fully deterministic
        )
        scheduler.set_timesteps(50)
        
        sample = torch.randn(1, 4, 8, 8)
        noise_pred = torch.randn(1, 4, 8, 8)
        timestep = scheduler.timesteps[10].item()
        
        # Run step twice
        output1 = scheduler.step(noise_pred, timestep, sample)
        output2 = scheduler.step(noise_pred, timestep, sample)
        
        # Should be identical (deterministic)
        assert torch.allclose(output1, output2)
    
    def test_stochastic_sampling(self):
        """Test DDIM becomes stochastic when eta=1."""
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            num_inference_steps=50,
            eta=1.0  # Fully stochastic
        )
        scheduler.set_timesteps(50)
        
        sample = torch.randn(1, 4, 8, 8)
        noise_pred = torch.randn(1, 4, 8, 8)
        timestep = scheduler.timesteps[10].item()
        
        # Run step twice with different generators
        generator1 = torch.Generator().manual_seed(42)
        generator2 = torch.Generator().manual_seed(43)
        
        output1 = scheduler.step(noise_pred, timestep, sample, generator=generator1)
        output2 = scheduler.step(noise_pred, timestep, sample, generator=generator2)
        
        # Should be different (stochastic)
        assert not torch.allclose(output1, output2)
    
    def test_step_output_shape(self):
        """Test DDIM step preserves shape."""
        scheduler = DDIMScheduler()
        scheduler.set_timesteps(50)
        
        sample = torch.randn(2, 4, 64, 64)
        noise_pred = torch.randn(2, 4, 64, 64)
        timestep = scheduler.timesteps[10].item()
        
        output = scheduler.step(noise_pred, timestep, sample)
        
        assert output.shape == sample.shape
        assert not torch.isnan(output).any()
    
    def test_faster_than_ddpm(self):
        """Test DDIM uses fewer steps than DDPM."""
        ddpm = DDPMScheduler(num_train_timesteps=1000)
        ddim = DDIMScheduler(
            num_train_timesteps=1000,
            num_inference_steps=50
        )
        
        # DDIM should use 20x fewer steps
        assert len(ddim.timesteps) == 50
        assert len(ddpm) == 1000
        assert len(ddim.timesteps) < len(ddpm) / 10
    
    def test_variance_computation(self):
        """Test variance calculation for stochastic DDIM."""
        scheduler = DDIMScheduler(eta=1.0)
        scheduler.set_timesteps(50)
        
        # Get two consecutive timesteps
        t = scheduler.timesteps[10].item()
        t_prev = t - (1000 // 50)
        
        variance = scheduler._get_variance(t, t_prev)
        
        assert variance > 0
        assert not torch.isnan(variance)


class TestSchedulerComparison:
    """Compare DDPM and DDIM schedulers."""
    
    def test_same_forward_process(self):
        """Test both schedulers use same forward process."""
        ddpm = DDPMScheduler(num_train_timesteps=1000)
        ddim = DDIMScheduler(num_train_timesteps=1000)
        
        original = torch.randn(2, 4, 8, 8)
        noise = torch.randn(2, 4, 8, 8)
        timesteps = torch.tensor([500, 500])
        
        noisy_ddpm = ddpm.add_noise(original, noise, timesteps)
        noisy_ddim = ddim.add_noise(original, noise, timesteps)
        
        # Forward process should be identical
        assert torch.allclose(noisy_ddpm, noisy_ddim)
    
    def test_different_reverse_process(self):
        """Test DDPM and DDIM have different reverse processes."""
        ddpm = DDPMScheduler()
        ddim = DDIMScheduler(eta=0.0)  # Deterministic DDIM
        
        sample = torch.randn(1, 4, 8, 8)
        noise_pred = torch.randn(1, 4, 8, 8)
        timestep = 500
        
        # Same generator for fair comparison
        gen_ddpm = torch.Generator().manual_seed(42)
        gen_ddim = torch.Generator().manual_seed(42)
        
        output_ddpm = ddpm.step(noise_pred, timestep, sample, gen_ddpm)
        output_ddim = ddim.step(noise_pred, timestep, sample, generator=gen_ddim)
        
        # Outputs should be different (different algorithms)
        assert not torch.allclose(output_ddpm, output_ddim)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])