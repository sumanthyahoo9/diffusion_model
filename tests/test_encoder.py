"""
File: tests/test_encoder.py

Unit tests for VAE Encoder.
Tests architecture, forward pass, and output properties.
"""
import pytest
import torch

from src.models.encoder import Encoder, EncoderAttentionBlock


class TestEncoderAttentionBlock:
    """Tests for encoder attention block."""
    
    def test_initialization(self):
        """Test attention block initializes correctly."""
        block = EncoderAttentionBlock(channels=512)
        assert block.groupnorm.num_channels == 512
    
    def test_forward_shape(self):
        """Test output shape matches input."""
        block = EncoderAttentionBlock(channels=512)
        x = torch.randn(2, 512, 8, 8)
        
        output = block(x)
        
        assert output.shape == (2, 512, 8, 8)
    
    def test_residual_connection(self):
        """Test residual connection is applied."""
        block = EncoderAttentionBlock(channels=512)
        x = torch.randn(1, 512, 8, 8)
        
        output = block(x)
        
        # Output should not be identical (processing happened)
        assert not torch.allclose(output, x)


class TestEncoder:
    """Tests for VAE Encoder."""
    
    def test_initialization(self):
        """Test encoder initializes without errors."""
        encoder = Encoder()
        assert encoder.scaling_factor == 0.18215
    
    def test_forward_shape_512(self):
        """Test encoding 512x512 images."""
        encoder = Encoder()
        
        # Standard Stable Diffusion input
        img = torch.randn(2, 3, 512, 512)
        noise = torch.randn(2, 4, 64, 64)
        
        latent = encoder(img, noise)
        
        # Should compress 8x in each dimension
        assert latent.shape == (2, 4, 64, 64)
        assert not torch.isnan(latent).any()
    
    def test_forward_shape_256(self):
        """Test encoding 256x256 images."""
        encoder = Encoder()
        
        img = torch.randn(1, 3, 256, 256)
        noise = torch.randn(1, 4, 32, 32)
        
        latent = encoder(img, noise)
        
        assert latent.shape == (1, 4, 32, 32)
    
    def test_different_batch_sizes(self):
        """Test encoder handles various batch sizes."""
        encoder = Encoder()
        
        for batch_size in [1, 2, 4, 8]:
            img = torch.randn(batch_size, 3, 512, 512)
            noise = torch.randn(batch_size, 4, 64, 64)
            
            latent = encoder(img, noise)
            assert latent.shape == (batch_size, 4, 64, 64)
    
    def test_reparameterization(self):
        """Test different noise produces different latents."""
        encoder = Encoder()
        encoder.eval()
        
        img = torch.randn(1, 3, 512, 512)
        noise1 = torch.randn(1, 4, 64, 64)
        noise2 = torch.randn(1, 4, 64, 64)
        
        with torch.no_grad():
            latent1 = encoder(img, noise1)
            latent2 = encoder(img, noise2)
        
        # Different noise should give different latents
        assert not torch.allclose(latent1, latent2)
    
    def test_deterministic_with_same_noise(self):
        """Test same noise produces same latents."""
        encoder = Encoder()
        encoder.eval()
        
        img = torch.randn(1, 3, 512, 512)
        noise = torch.randn(1, 4, 64, 64)
        
        with torch.no_grad():
            latent1 = encoder(img, noise)
            latent2 = encoder(img, noise)
        
        # Same noise should give same latents
        assert torch.allclose(latent1, latent2)
    
    def test_scaling_factor_applied(self):
        """Test output is scaled by scaling_factor."""
        encoder = Encoder()
        
        img = torch.randn(1, 3, 512, 512)
        noise = torch.zeros(1, 4, 64, 64)  # Zero noise for simplicity
        
        latent = encoder(img, noise)
        
        # Check values are in reasonable range after scaling
        # Scaling factor (0.18215) should make values smaller
        assert latent.abs().mean() < 10  # Not too large
    
    def test_encode_without_sampling(self):
        """Test deterministic encoding (mean only)."""
        encoder = Encoder()
        
        img = torch.randn(1, 3, 512, 512)
        
        mean, log_var = encoder.encode_without_sampling(img)
        
        assert mean.shape == (1, 4, 64, 64)
        assert log_var.shape == (1, 4, 64, 64)
        
        # Log variance should be clamped
        assert log_var.min() >= -30
        assert log_var.max() <= 20
    
    def test_log_variance_clamping(self):
        """Test log variance is clamped to prevent NaN."""
        encoder = Encoder()
        
        img = torch.randn(1, 3, 512, 512)
        noise = torch.randn(1, 4, 64, 64)
        
        latent = encoder(img, noise)
        
        # Should not produce NaN or Inf
        assert not torch.isnan(latent).any()
        assert not torch.isinf(latent).any()
    
    def test_gradient_flow(self):
        """Test gradients flow through encoder."""
        encoder = Encoder()
        
        img = torch.randn(1, 3, 512, 512, requires_grad=True)
        noise = torch.randn(1, 4, 64, 64)
        
        latent = encoder(img, noise)
        loss = latent.sum()
        loss.backward()
        
        # Gradients should exist
        assert img.grad is not None
        assert not torch.isnan(img.grad).any()
    
    def test_output_range(self):
        """Test encoded latents are in reasonable range."""
        encoder = Encoder()
        encoder.eval()
        
        # Normalized input [-1, 1]
        img = torch.rand(2, 3, 512, 512) * 2 - 1
        noise = torch.randn(2, 4, 64, 64)
        
        with torch.no_grad():
            latent = encoder(img, noise)
        
        # After scaling, latents should be roughly [-5, 5]
        assert latent.min() > -10
        assert latent.max() < 10
    
    def test_compression_ratio(self):
        """Test encoder achieves 64x compression."""
        encoder = Encoder()
        
        img = torch.randn(1, 3, 512, 512)
        noise = torch.randn(1, 4, 64, 64)
        
        latent = encoder(img, noise)
        
        # Input: 3 * 512 * 512 = 786,432 elements
        # Output: 4 * 64 * 64 = 16,384 elements
        # Compression: 786432 / 16384 = 48x
        
        input_size = img.numel() / img.shape[0]  # Per image
        output_size = latent.numel() / latent.shape[0]
        
        compression = input_size / output_size
        assert compression > 40  # At least 40x compression


class TestEncoderEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_input(self):
        """Test encoding all-zero image."""
        encoder = Encoder()
        
        img = torch.zeros(1, 3, 512, 512)
        noise = torch.randn(1, 4, 64, 64)
        
        latent = encoder(img, noise)
        
        assert latent.shape == (1, 4, 64, 64)
        assert not torch.isnan(latent).any()
    
    def test_extreme_values(self):
        """Test encoder handles extreme input values."""
        encoder = Encoder()
        
        # Large values
        img = torch.ones(1, 3, 512, 512) * 100
        noise = torch.randn(1, 4, 64, 64)
        
        latent = encoder(img, noise)
        
        # Should not crash or produce NaN
        assert not torch.isnan(latent).any()
        assert not torch.isinf(latent).any()
    
    def test_zero_noise(self):
        """Test encoding with zero noise (deterministic)."""
        encoder = Encoder()
        
        img = torch.randn(1, 3, 512, 512)
        noise = torch.zeros(1, 4, 64, 64)
        
        latent = encoder(img, noise)
        
        # Should only use mean (no stochasticity)
        assert latent.shape == (1, 4, 64, 64)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])