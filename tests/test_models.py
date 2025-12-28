"""
File: tests/test_models.py

Comprehensive unit tests for diffusion model components.
Tests cover:
- TimeEmbedding
- ResidualBlock
- AttentionBlock
- Upsample
- SwitchSequential
- UNet
- FinalLayer
- Diffusion (main model)
"""
import sys
import pytest
import torch

from src.models.embeddings import TimeEmbedding
from src.models.blocks import ResidualBlock, AttentionBlock
from src.models.unet import Upsample, SwitchSequential, UNet
from src.models.diffusion import FinalLayer, Diffusion
sys.path.append('../src/models')


class TestTimeEmbedding:
    """Tests for TimeEmbedding module."""
    
    def test_initialization(self):
        """Test TimeEmbedding initializes correctly."""
        model = TimeEmbedding(n_embed=320)
        assert model.linear_1.in_features == 320
        assert model.linear_1.out_features == 1280
        assert model.linear_2.in_features == 1280
        assert model.linear_2.out_features == 1280
    
    def test_forward_shape(self):
        """Test output shape is correct."""
        model = TimeEmbedding(n_embed=320)
        batch_size = 4
        x = torch.randn(batch_size, 320)
        
        output = model(x)
        
        assert output.shape == (batch_size, 1280)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_batch_sizes(self):
        """Test model handles different batch sizes."""
        model = TimeEmbedding(n_embed=320)
        
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 320)
            output = model(x)
            assert output.shape == (batch_size, 1280)
    
    def test_gradient_flow(self):
        """Test gradients flow properly during backprop."""
        model = TimeEmbedding(n_embed=320)
        x = torch.randn(2, 320, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestResidualBlock:
    """Tests for ResidualBlock module."""
    
    def test_initialization_same_channels(self):
        """Test initialization when in_channels == out_channels."""
        model = ResidualBlock(320, 320, n_time=1280)
        assert isinstance(model.residual_layer, torch.nn.Identity)
    
    def test_initialization_different_channels(self):
        """Test initialization when in_channels != out_channels."""
        model = ResidualBlock(320, 640, n_time=1280)
        assert isinstance(model.residual_layer, torch.nn.Conv2d)
        assert model.residual_layer.in_channels == 320
        assert model.residual_layer.out_channels == 640
    
    def test_forward_shape(self):
        """Test output shape matches expected dimensions."""
        model = ResidualBlock(320, 640, n_time=1280)
        batch_size = 2
        feature = torch.randn(batch_size, 320, 32, 32)
        time = torch.randn(batch_size, 1280)
        
        output = model(feature, time)
        
        assert output.shape == (batch_size, 640, 32, 32)
        assert not torch.isnan(output).any()
    
    def test_residual_connection(self):
        """Test residual connection is properly added."""
        model = ResidualBlock(320, 320, n_time=1280)
        feature = torch.randn(2, 320, 16, 16)
        time = torch.randn(2, 1280)
        
        # Set model to eval mode to make it deterministic
        model.eval()
        with torch.no_grad():
            output = model(feature, time)
        
        # Output should not be identical to input (processing happened)
        assert not torch.allclose(output, feature)
    
    def test_time_conditioning(self):
        """Test different time embeddings produce different outputs."""
        model = ResidualBlock(320, 320, n_time=1280)
        feature = torch.randn(2, 320, 16, 16)
        time1 = torch.randn(2, 1280)
        time2 = torch.randn(2, 1280)
        
        model.eval()
        with torch.no_grad():
            output1 = model(feature, time1)
            output2 = model(feature, time2)
        
        # Different time embeddings should produce different outputs
        assert not torch.allclose(output1, output2)


class TestAttentionBlock:
    """Tests for AttentionBlock module."""
    
    @pytest.fixture
    def mock_attention_classes(self, monkeypatch):
        """Mock SelfAttention and CrossAttention for testing."""
        class MockSelfAttention(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
            def forward(self, x):
                return x
        
        class MockCrossAttention(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
            def forward(self, x, context):
                return x
        
        import blocks
        monkeypatch.setattr('blocks.SelfAttention', MockSelfAttention)
        monkeypatch.setattr('blocks.CrossAttention', MockCrossAttention)
    
    def test_initialization(self, mock_attention_classes):
        """Test AttentionBlock initializes correctly."""
        n_heads, n_embed = 8, 40
        model = AttentionBlock(n_heads, n_embed, d_context=768)
        
        channels = n_heads * n_embed  # 320
        assert model.group_norm.num_channels == channels
        assert model.conv_input.in_channels == channels
    
    def test_forward_shape(self, mock_attention_classes):
        """Test output shape preservation."""
        model = AttentionBlock(8, 40, d_context=768)
        batch_size = 2
        h, w = 16, 16
        channels = 8 * 40  # 320
        
        x = torch.randn(batch_size, channels, h, w)
        context = torch.randn(batch_size, 77, 768)  # CLIP context
        
        output = model(x, context)
        
        assert output.shape == (batch_size, channels, h, w)
        assert not torch.isnan(output).any()
    
    def test_spatial_to_sequence_transform(self, mock_attention_classes):
        """Test correct transformation from spatial to sequence format."""
        model = AttentionBlock(8, 40, d_context=768)
        batch_size = 2
        channels = 320
        h, w = 8, 8
        
        x = torch.randn(batch_size, channels, h, w)
        context = torch.randn(batch_size, 77, 768)
        
        # Check intermediate shapes by inspecting forward logic
        # This would require hooks in practice, here we just check final shape
        output = model(x, context)
        assert output.shape == x.shape


class TestUpsample:
    """Tests for Upsample module."""
    
    def test_initialization(self):
        """Test Upsample initializes correctly."""
        model = Upsample(channels=320)
        assert model.conv.in_channels == 320
        assert model.conv.out_channels == 320
    
    def test_upsampling_factor(self):
        """Test spatial dimensions are doubled."""
        model = Upsample(channels=320)
        batch_size = 2
        h, w = 16, 16
        
        x = torch.randn(batch_size, 320, h, w)
        output = model(x)
        
        assert output.shape == (batch_size, 320, 2*h, 2*w)
    
    def test_channel_preservation(self):
        """Test number of channels is preserved."""
        for channels in [64, 128, 320, 640, 1280]:
            model = Upsample(channels=channels)
            x = torch.randn(2, channels, 8, 8)
            
            output = model(x)
            assert output.shape[1] == channels


class TestSwitchSequential:
    """Tests for SwitchSequential routing module."""
    
    def test_routing_residual_block(self):
        """Test ResidualBlock receives (feature, time)."""
        res_block = ResidualBlock(320, 320)
        model = SwitchSequential(res_block)
        
        x = torch.randn(2, 320, 16, 16)
        context = torch.randn(2, 77, 768)
        time = torch.randn(2, 1280)
        
        output = model(x, context, time)
        assert output.shape == (2, 320, 16, 16)
    
    def test_routing_conv_layer(self):
        """Test regular Conv2d receives only (x)."""
        conv = torch.nn.Conv2d(4, 320, kernel_size=3, padding=1)
        model = SwitchSequential(conv)
        
        x = torch.randn(2, 4, 64, 64)
        context = torch.randn(2, 77, 768)
        time = torch.randn(2, 1280)
        
        output = model(x, context, time)
        assert output.shape == (2, 320, 64, 64)
    
    def test_mixed_layers(self):
        """Test sequential application of different layer types."""
        conv = torch.nn.Conv2d(320, 320, kernel_size=3, padding=1)
        res_block = ResidualBlock(320, 320)
        model = SwitchSequential(conv, res_block)
        
        x = torch.randn(2, 320, 16, 16)
        context = torch.randn(2, 77, 768)
        time = torch.randn(2, 1280)
        
        output = model(x, context, time)
        assert output.shape == (2, 320, 16, 16)


class TestUNet:
    """Tests for UNet architecture."""
    
    def test_initialization(self):
        """Test UNet initializes without errors."""
        model = UNet()
        assert len(model.encoders) == 12
        assert len(model.decoders) == 12
    
    def test_forward_shape(self):
        """Test UNet produces correct output shape."""
        model = UNet()
        batch_size = 1
        
        # Standard Stable Diffusion inputs
        latent = torch.randn(batch_size, 4, 64, 64)
        context = torch.randn(batch_size, 77, 768)
        time = torch.randn(batch_size, 1280)
        
        output = model(latent, context, time)
        
        # U-Net should preserve spatial dimensions
        assert output.shape == (batch_size, 320, 64, 64)
    
    def test_skip_connections(self):
        """Test skip connections are properly consumed."""
        model = UNet()
        
        latent = torch.randn(1, 4, 64, 64)
        context = torch.randn(1, 77, 768)
        time = torch.randn(1, 1280)
        
        # Should not raise errors about skip connection mismatch
        output = model(latent, context, time)
        assert output.shape[0] == 1
    
    def test_different_batch_sizes(self):
        """Test UNet handles various batch sizes."""
        model = UNet()
        context = torch.randn(1, 77, 768)
        time = torch.randn(1, 1280)
        
        for batch_size in [1, 2, 4]:
            latent = torch.randn(batch_size, 4, 64, 64)
            # Expand context and time to match batch size
            ctx = context.repeat(batch_size, 1, 1)
            t = time.repeat(batch_size, 1)
            
            output = model(latent, ctx, t)
            assert output.shape == (batch_size, 320, 64, 64)


class TestFinalLayer:
    """Tests for FinalLayer module."""
    
    def test_initialization(self):
        """Test FinalLayer initializes correctly."""
        model = FinalLayer(in_channels=320, out_channels=4)
        assert model.conv.in_channels == 320
        assert model.conv.out_channels == 4
    
    def test_forward_shape(self):
        """Test correct output shape."""
        model = FinalLayer(320, 4)
        batch_size = 2
        
        x = torch.randn(batch_size, 320, 64, 64)
        output = model(x)
        
        assert output.shape == (batch_size, 4, 64, 64)
    
    def test_spatial_preservation(self):
        """Test spatial dimensions are preserved."""
        model = FinalLayer(320, 4)
        
        for h, w in [(32, 32), (64, 64), (128, 128)]:
            x = torch.randn(1, 320, h, w)
            output = model(x)
            assert output.shape == (1, 4, h, w)


class TestDiffusion:
    """Tests for complete Diffusion model."""
    
    def test_initialization(self):
        """Test Diffusion model initializes correctly."""
        model = Diffusion()
        assert isinstance(model.time_embedding, TimeEmbedding)
        assert isinstance(model.unet, UNet)
        assert isinstance(model.final, FinalLayer)
    
    def test_forward_shape(self):
        """Test end-to-end forward pass."""
        model = Diffusion()
        batch_size = 2
        
        latent = torch.randn(batch_size, 4, 64, 64)
        context = torch.randn(batch_size, 77, 768)
        time = torch.randn(batch_size, 320)
        
        output = model(latent, context, time)
        
        # Output should match latent shape (noise prediction)
        assert output.shape == (batch_size, 4, 64, 64)
        assert not torch.isnan(output).any()
    
    def test_noise_prediction_property(self):
        """Test model produces different outputs for different timesteps."""
        model = Diffusion()
        model.eval()
        
        latent = torch.randn(1, 4, 64, 64)
        context = torch.randn(1, 77, 768)
        time1 = torch.randn(1, 320)
        time2 = torch.randn(1, 320)
        
        with torch.no_grad():
            output1 = model(latent, context, time1)
            output2 = model(latent, context, time2)
        
        # Different timesteps should produce different predictions
        assert not torch.allclose(output1, output2)
    
    def test_gradient_flow(self):
        """Test gradients flow through entire model."""
        model = Diffusion()
        
        latent = torch.randn(1, 4, 64, 64, requires_grad=True)
        context = torch.randn(1, 77, 768)
        time = torch.randn(1, 320)
        
        output = model(latent, context, time)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist for model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_parameter_count(self):
        """Test model has reasonable number of parameters."""
        model = Diffusion()
        total_params = sum(p.numel() for p in model.parameters())
        
        # Stable Diffusion U-Net has ~860M parameters
        # This is a sanity check
        assert total_params > 1_000_000, "Model seems too small"
        assert total_params < 2_000_000_000, "Model seems too large"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])