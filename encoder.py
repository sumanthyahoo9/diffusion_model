"""
This script has the Encoder for the Diffusion model with
The encoder compresses an input image and obtain latent representations for noisification
"""
import torch
from torch import nn
from torch.nn import functional as F
from decoder import AttentionBlock, ResidualBlock


class Encoder(nn.Module):
	"""
	This module stacks Convolutional layers and Residual blocks to convert an input image into a representation of size (8, 8)
	"""
	def __init__(self):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Conv2d(3, 128, kernel_size=3, padding=1),
			ResidualBlock(128, 128),
			ResidualBlock(128, 128),
			nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
			ResidualBlock(128, 256),
			ResidualBlock(256, 256),
			nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
			ResidualBlock(256, 512),
			ResidualBlock(512, 512),
			nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
			ResidualBlock(512, 512),
			ResidualBlock(512, 512),
			ResidualBlock(512, 512),
			AttentionBlock(512),
			ResidualBlock(512, 512),
			nn.GroupNorm(32, 512),  # Figure out what this does
			nn.SiLU(),  # Figure out what this does
			nn.Conv2d(512, 8, kernel_size=3, padding=1),
			nn.Conv2d(8, 8, kernel_size=1, padding=0)
		)

	def forward(self, x, noise):
		for module in self.layers:
			if getattr(module, 'stride', None) == (2, 2): # Padding at down-sampling should be asymmetric
				x = F.pad(x, (0, 1, 0, 1))
			x = module(x)
		assert x.shape == torch.Size([8, 8])
		print(f"The shape of the representation after the convolutional layers and Residual blocks is {x.shape}")

		mean, log_variance = torch.chunk(x, 2, dim = 1)
		log_variance = torch.clamp(log_variance, -30, 20)
		variance = log_variance.exp()
		std_dev = variance.sqrt()
		x = mean + std_dev * noise
		x *= 0.18215

		return x
