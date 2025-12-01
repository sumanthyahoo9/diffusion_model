"""
This script has the modules for the Clip Embeddings and the Clip Player
"""
import torch
from torch import nn
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
	def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
		super().__init__()
		self.token_embedding = nn.Embedding(n_vocab, n_embed)
		self.position_value = nn.Parameter(torch.zeros(n_tokens, n_embed))

	def forward(self, tokens):
		x = self.token_embedding(tokens)
		x += self.position_value
		return x


class CLIPlayer(nn.Module):
	def __init__(self, n_heads: int, n_embed: int):
		super().__init__()
		self.layer_norm_1 = nn.LayerNorm(n_embed)
		self.attention = SelfAttention(n_heads, n_embed)
		self.layer_norm_2 = nn.LayerNorm(n_embed)
		self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
		self.linear_2 = nn.Linear(4 * n_embed, n_embed)

	def forward(self, x):
		residue = x
		x = self.layer_norm_1(x)
		x = self.attention(x, causal_mask=True)
		x = x + residue

		residue = x
		x = self.layernorm_2(x)
		x = self.linear_1(x)
		x = x * torch.sigmoid(1.702 * x) # QuickGELU activation function
		x = self.linear_2(x)
		x = x + residue

		return x


class CLIP(nn.Module):
	def __init__(self):
		super().__init__()
		self.embedding = CLIPEmbedding(49408, 768, 77)
		self.layers = nn.ModuleList([
			CLIPlayer(12, 768) for _ in range(12)
			])
		self.layer_norm = nn.LayerNorm(768)

	def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
		tokens = tokens.type(torch.long)
		state = self.embedding(tokens)
		for layer in self.layers:
			state = layer(state)
		output = self.layernorm(state)
		return output
	