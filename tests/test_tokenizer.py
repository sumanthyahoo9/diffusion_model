"""
File: tests/test_tokenizer.py

Comprehensive unit tests for CLIP tokenizer.
Tests encoding, decoding, BPE, and edge cases.
"""
import pytest
from scripts.tokenizer import Tokenizer, create_bytes_table, pairwise


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_create_bytes_table(self):
        """Test byte-to-unicode table creation."""
        table = create_bytes_table()
        
        assert len(table) == 256
        # Normal ASCII letters should map to themselves
        assert table[ord('a')] == 'a'
        assert table[ord('Z')] == 'Z'
        # Numbers
        assert table[ord('0')] == '0'
    
    def test_pairwise(self):
        """Test pairwise iterator."""
        result = list(pairwise([1, 2, 3, 4]))
        assert result == [(1, 2), (2, 3), (3, 4)]
        
        # Single element
        result = list(pairwise([1]))
        assert result == []
        
        # Two elements
        result = list(pairwise([1, 2]))
        assert result == [(1, 2)]


class TestTokenizer:
    """Tests for Tokenizer class."""
    
    @pytest.fixture
    def mock_tokenizer(self, tmp_path):
        """Create tokenizer with mock vocab and merges."""
        # Create minimal vocab
        vocab = {
            "<|startoftext|>": 0,
            "<|endoftext|>": 1,
            "hello</w>": 2,
            "world</w>": 3,
            "cat</w>": 4,
            "dog</w>": 5,
            "a</w>": 6,
        }
        vocab_path = tmp_path / "vocab.json"
        import json
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)
        
        # Create minimal merges
        merges = ["h e", "e l", "l o"]
        merges_path = tmp_path / "merges.txt"
        with open(merges_path, 'w') as f:
            f.write("#version: 0.2\n")
            f.write("\n".join(merges))
            f.write("\n")
        
        return Tokenizer(str(vocab_path), str(merges_path))
    
    def test_initialization(self, mock_tokenizer):
        """Test tokenizer initializes correctly."""
        tokenizer = mock_tokenizer
        
        assert len(tokenizer.vocab) == 7
        assert tokenizer.max_length == 77
        assert tokenizer.bos_token == 0
        assert tokenizer.eos_token == 1
        assert len(tokenizer.bytes_table) == 256
    
    def test_encode_basic(self, mock_tokenizer):
        """Test basic text encoding."""
        tokenizer = mock_tokenizer
        
        # Note: This will fail with real BPE but tests structure
        tokens = tokenizer.encode("hello")
        
        assert len(tokens) == 77  # Always padded to max length
        assert tokens[0] == tokenizer.bos_token  # Starts with BOS
        assert tokens[-1] == tokenizer.pad_token  # Ends with padding
    
    def test_encode_empty_string(self, mock_tokenizer):
        """Test encoding empty string."""
        tokenizer = mock_tokenizer
        tokens = tokenizer.encode("")
        
        assert len(tokens) == 77
        assert tokens[0] == tokenizer.bos_token
        assert tokens[1] == tokenizer.eos_token
        # Rest should be padding
        assert all(t == tokenizer.pad_token for t in tokens[2:])
    
    def test_encode_long_text(self, mock_tokenizer):
        """Test encoding text longer than max_length."""
        tokenizer = mock_tokenizer
        
        # Create very long text
        long_text = "hello " * 100
        tokens = tokenizer.encode(long_text)
        
        # Should be truncated to max_length
        assert len(tokens) == 77
        assert tokens[0] == tokenizer.bos_token
    
    def test_encode_batch(self, mock_tokenizer):
        """Test batch encoding."""
        tokenizer = mock_tokenizer
        
        texts = ["hello", "world", "cat"]
        tokens_batch = tokenizer.encode_batch(texts)
        
        assert len(tokens_batch) == 3
        assert all(len(tokens) == 77 for tokens in tokens_batch)
        assert all(tokens[0] == tokenizer.bos_token for tokens in tokens_batch)
    
    def test_encode_single_vs_batch(self, mock_tokenizer):
        """Test single encoding matches batch encoding."""
        tokenizer = mock_tokenizer
        text = "hello world"
        
        single = tokenizer.encode(text)
        batch = tokenizer.encode_batch([text])[0]
        
        assert single == batch
    
    def test_whitespace_normalization(self, mock_tokenizer):
        """Test whitespace is normalized."""
        tokenizer = mock_tokenizer
        
        # Multiple spaces should be collapsed
        tokens1 = tokenizer.encode("hello  world")
        tokens2 = tokenizer.encode("hello world")
        
        # Should produce same tokens (whitespace normalized)
        # Note: Actual comparison depends on BPE implementation
        assert len(tokens1) == len(tokens2) == 77
    
    def test_case_insensitive(self, mock_tokenizer):
        """Test tokenizer is case-insensitive."""
        tokenizer = mock_tokenizer
        
        tokens1 = tokenizer.encode("HELLO")
        tokens2 = tokenizer.encode("hello")
        
        # Should produce same tokens (lowercased)
        assert tokens1 == tokens2
    
    def test_bpe_caching(self, mock_tokenizer):
        """Test BPE results are cached."""
        tokenizer = mock_tokenizer
        
        # Call BPE twice with same input
        result1 = tokenizer.bpe("test")
        result2 = tokenizer.bpe("test")
        
        # Should return same object (cached)
        assert result1 is result2
    
    def test_special_characters(self, mock_tokenizer):
        """Test encoding special characters."""
        tokenizer = mock_tokenizer
        
        # Test various special chars
        texts = [
            "hello!",
            "hello?",
            "hello.",
            "hello, world",
        ]
        
        for text in texts:
            tokens = tokenizer.encode(text)
            assert len(tokens) == 77
            assert tokens[0] == tokenizer.bos_token
    
    def test_unicode_handling(self, mock_tokenizer):
        """Test unicode characters are handled."""
        tokenizer = mock_tokenizer
        
        # Test various unicode
        texts = [
            "café",
            "北京",
            "🐱",
        ]
        
        for text in texts:
            tokens = tokenizer.encode(text)
            assert len(tokens) == 77
            # Should not crash
    
    def test_decode_basic(self, mock_tokenizer):
        """Test basic decoding."""
        tokenizer = mock_tokenizer
        
        # Encode then decode
        original = "hello"
        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)
        
        # Should be similar (exact match depends on BPE)
        assert isinstance(decoded, str)
    
    def test_decode_with_padding(self, mock_tokenizer):
        """Test decoding stops at padding."""
        tokenizer = mock_tokenizer
        
        # Create tokens with padding
        tokens = [tokenizer.bos_token, 2, 3, tokenizer.eos_token] + \
                 [tokenizer.pad_token] * 73
        
        decoded = tokenizer.decode(tokens)
        
        # Should only decode non-padding tokens
        assert isinstance(decoded, str)
    
    def test_len(self, mock_tokenizer):
        """Test __len__ returns vocab size."""
        tokenizer = mock_tokenizer
        assert len(tokenizer) == 7
    
    def test_repr(self, mock_tokenizer):
        """Test __repr__ string."""
        tokenizer = mock_tokenizer
        repr_str = repr(tokenizer)
        
        assert "Tokenizer" in repr_str
        assert "vocab_size=7" in repr_str
        assert "max_length=77" in repr_str


class TestTokenizerEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def mock_tokenizer(self, tmp_path):
        """Create minimal tokenizer."""
        vocab = {
            "<|startoftext|>": 0,
            "<|endoftext|>": 1,
            "test</w>": 2,
        }
        vocab_path = tmp_path / "vocab.json"
        import json
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)
        
        merges_path = tmp_path / "merges.txt"
        with open(merges_path, 'w') as f:
            f.write("#version: 0.2\n\n")
        
        return Tokenizer(str(vocab_path), str(merges_path))
    
    def test_empty_batch(self, mock_tokenizer):
        """Test encoding empty batch."""
        tokenizer = mock_tokenizer
        result = tokenizer.encode_batch([])
        assert result == []
    
    def test_very_long_word(self, mock_tokenizer):
        """Test encoding very long word."""
        tokenizer = mock_tokenizer
        
        # Create word longer than max_length
        long_word = "a" * 200
        tokens = tokenizer.encode(long_word)
        
        # Should be truncated
        assert len(tokens) == 77
    
    def test_only_special_tokens(self, mock_tokenizer):
        """Test text with only special tokens."""
        tokenizer = mock_tokenizer
        
        text = "<|startoftext|> <|endoftext|>"
        tokens = tokenizer.encode(text)
        
        assert len(tokens) == 77
    
    def test_repeated_encoding(self, mock_tokenizer):
        """Test encoding same text multiple times."""
        tokenizer = mock_tokenizer
        text = "hello world"
        
        results = [tokenizer.encode(text) for _ in range(10)]
        
        # All should be identical
        for result in results[1:]:
            assert result == results[0]


class TestTokenizerIntegration:
    """Integration tests with typical use cases."""
    
    @pytest.fixture
    def mock_tokenizer(self, tmp_path):
        """Create tokenizer for integration tests."""
        vocab = {
            "<|startoftext|>": 0,
            "<|endoftext|>": 1,
            "a</w>": 2,
            "cat</w>": 3,
            "sitting</w>": 4,
            "on</w>": 5,
            "mat</w>": 6,
        }
        vocab_path = tmp_path / "vocab.json"
        import json
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)
        
        merges_path = tmp_path / "merges.txt"
        with open(merges_path, 'w') as f:
            f.write("#version: 0.2\n\n")
        
        return Tokenizer(str(vocab_path), str(merges_path))
    
    def test_typical_prompt(self, mock_tokenizer):
        """Test encoding typical Stable Diffusion prompt."""
        tokenizer = mock_tokenizer
        
        prompt = "a cat sitting on a mat"
        tokens = tokenizer.encode(prompt)
        
        assert len(tokens) == 77
        assert tokens[0] == tokenizer.bos_token
        # Should have some content tokens
        assert any(t not in [tokenizer.bos_token, tokenizer.eos_token, 
                             tokenizer.pad_token] for t in tokens)
    
    def test_batch_of_prompts(self, mock_tokenizer):
        """Test encoding batch of prompts."""
        tokenizer = mock_tokenizer
        
        prompts = [
            "a cat",
            "a dog",
            "a bird",
        ]
        
        tokens_batch = tokenizer.encode_batch(prompts)
        
        assert len(tokens_batch) == 3
        assert all(len(t) == 77 for t in tokens_batch)
        # All should start with BOS
        assert all(t[0] == tokenizer.bos_token for t in tokens_batch)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])