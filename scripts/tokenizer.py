"""
File: scripts/tokenizer.py
Unit Test: tests/test_scripts.py::TestTokenizer

CLIP tokenizer for text prompt encoding.
Uses Byte-Pair Encoding (BPE) for subword tokenization.

Based on OpenAI's CLIP tokenizer with vocabulary of 49,408 tokens
and maximum sequence length of 77 tokens.
"""
import json
import functools
import unicodedata
from typing import List, Tuple, Dict
import regex as re


def create_bytes_table() -> Dict[int, str]:
    """
    Create byte-to-unicode mapping table.
    
    Maps bytes (0-255) to unicode characters, handling special cases
    for control characters and whitespace.
    
    Returns:
        Dict mapping byte values to unicode characters
    """
    table = {}
    special_count = 0
    
    for byte in range(256):
        # Get unicode category (C=control, Z=separator)
        category = unicodedata.category(chr(byte))
        
        if category[0] not in ['C', 'Z']:
            # Normal printable character
            table[byte] = chr(byte)
        else:
            # Map special chars to private use area (256+)
            table[byte] = chr(special_count + 256)
            special_count += 1
    
    return table


def pairwise(seq):
    """
    Create iterator of consecutive pairs from sequence.
    
    Args:
        seq: Input sequence
        
    Yields:
        Consecutive pairs (a, b)
        
    Example:
        >>> list(pairwise([1, 2, 3, 4]))
        [(1, 2), (2, 3), (3, 4)]
    """
    a = iter(seq)
    b = iter(seq)
    next(b)
    return zip(a, b)


class Tokenizer:
    """
    CLIP tokenizer using Byte-Pair Encoding (BPE).
    
    Tokenizes text prompts into sequences of token IDs for CLIP encoding.
    
    Vocabulary:
    - Size: 49,408 tokens
    - Special tokens: <|startoftext|>, <|endoftext|>
    - Max length: 77 tokens (including special tokens)
    
    Files required:
    - vocab.json: Token to ID mapping
    - merges.txt: BPE merge rules
    
    Example:
        >>> tokenizer = Tokenizer()
        >>> tokens = tokenizer.encode("A cat sitting on a mat")
        >>> len(tokens)
        77  # Always padded to max length
    """
    
    def __init__(
        self,
        vocab_path: str = "vocab.json",
        merges_path: str = "merges.txt"
    ):
        """
        Initialize tokenizer with vocabulary and merge rules.
        
        Args:
            vocab_path: Path to vocabulary JSON file
            merges_path: Path to BPE merges text file
        """
        # Load vocabulary
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab: Dict[str, int] = json.load(f)
        
        # Load BPE merges
        with open(merges_path, encoding="utf-8") as f:
            lines = f.read().split('\n')
            lines = lines[1:-1]  # Skip header and last empty line
            self.merges: Dict[Tuple[str, str], int] = {
                tuple(bigram.split()): i for i, bigram in enumerate(lines)
            }
        
        # Special tokens
        self.bos_token = self.vocab["<|startoftext|>"]
        self.eos_token = self.vocab["<|endoftext|>"]
        self.pad_token = self.vocab["<|endoftext|>"]  # Use EOS for padding
        
        # Configuration
        self.max_length = 77
        
        # Byte encoding table
        self.bytes_table = create_bytes_table()
        
        # Regex pattern for chunking text
        # Matches: special tokens, contractions, letters, numbers, other
        self.chunk_pattern = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE
        )
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Process:
        1. Normalize unicode (NFC)
        2. Normalize whitespace
        3. Lowercase
        4. Split into chunks (words, numbers, punctuation)
        5. Apply BPE to each chunk
        6. Add special tokens (BOS, EOS)
        7. Pad/truncate to max_length
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs (length = max_length = 77)
            
        Example:
            >>> tokenizer = Tokenizer()
            >>> tokens = tokenizer.encode("Hello world!")
            >>> tokens[0] == tokenizer.bos_token
            True
            >>> tokens[-1] == tokenizer.pad_token
            True
        """
        # Normalize unicode to canonical form
        text = unicodedata.normalize('NFC', text)
        
        # Normalize whitespace (collapse multiple spaces)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Lowercase (CLIP is case-insensitive)
        text = text.lower()
        
        # Start with BOS token
        tokens = [self.bos_token]
        
        # Process each chunk
        for chunk in re.findall(self.chunk_pattern, text):
            # Encode chunk as bytes, then map through bytes table
            chunk_bytes = chunk.encode('utf-8')
            chunk_unicode = ''.join(self.bytes_table[byte] for byte in chunk_bytes)
            
            # Apply BPE to get subword tokens
            bpe_tokens = self.bpe(chunk_unicode)
            
            # Convert tokens to IDs
            tokens.extend(self.vocab[token] for token in bpe_tokens)
        
        # Add EOS token
        tokens.append(self.eos_token)
        
        # Truncate if too long
        tokens = tokens[:self.max_length]
        
        # Pad if too short
        token_length = len(tokens)
        pad_length = self.max_length - token_length
        tokens += [self.pad_token] * pad_length
        
        return tokens
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of token ID lists, each of length max_length
            
        Example:
            >>> tokenizer = Tokenizer()
            >>> batch = ["Hello", "World"]
            >>> tokens = tokenizer.encode_batch(batch)
            >>> len(tokens)
            2
            >>> len(tokens[0])
            77
        """
        return [self.encode(text) for text in texts]
    
    @functools.lru_cache(maxsize=10000)
    def bpe(self, chunk: str) -> Tuple[str, ...]:
        """
        Apply Byte-Pair Encoding to text chunk.
        
        BPE iteratively merges the most frequent adjacent pairs
        of characters/subwords until no more merges are possible.
        
        Args:
            chunk: Text chunk to encode
            
        Returns:
            Tuple of BPE tokens
            
        Example:
            >>> tokenizer = Tokenizer()
            >>> tokenizer.bpe("hello")
            ('hello</w>',)  # Single token if in vocab
        """
        # Split into characters, mark end with </w>
        words = list(chunk)
        words[-1] = words[-1] + "</w>"
        
        # Iteratively merge pairs
        while len(words) > 1:
            # Find all valid pairs (that exist in merge rules)
            valid_pairs = [
                pair for pair in pairwise(words)
                if pair in self.merges
            ]
            
            if not valid_pairs:
                break  # No more merges possible
            
            # Select pair with lowest merge priority (merged earliest)
            bigram = min(valid_pairs, key=lambda pair: self.merges[pair])
            first, second = bigram
            
            # Merge occurrences of this bigram
            new_words = []
            i = 0
            while i < len(words):
                if (i < len(words) - 1 and
                    words[i] == first and
                    words[i + 1] == second):
                    # Merge this pair
                    new_words.append(first + second)
                    i += 2
                else:
                    # Keep as is
                    new_words.append(words[i])
                    i += 1
            
            words = new_words
        
        return tuple(words)
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text (approximate).
        
        Note: This is a simplified decoder. Full decoding requires
        reversing the BPE process and byte-to-unicode mapping.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text string
        """
        # Create reverse vocabulary
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Decode tokens
        text_tokens = []
        for token_id in tokens:
            if token_id == self.pad_token:
                break  # Stop at padding
            if token_id in [self.bos_token, self.eos_token]:
                continue  # Skip special tokens
            
            token = id_to_token.get(token_id, "<unk>")
            text_tokens.append(token)
        
        # Join and clean up
        text = ''.join(text_tokens)
        text = text.replace('</w>', ' ')
        text = text.strip()
        
        return text
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def __repr__(self) -> str:
        return (
            f"Tokenizer(vocab_size={len(self.vocab)}, "
            f"max_length={self.max_length})"
        )