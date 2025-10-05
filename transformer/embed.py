import numpy as np
np.random.seed(42)

class TokenEmbedding:
    def __init__(self, vocab_size, d_model, pad_id=None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        self.W = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02

    def forward(self, token_ids):
        return self.W[token_ids]  # (batch, seq, d_model)

class PositionalEncoding:
    def __init__(self, max_len, d_model, kind="sinusoidal"):
        assert kind in ("sinusoidal", "learned", "rope")
        self.kind = kind
        self.max_len = max_len
        self.d_model = d_model
        if kind == "sinusoidal":
            self.pe = self._sinusoidal(max_len, d_model)
        elif kind == "learned":
            self.pe = np.random.randn(max_len, d_model).astype(np.float32) * 0.01
        else:
            # RoPE doesn't precompute add vectors; it's applied in attention
            self.pe = None

    def _sinusoidal(self, max_len, d_model):
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pos = np.arange(0, max_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        return pe

    def forward(self, seq_len):
        if self.kind in ("sinusoidal", "learned"):
            return self.pe[:seq_len]  # (seq, d_model)
        else:
            return None  # RoPE handled in attention
