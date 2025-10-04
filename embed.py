import numpy as np

np.random.seed(42)

def xavier_init(in_dim, out_dim):
    """Xavier normal init approximation"""
    std = np.sqrt(2.0 / (in_dim + out_dim))
    return np.random.randn(in_dim, out_dim).astype(np.float32) * std

def gelu(x):
    # GELU approximate (with tanh)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # embedding matrix (vocab, d_model)
        self.W = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02
        
    def forward(self, token_ids):
        return self.W[token_ids]  
