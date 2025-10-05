import numpy as np
from utils.utils import xavier_init, gelu
from transformer.attention import MultiHeadAttention

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones((d_model,), dtype=np.float32)
        self.beta = np.zeros((d_model,), dtype=np.float32)

    def forward(self, x):
        # x: (..., d_model)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = xavier_init(d_model, d_ff)
        self.b1 = np.zeros((d_ff,), dtype=np.float32)
        self.W2 = xavier_init(d_ff, d_model)
        self.b2 = np.zeros((d_model,), dtype=np.float32)

    def forward(self, x):
        y = x @ self.W1 + self.b1
        y = gelu(y)
        y = y @ self.W2 + self.b2
        return y
    
class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff, rope=False):
        self.mha = MultiHeadAttention(d_model, num_heads, rope=rope)
        self.ln1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x, causal_mask_mat=None, padding_mask=None, pos_kind="sinusoidal"):
        ln_x = self.ln1.forward(x)
        mha_out, attn_weights = self.mha.forward(ln_x, causal_mask_mat=causal_mask_mat, padding_mask=padding_mask, pos_kind=pos_kind)
        x = x + mha_out
        ln_x2 = self.ln2.forward(x)
        ffn_out = self.ffn.forward(ln_x2)
        x = x + ffn_out
        return x, attn_weights