import numpy as np
from utils.utils import xavier_init, softmax
from helper.rope_helper import apply_rope

class ScaledDotProductAttention:
    def __init__(self):
        pass

    def forward(self, q, k, v, mask=None, padding_mask=None):
        """
        q,k,v: (batch, heads, seq, head_dim)
        mask: causal mask (seq, seq) boolean where True = masked (future) (optional)
        padding_mask: (batch, seq) boolean where True = pad token positions (optional)
        """
        head_dim = q.shape[-1]
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(head_dim)  # (batch, heads, seq, seq)
        if mask is not None:
            mask_b = mask[None, None, :, :]  # broadcast
            scores = np.where(mask_b, -1e9, scores)
        if padding_mask is not None:
            # prevent attention to padding positions: padding_mask (batch, seq) -> (batch, 1, 1, seq) to mask key positions
            pm = padding_mask[:, None, None, :]
            scores = np.where(pm, -1e9, scores)
        attn = softmax(scores, axis=-1)  # (batch, heads, seq, seq)
        out = np.matmul(attn, v)  # (batch, heads, seq, head_dim)
        return out, attn

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, rope=False):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope

        self.W_q = xavier_init(d_model, d_model)
        self.W_k = xavier_init(d_model, d_model)
        self.W_v = xavier_init(d_model, d_model)
        self.W_o = xavier_init(d_model, d_model)

        self.attn = ScaledDotProductAttention()

    def _split_heads(self, x):
        # (batch, seq, d_model) -> (batch, heads, seq, head_dim)
        b, seq, _ = x.shape
        x = x.reshape(b, seq, self.num_heads, self.head_dim)
        return np.transpose(x, (0,2,1,3))

    def _combine_heads(self, x):
        # (batch, heads, seq, head_dim) -> (batch, seq, d_model)
        x = np.transpose(x, (0,2,1,3))
        b, seq, heads, head_dim = x.shape
        return x.reshape(b, seq, heads * head_dim)

    def forward(self, x, causal_mask_mat=None, padding_mask=None, pos_kind="sinusoidal"):
        # x: (batch, seq, d_model)
        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v
        qh = self._split_heads(q)
        kh = self._split_heads(k)
        vh = self._split_heads(v)

        # apply RoPE if enabled
        if self.rope or pos_kind == "rope":
            qh = apply_rope(qh, qh.shape[2])
            kh = apply_rope(kh, kh.shape[2])

        out_heads, attn_weights = self.attn.forward(qh, kh, vh, mask=causal_mask_mat, padding_mask=padding_mask)
        concat = self._combine_heads(out_heads)
        out = concat @ self.W_o
        return out, attn_weights
