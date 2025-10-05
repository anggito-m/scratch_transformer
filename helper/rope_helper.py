import numpy as np

def apply_rope(q_or_k, seq_len):
    """
    q_or_k: (batch, heads, seq, head_dim)
    apply RoPE assuming head_dim is even
    """
    b, h, seq, dim = q_or_k.shape
    assert dim % 2 == 0
    # create sinusoidal frequencies
    pos = np.arange(seq)  # (seq,)
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    # angles: (seq, dim/2)
    angles = np.einsum('i,j->ij', pos, inv_freq)  # seq x (dim/2)
    sin = np.sin(angles)[None, None, :, :]  # (1,1,seq,dim/2)
    cos = np.cos(angles)[None, None, :, :]
    # split last dim into even/odd
    x1 = q_or_k[..., 0::2]  # (b,h,seq,dim/2)
    x2 = q_or_k[..., 1::2]
    x_rotated_0 = x1 * cos - x2 * sin
    x_rotated_1 = x1 * sin + x2 * cos
    # interleave back
    out = np.empty_like(q_or_k)
    out[..., 0::2] = x_rotated_0
    out[..., 1::2] = x_rotated_1
    return out
