import numpy as np

def get_attention_average(attn_weights_all):
    """
    attn_weights_all: list of attn weight arrays per layer, each (batch, heads, seq, seq)
    returns average over layers and heads: (batch, seq, seq)
    """
    stacked = np.stack(attn_weights_all, axis=0)  # (layers, batch, heads, seq, seq)
    avg = np.mean(stacked, axis=(0,2))  # mean over layers and heads -> (batch, seq, seq)
    return avg