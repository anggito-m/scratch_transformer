import numpy as np

def causal_mask(seq_len):
    # returns boolean mask where True = masked (future positions)
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

def padding_mask_from_ids(token_ids, pad_id):
    # token_ids: (batch, seq) -> returns mask (batch, seq) where True if pad
    return (token_ids == pad_id)
