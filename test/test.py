
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from transformer.model import DecoderOnlyTransformer
from utils.utils import softmax
from helper.attention_average_helper import get_attention_average
from helper.attention_visualization_helper import visualize_attention

np.random.seed(1)
vocab_size = 64
model = DecoderOnlyTransformer(vocab_size=vocab_size, max_seq_len=16, d_model=64,
                               num_heads=8, d_ff=256, num_layers=2,
                               pos_kind="sinusoidal", tie_weights=True, pad_id=0)
batch = 2
seq = 10
token_ids = np.random.randint(1, vocab_size, size=(batch, seq))
# add pad in second example
token_ids[1, 7:] = 0
logits, attn = model.forward(token_ids)

print("token_ids:", token_ids)
print("logits.shape:", logits.shape)   # (2, 10, 64)
probs = softmax(logits[0, -1])
print("sum probs (should be 1):", probs.sum())
attn_avg = get_attention_average(attn)
print("attn_avg shape:", attn_avg.shape)  # (batch, seq, seq)
# mapping id ke "token string" sederhana
token_strs = [f"T{i}" for i in token_ids[0]]

# visualisasi attention dari layer terakhir, head ke-0
last_layer_attn = attn[-1]  # ambil dari decoder block terakhir
# print(last_layer_attn)
visualize_attention(last_layer_attn, token_strs, head=0)
