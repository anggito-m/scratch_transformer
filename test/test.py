
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


print("\n=== CEK DIMENSI TENSOR ===")
print("token_ids:", token_ids)
print("logits.shape:", logits.shape)     # (2, 10, 64)
print("attention shape (layer, batch, head, seq, seq):", [a.shape for a in attn])
attn_avg = get_attention_average(attn)
print("attn_avg shape:", attn_avg.shape)  # (batch, seq, seq)
assert logits.shape == (batch, seq, vocab_size), "Logits shape mismatch"
assert attn_avg.shape == (batch, seq, seq), "Attention average shape mismatch"

print("=== CEK HASIL SOFTMAX ===")
print(logits[0, -1]) 
probs = softmax(logits[0, -1])
print("sum probs (should be 1):", probs.sum())
assert np.allclose(probs.sum(), 1.0, atol=1e-6), "Softmax output invalid"

print("\n=== CEK MASKING (CAUSAL MASK) ===")
seq = 10
# boolean mask: True = masked (future), False = visible
causal_mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)

print("Causal mask (1 = masked, 0 = visible):")
print(causal_mask.astype(int))

# sanity: jumlah elemen yang dimask = seq*(seq-1)/2
expected = seq * (seq - 1) // 2
assert np.sum(causal_mask) == expected
print("causal mask shape OK and count OK.")

print("\n=== VISUALISASI ATTENTION ===")
token_strs = [f"T{i}" for i in token_ids[0]]
last_layer_attn = attn[-1]  # layer terakhir
visualize_attention(last_layer_attn, token_strs, head=0)
