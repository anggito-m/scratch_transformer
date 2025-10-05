import numpy as np
from utils.causal_padding_mask_utils import causal_mask, padding_mask_from_ids
from utils.utils import xavier_init
from transformer.layer import DecoderBlock, LayerNorm
from transformer.embed import TokenEmbedding
from transformer.embed import PositionalEncoding


class DecoderOnlyTransformer:
    def __init__(self, vocab_size, max_seq_len, d_model=64, num_heads=8, d_ff=256, num_layers=2,
                 pos_kind="sinusoidal", tie_weights=False, pad_id=None):
        """
        pos_kind: "sinusoidal" | "learned" | "rope"
        tie_weights: if True, tie embedding matrix and output projection
        """
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pos_kind = pos_kind
        self.pad_id = pad_id

        self.embedding = TokenEmbedding(vocab_size, d_model, pad_id=pad_id)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model, kind=pos_kind)
        self.blocks = [DecoderBlock(d_model, num_heads, d_ff, rope=(pos_kind=="rope")) for _ in range(num_layers)]
        self.ln_f = LayerNorm(d_model)
        # output projection
        self.W_out = xavier_init(d_model, vocab_size)
        self.b_out = np.zeros((vocab_size,), dtype=np.float32)

        # weight tying option
        self.tie_weights = tie_weights
        if self.tie_weights:
            # set W_out to transpose view of embedding matrix (we keep pointer)
            # Implementation detail: we'll ensure on forward we use embedding.W.T
            pass

    def forward(self, token_ids):
        # token_ids: (batch, seq)
        b, seq = token_ids.shape
        x = self.embedding.forward(token_ids)  # (b, seq, d_model)
        if self.pos_kind in ("sinusoidal", "learned"):
            pos = self.pos_encoding.forward(seq)  # (seq, d_model)
            x = x + pos[None, :, :]
        # causal mask (seq, seq)
        c_mask = causal_mask(seq)
        # padding mask (batch, seq) if pad_id given
        pad_mask = None
        if self.pad_id is not None:
            pad_mask = padding_mask_from_ids(token_ids, self.pad_id)  # True where pad

        attn_weights_all = []
        for blk in self.blocks:
            x, attn_weights = blk.forward(x, causal_mask_mat=c_mask, padding_mask=pad_mask, pos_kind=self.pos_kind)
            attn_weights_all.append(attn_weights)
        x = self.ln_f.forward(x)
        # output projection: optionally tie weights
        if self.tie_weights:
            logits = x @ self.embedding.W.T + self.b_out  # tie weights
        else:
            logits = x @ self.W_out + self.b_out
        return logits, attn_weights_all