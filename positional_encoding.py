import numpy as np

class PositionalEncoding:
    def __init__(self, max_len, d_model):
        self.max_len = max_len
        self.d_model = d_model
        self.pe = self._build_pe(max_len, d_model)
        
    def _build_pe(self, max_len, d_model):
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len)[:, None]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe  # shape (max_len, d_model)
    
    def forward(self, seq_len):
        return self.pe[:seq_len]  # (seq, d_model)
