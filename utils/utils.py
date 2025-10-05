import numpy as np

def xavier_init(in_dim, out_dim):
    std = np.sqrt(2.0 / (in_dim + out_dim))
    return np.random.randn(in_dim, out_dim).astype(np.float32) * std

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)