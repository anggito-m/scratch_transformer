def plot_attention(attn_matrix, example=0):
    """Optional plotting (requires matplotlib). attn_matrix: (batch, seq, seq)"""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib required for plotting attention") from e
    mat = attn_matrix[example]
    plt.imshow(mat, aspect='auto')
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.colorbar()
    plt.title(f"Attention (example {example})")
    plt.show()
def visualize_attention(attn, tokens, head=0):
    import matplotlib.pyplot as plt
    plt.imshow(attn[0, head], cmap="viridis")
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.colorbar()
    plt.title(f"Attention Head {head}")
    plt.show()
