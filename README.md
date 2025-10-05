# Transformer from Scratch (NumPy)
Implementation of a decoder-only (GPT-style) Transformer from scratch using NumPy.

## Report
[Click Here](https://drive.google.com/file/d/1aLoeRc4Xih1oJ2a8Cb6Pt1zs2wBhfy5G/view?usp=sharing).

## Requirements
- Python 3.8+
- NumPy
- (Optional) matplotlib for plotting attention

## How to run
1. Install dependencies:
   pip install numpy
   # optional
   pip install matplotlib

2. Run sanity checks:
   cd test
   python transformer_test.py

4. Run tests:
   cd test
   python tests.py

## Features
- Token embedding, positional encoding (sinusoidal / learned / RoPE)
- Multi-head scaled dot-product attention with causal + padding mask
- Feed-forward network, LayerNorm (pre-norm)
- Weight tying option (embed <-> output)
- Attention visualization helper (optional)

## Notes
All math operations are implemented using NumPy only.

## Proof of Test Results
1. Cek Dimensi Tensor
   <img width="747" height="115" alt="Cuplikan layar 2025-10-05 224931" src="https://github.com/user-attachments/assets/a7c160c9-62c6-4176-89ea-5d7a89c8ee15" />

2. Cek Softmax
   <img width="681" height="293" alt="Cuplikan layar 2025-10-05 225858" src="https://github.com/user-attachments/assets/6bc27ee6-43cf-4e3e-addb-c668b7c05720" />

3. Cek Masking
   <img width="358" height="271" alt="Cuplikan layar 2025-10-05 230552" src="https://github.com/user-attachments/assets/8068a2f2-aeec-4e9a-8f00-77d0b26cfe5d" />

4. Visualisasi Distribusi Masking
   <img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/1684f41f-d4c2-4832-97c3-b028c220342c" />
