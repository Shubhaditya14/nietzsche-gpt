🧠 Nietzsche-GPT

A miniature GPT-style Transformer trained from scratch on Nietzsche’s writings.

This project implements a decoder-only Transformer (like GPT-2) entirely in PyTorch.
The model learns to predict the next character, allowing it to generate text in a Nietzsche-like philosophical style.

🚀 Method

Build a character-level vocabulary from nietzsche.txt

Convert text ↔ integers using custom encode/decode

Train a GPT block with:

Multi-Head Self-Attention

Causal masking

Feedforward MLP

Residual connections + LayerNorm

Optimize using cross-entropy and AdamW

Generate text by sampling one character at a time from model logits

💬 Example Output
what is truth, feelings; the false in the most great nature...

🏁 Run

Train:

python src/train.py


Interact:

python src/interact.py

❤️ Credits

Inspired by Andrej Karpathy’s nanoGPT lecture.
