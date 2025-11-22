# nietzsche-gpt
🧠 Nietzsche-GPT

A miniature GPT language model trained from scratch on Nietzsche’s writings.

This project implements a decoder-only Transformer entirely from scratch in PyTorch, trained on a Nietzsche text dataset (~1MB).
The model learns to predict the next character using self-attention, and generates original text in a Nietzsche-like style.

🚀 Features

Full Transformer architecture implemented manually

Multi-Head Self-Attention (MHSA)

Causal masking for autoregressive generation

Feedforward MLP layers

Residual connections + LayerNorm

Learned token + positional embeddings

Character-level tokenizer

Training pipeline from scratch

Interactive CLI for chatting with NietzscheGPT

📚 Methodology
1️⃣ Dataset

Simple UTF-8 .txt file (nietzsche.txt)

Character-level vocabulary (each unique character → token)

We build:

stoi (char → id)

itos (id → char)

encode(string)

decode(list[int])

The full dataset is encoded to integers and split:

90% training

10% validation

2️⃣ Objective: Next-Token Prediction

The model is trained to:

Predict the next character given previous characters.

This forces the model to learn:

grammar

structure

dependencies

Nietzsche’s writing style

We use:

Cross-entropy loss

AdamW optimizer

3️⃣ Model Architecture
🔷 miniGPT (GPT-style Transformer)
Token Embeddings
+ Position Embeddings
↓
[ Transformer Block × N ]
↓
LayerNorm
↓
Linear head → vocabulary logits

4️⃣ Self-Attention (Core Mechanism)

Each token produces vectors:

Query Q

Key K

Value V

Attention:

A = softmax( Q · Kᵀ / sqrt(d_k) )
Output = A @ V

⛔ Causal Mask

The model cannot attend to future tokens.
We use torch.tril() to enforce autoregressive behavior.

5️⃣ Multi-Head Attention (MHA)

Multiple attention heads:

capture different relationships

concatenate results

project back to embedding dimension

6️⃣ Feedforward Neural Network (MLP)

Each token independently passes through:

Linear → GELU → Linear


Expanded width (4× embedding size) adds non-linearity and reasoning ability.

7️⃣ Residual Connections + LayerNorm

We use the GPT-2 pre-norm architecture:

x = x + MHA( LayerNorm(x) )
x = x + MLP( LayerNorm(x) )


Benefits:

stable training

deeper networks train effectively

8️⃣ Training Procedure

Training loop:

Sample random sequences (block_size)

Predict the next token

Compute loss

Backpropagate

Update weights

Validation loss is logged periodically to monitor learning.

9️⃣ Text Generation (Sampling)

Generation uses:

Last-step logits

Softmax to get probabilities

torch.multinomial to sample next character

Append token → repeat

Example output:

what is truth, feelings; the false in the most great
nature.
Our books Her last of trausifically to them ext...

💬 Interactive Mode

Use interact.py to chat with NietzscheGPT:

You: what is truth?
NietzscheGPT: what is truth, feelings; the false in the most great nature...

🛠 Project Structure
nietzsche-gpt/
│
├── data.py               # dataset, vocab, encode/decode
├── model.py              # full Transformer implementation
│
├── src/
│   ├── train.py          # training loop
│   ├── interact.py       # interactive CLI chat
│
├── nietzsche.txt         # dataset
└── README.md             # this file

📈 Results

Learns recognizable Nietzsche-like structure

Semi-coherent philosophical phrases

Val loss improves steadily

Can be scaled easily for better output