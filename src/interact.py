import torch
from model import miniGPT
import sys
import os

# Let Python know the parent directory exists for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import data
from data import (
    get_batch,
    vocab_size,
    block_size,
    train_data,
    val_data,
    encode,
    decode,
    get_data_splits
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using:", device)

# --------------------------
# Load your trained miniGPT
# --------------------------
model = miniGPT(
    vocab_size=vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    block_size=128,          # <-- IMPORTANT MATCH
).to(device)

model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

print("Model loaded. Type your prompt below.\n")

# --------------------------
# Chat loop
# --------------------------
while True:
    prompt = input("You: ").strip()
    if prompt == "":
        continue
    if prompt.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    # encode prompt → tokens
    prompt_tokens = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

    # generate continuation
    out = model.generate(prompt_tokens, max_new_tokens=200, device=device)

    # decode to text
    text = decode(out[0].tolist())

    print("\nNietzscheGPT:", text)
    print("\n" + "-"*50 + "\n")
