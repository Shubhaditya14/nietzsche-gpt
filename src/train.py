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
    decode,
    get_data_splits
)
# hyperparameters
batch_size = 32
block_size = 128
learning_rate = 3e-4
max_iters = 5000
eval_interval = 200
device = "cuda" if torch.cuda.is_available() else "mps"

train_data, val_data, vocab_size, encode, decode = get_data_splits()

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

model = miniGPT(
    vocab_size=vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    block_size=block_size
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10)
        for k in range(10):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model.pt")
print("Training complete.")

start = torch.zeros((1,1), dtype=torch.long).to(device)
out = model.generate(start, max_new_tokens=500, device=device)
print(decode(out[0].tolist()))

torch.save(model.state_dict(), "model.pt")
print("Training complete.")