import torch
from model import BigramLanguageModel
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
    decode
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using:", device)

model = BigramLanguageModel(vocab_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

max_iters = 2000
eval_interval = 200

for step in range(max_iters):

    xb, yb = get_batch('train')
    xb = xb.to(device)
    yb = yb.to(device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print(f"step {step}, loss {loss.item()}")

# generate text after training
start = torch.zeros((1,1), dtype=torch.long).to(device)
generated = model.generate(start, max_new_tokens=300, device=device)
from data import decode
print(decode(generated[0].tolist()))

