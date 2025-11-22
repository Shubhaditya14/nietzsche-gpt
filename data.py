import torch

# load dataset
with open('nietzsche.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Length of dataset in characters:", len(text))

# get all unique characters in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Unique characters:", vocab_size)
print(chars)

# create mappings from characters to integers, and back
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    """string to list of ints"""
    return [stoi[c] for c in s]

def decode(l):
    """list of ints to string"""
    return ''.join([itos[i] for i in l])

import torch

# entire dataset encoded as integers
data = torch.tensor(encode(text), dtype=torch.long)
print("Tensor shape:", data.shape)
print("Example:", data[:100])

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 64  # you can change later
batch_size = 32  # you can change later

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    
    return x, y

# test
xb, yb = get_batch('train')
print(xb.shape, yb.shape)
print(xb[0])
print(yb[0])

def get_data_splits():
    # load file
    with open("nietzsche.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # get unique chars
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # build mappings
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join(itos[i] for i in l)

    # encode entire dataset
    data = torch.tensor(encode(text), dtype=torch.long)

    # 90% split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data   = data[n:]

    return train_data, val_data, vocab_size, encode, decode

def encode(s):
    """string → list of ints"""
    return [stoi[c] for c in s]

def decode(l):
    """list of ints → string"""
    return ''.join([itos[i] for i in l])
