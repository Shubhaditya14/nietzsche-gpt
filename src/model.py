import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embed=128):
        super().__init__()
        # token embedding table (learned lookup table)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # final linear layer projecting embeddings to logits
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (batch_size, block_size) with ints
        # get embeddings: (batch_size, block_size, n_embed)
        tok_emb = self.token_embedding_table(idx)

        # logits for all positions
        logits = self.lm_head(tok_emb)  # (batch_size, block_size, vocab_size)

        # if no targets → inference mode → return logits only
        if targets is None:
            return logits, None

        # reshape for cross-entropy
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)

        loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, device):
        # idx: (batch, current_length)
        for _ in range(max_new_tokens):
            # get latest logits
            logits, _ = self.forward(idx)

            # focus on last time-step
            logits = logits[:, -1, :]  # (B, vocab_size)

            # convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # sample
            next_token = torch.multinomial(probs, num_samples=1)  # (B,1)

            # append to sequence
            idx = torch.cat((idx, next_token), dim=1)

        return idx
    
class SelfAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_size):
        super().__init__()

        # 1) Linear projections: turn embeddings into Q, K, V
        self.key   = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        # Causal mask: prevent attention to future tokens
        self.register_buffer('tril', torch.tril(torch.ones(1024, 1024)))

    def forward(self, x):
        B, T, C = x.shape   # batch, time, channels

        # 2) Project input embeddings into K, Q, V
        K = self.key(x)     # (B, T, head_size)
        Q = self.query(x)   # (B, T, head_size)

        # 3) Compute attention scores (similarity)
        #    Q @ K^T = (B, T, T)
        att = Q @ K.transpose(-2, -1)

        # Scale by sqrt(head_size) (stabilizes softmax)
        att = att / math.sqrt(K.size(-1))

        # 4) Apply causal mask: block future positions
        att = att.masked_fill(self.tril[ :T, :T] == 0, float('-inf'))

        # 5) Softmax → attention weights
        att = torch.softmax(att, dim=-1)

        # 6) Weighted sum of V
        V = self.value(x)
        out = att @ V  # (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, head_size):
        super().__init__()

        # Create multiple self-attention heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_dim, head_size) 
            for _ in range(num_heads)
        ])

        # After concatenating all heads, project back to embed_dim
        self.proj = nn.Linear(num_heads * head_size, embed_dim)

    def forward(self, x):
        # Run all heads in parallel and concatenate their outputs
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # Final linear projection mixes head outputs
        out = self.proj(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # expand
            nn.GELU(),                            # nonlinearity (better than ReLU)
            nn.Linear(4 * embed_dim, embed_dim)   # project back down
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_size = embed_dim // num_heads

        self.sa = MultiHeadAttention(num_heads, embed_dim, head_size)
        self.ff = FeedForward(embed_dim)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # residual 1
        x = x + self.ff(self.ln2(x))   # residual 2
        return x

class miniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4, block_size=128):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads) for _ in range(num_layers)]
        )

        self.ln_f = nn.LayerNorm(embed_dim)  # final norm
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, device):
        for _ in range(max_new_tokens):
            # crop idx to block_size
            idx_cond = idx[:, -self.block_size:]
            
            # get logits
            logits, _ = self(idx_cond)

            # take last timestep
            logits = logits[:, -1, :]

            # softmax → probabilities
            probs = F.softmax(logits, dim=-1)

            # sample
            next_token = torch.multinomial(probs, num_samples=1)

            # append
            idx = torch.cat((idx, next_token), dim=1)

        return idx

