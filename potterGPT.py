from torch import nn
from torch.nn import functional as F
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GPTConfig:
    def __init__(self, n_layers, n_embd, block_size, n_head, vocab_size, dropout):
        self.n_layers = n_layers
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.vocab_size = vocab_size
        self.dropout = dropout

class PotterGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.position_embedding_table = nn.Embedding(self.config.block_size, self.config.n_embd)

        self.blocks = nn.Sequential(*[TransformerBlock(self.config) for _ in range(self.config.n_layers)])

        self.ln_f = nn.LayerNorm(self.config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, config, head_size):
        super().__init__()
        self.config = config

        self.key = nn.Linear(self.config.n_embd, head_size, bias=False)  # As a token, what information do I have?
        self.query = nn.Linear(self.config.n_embd, head_size, bias=False)  # As a token, what am I looking for?
        self.value = nn.Linear(self.config.n_embd, head_size, bias=False)  # As a token, what do I do with the information I found?

        # Essential for decoder structure, it prevents information from future tokens from flowing in
        self.register_buffer('tril', torch.tril(torch.ones(self.config.block_size, self.config.block_size)))

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape

        k = self.key(x)  # (B,T, head_size)
        q = self.query(x)  # (B,T, head_size)
        # compute attention scores, normaizing by the shape of K
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        # This step prevents information from future tokens from flowing in
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, config, head_size):
        super().__init__()
        self.config = config
        # Initialize all attention heads
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(self.config.n_head)])
        self.proj = nn.Linear(head_size * self.config.n_head, self.config.n_embd)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        # Forward steps through all the attention heads and then concatenate
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(self.config.n_embd, 4 * self.config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * self.config.n_embd, self.config.n_embd),
            nn.Dropout(self.config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.config = config

        head_size = self.config.n_embd // self.config.n_head
        self.self_attention_heads = MultiHeadAttention(self.config, head_size)
        self.ffwd = FeedForward(self.config)
        self.ln1 = nn.LayerNorm(self.config.n_embd)
        self.ln2 = nn.LayerNorm(self.config.n_embd)

    def forward(self, x):
        x = x + self.self_attention_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
