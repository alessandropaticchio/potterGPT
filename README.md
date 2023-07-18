# PotterGPT ⚡

This repository contains an educational project I built in order to learn how GPT works. 
I basically rewrote its architecture from scratch in order to understand all the nitty-gritty details
of GPT.
The output model is a Toy Language Model which is trained on the 1st book of the Harry Potter saga, and 
is capable of generating plausible sequences of characters, even though they don't make much sense 
given my limited computational resources.

I thought of sharing this repository along with some comments about its implementation as I may find it useful
in the future to refresh some concepts.

You can watch a demo of **potterGPT** [here](https://youtu.be/LttgYSZfnF4)!

## GPT Architecture
GPT is a Transformer-based model. It is trained in order to give the next most likely token, given an initial sequence of tokens.

### Tokenization 🧩
Now, our input is generally plain-text. However neural networks accepts numbers as input.
How do we go from one to another?

First, we need to assign a unique identifier to each and every character/sequence-of-characters/word in our
dataset. For this specific project, I adopted a simple character-level tokenization, that assigns
an integer to every character that is present in my input data.


```
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
```

However, more sophisticated models use better tokenizer.
Now we just need to assign an embedding to each and every token: we can do that with an Embedding matrix,
with ```vocab_size``` entries:

```
self.token_embedding_table = nn.Embedding(self.config.vocab_size, self.config.n_embd)
```

The ```token_embedding_table``` will be used to encode every token before passing through 
the network into an embedding vector of size ```n_embd```.

At this point, though, we need to encode the sequentiality of the tokens. For our specific case, we do that 
by using an additional matrix with ```vocab_size``` entries.

```
self.position_embedding_table = nn.Embedding(self.config.block_size, self.config.n_embd)
```

Every entry of ```position_embedding_table``` has ```block_size``` elements, where ```block_size``` represents
the context-length of my model, a.k.a. the maximum length my model can process.

After the input sample is passed through the two embedding matrices, their outputs is summed:

```
tok_emb = self.token_embedding_table(idx)  # (B,T,C)
pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
x = tok_emb + pos_emb  # (B,T,C)
```

### Attention mechanism 🔍 
Attention mechanism is a communication-mechanism which makes information flow in-between tokens (or words, put it simply).
Let's look at an example:

Say that this is our input:

```
input_tokens = [10, 43, 22, 1, 44, 15, 2, 33]
```

This input basically contains also the target variable, given our target task.
In fact, the input contains the information that token ```[10]``` is followed by 
```[43]```, that tokens ```[10, 43]``` are followed by ```[22]```, that tokens 
```[10, 43, 22]``` are followed by token ```[1]```, and so on and so forth.

We would like that each token to incorporate information from previous ones.
How do we do that?

Attention mechanism achieves that by using 3 matrices:

```
self.key = nn.Linear(self.config.n_embd, head_size, bias=False)  # As a token, what information do I have?
self.query = nn.Linear(self.config.n_embd, head_size, bias=False)  # As a token, what am I looking for?
self.value = nn.Linear(self.config.n_embd, head_size, bias=False)  # As a token, what do I do with the information I found?
```

Whenever the embedding of a token is passed through these matrices, it's both (i) passing information
to other tokens and (ii) retrieving information from other tokens.

The key point here is that each token should only communicate with preceding tokens, since we want to predict the next one.
This is why we say that GPT is a Decoder-only transformer architecture, and it's the main difference from BERT, which is an Encoder-only
architecture, where information is not stopped from flowing also from subsequent tokens.
This is achieved by using a triangular matrix like this (for 8-th length samples):

```
wei = [[1, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 0, 0, 0],
[1, 1, 1, 1, 1, 1, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 0],
[1, 1, 1, 1, 1, 1, 1, 1],
```

and in code the whole forward pass through the whole attention head is:

```
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
```

Turns out that one attention head might not be enough, since
a given token might want to look for different types information. Fear not: you can just pass the input
through multiple attention heads, concatenate their outputs all together and pass it through a linear layer:

```
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
```

The ```MultiHeadAttention``` component is the key part of the ```TransformerBlock```, that 
further process the input by normalizing it and passing it through an additional FeedForward step.
Also, residual connections are used to optimize training.

```
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
```

### Training 🧠 
What I personally find incredible is the simplicity of the idea behind the training of such a powerful model.
As I mentioned already, these kind of models are trained to iteratively predict the next most likely token.
This means that the output should be a probability distribution across the whole vocabulary.
And how do I measure the quality of this distribution? With cross-entropy! So this is basically a classification task!

```
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
```

Simply enough, each sample is passed through the whole network and, for each token, we output a 
```vocab_size```-long vector, that represents a probability distribution for the next token. 
We compare this distribution with our target (which will simply be a 0s-vector with just one 1), and measure the loss function.
That's it! 

The actual training mechanism will not be any different from some we've encountered before:

```
def train(self):
    optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    for iter in range(self.max_iters):
        print(f"Epoch {iter}...")
        # every once in a while evaluate the loss on train and val sets
        if iter % self.eval_interval == 0 or iter == self.max_iters - 1:
            losses = self.estimate_loss()
            print(f">> Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = self.get_batch('train')

        # evaluate the loss
        logits, loss = self.model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
```

### Generation 📝

Here we go to the fun-part! Generating new text is incredibly simple: 
we just need to feed the model with an initial sequence of tokens, sampling from the model's
output distribution and decoding it... that's it!

```
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
    
context = torch.LongTensor(encode("Petunia said to Harry:"), device=device).unsqueeze(0)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
```