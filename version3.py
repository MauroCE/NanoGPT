import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print("DEVICE: ", device)
eval_iters = 200
n_embd = 384  # each head is 384//6 = 64 dimensional, which is standard
n_layers = 6
n_head = 6
dropout = 0.2  # 20% of neurons are dropped out

# If you are using a Conda environment generated from scratch, you will need to run
# `conda install jupyter` and `conda install wget`
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# Read the text
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Grab all unique characters in the text, sorted, and compute vocabulary size
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Dictionaries mapping characters to their index and vice versa
str_to_int = {character: integer for integer, character in enumerate(chars)}
int_to_str = {integer: character for integer, character in enumerate(chars)}

# Encoder and Decoder functions
encode = lambda string: [str_to_int[character] for character in string]  # string --> list(int)
decode = lambda intlist: ''.join([int_to_str[integer] for integer in intlist])  # list(int) --> string


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]  # 90% training
val_data = data[n:]    # 10% validation


def get_batch(split):
    """Generates batch of data of inputs `x` and targets `y`."""
    data = train_data if split == "train" else val_data
    # Sample 4 integers from [0, n-block_size], representing off-sets, one for each batch
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    # Grab context and target
    context = torch.stack([data[i:i+block_size] for i in ix])  # (batch_size, block_size) = (4, 8)
    targets = torch.stack([data[i+1:i+block_size+1] for i in ix])  # (batch_size, block_size) = (4, 8)
    context, targets = context.to(device), targets.to(device)
    return context, targets


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """One head of self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Basically, using register buffer it is not treated as a parameter
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # randomly prevent some of the nodes from communicating with a dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, C)
        q = self.query(x)   # (B, T, C)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # Weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v      # (B, T, T) @ (B, T, C) = (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads fo self-attention, in parallel."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout) # add a dropout typically added right before the residual connection.

    def forward(self, x):
        # remember the output of each head is (B, T, C) so here we are concatenating the output on the final dimension
        # thus obtaining (B, T, num_heads*C)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # linear projection of the output
        out = self.proj(out)
        # dropout before residual/skip connection
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # see
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            # add a dropout typically added right before the residual connection.
            nn.Dropout(dropout)
        )
        # to understand why 4*n_embd see section 3.3 "Position-wise Feed-Forward Networks" in the
        # "Attention is All You Need" paper. There n_embd=512 and dff=2048

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""
    def __init__(self, n_embd, n_head):
        """Here n_embd is the embedding dimension and n_head is the number of heads."""
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(num_heads=n_head, head_size=head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # per-token transformation that normalizes the features
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # layer norm + self-attention + residual connection
        x = x + self.ffwd(self.ln2(x))  # layer norm + feed-forward + residual connection
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self):
        """Bigram model, see Karpathy's previous series of videos."""
        super().__init__()
        # Tokens read off the logits for the next token from a lookup table
        # Token embedding table has size (vocab_size, vocab_size)
        # The way it works is that the input, say 24 (the first one in xb above) will take the 24th row of this
        # embedding table.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # We now also encode the position. Each position from 0 to block_size-1 will have a corresponding embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)  # there should always be a layer norm at the end of the transformer but before the final linear layer
        self.lm_head = nn.Linear(n_embd, vocab_size)  # lm=language model

    def forward(self, idx, targets=None):
        """Forward pass. Takes `idx` and `targets` which are both `(B, T)` tensors of integers.
        Here `B` is the batch_size and `T` should be the block/context length."""
        B, T = idx.shape
        # PyTorch will grab the row corresponding to the indices provided and return logits in
        # the shape (batch, time, channel). Here batch=4, time=8, channel=65 (vocab size)
        # The logits here are like the scores for the next token in the sequence
        tok_emb = self.token_embedding_table(idx)  # (B, T, C=embedding_dimension), these re token embeddings now.
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Negative log-likelihood loss (cross-entropy). Importantly, when working with multi-dimensional inputs,
        # PyTorch's documentation https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        # mentions that it requires dimensions (B, C, T) using our notation. A simpler alternative is to simply shape it to (B*T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Here `idx` is the current context of tokens in some batch, so it is `(B, T)`. This function will continue the generation
        one by one, for both the B and T dimensions. It keeps doing this until max_new_tokens."""
        for _ in range(max_new_tokens):
            # We need to make sure that the idx that we feed into the model is the same size as the context
            idx_cond = idx[:, -block_size:]  # (B, T) --> (B, block_size)
            logits, loss = self(idx_cond)   # Get the predictions (calls forward(idx, targets=None))
            logits = logits[:, -1, :]  # (B, T, C) --> (B, C) we focus only on the last "time step" (last token in the context)
            probs = F.softmax(logits, dim=-1)  # Use Softmax to get probabilities. (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample using the probabilities (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # append the sampled index to the running sequence (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter} train loss: {losses['train']:.4f} val loss: {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))









