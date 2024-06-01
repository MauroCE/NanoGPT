import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        """Bigram model, see Karpathy's previous series of videos."""
        super().__init__()
        # Tokens read off the logits for the next token from a lookup table
        # Token embedding table has size (vocab_size, vocab_size)
        # The way it works is that the input, say 24 (the first one in xb above) will take the 24th row of this
        # embedding table.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """Forward pass. Takes `idx` and `targets` which are both `(B, T)` tensors of integers.
        Here `B` is the batch_size and `T` should be the block/context length."""
        # PyTorch will grab the row corresponding to the indices provided and return logits in
        # the shape (batch, time, channel). Here batch=4, time=8, channel=65 (vocab size)
        # The logits here are like the scores for the next token in the sequence
        logits = self.token_embedding_table(idx)  # (B, T, C)

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
            logits, loss = self(idx)   # Get the predictions (calls forward(idx, targets=None))
            logits = logits[:, -1, :]  # (B, T, C) --> (B, C) we focus only on the last "time step" (last token in the context)
            probs = F.softmax(logits, dim=-1)  # Use Softmax to get probabilities. (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample using the probabilities (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # append the sampled index to the running sequence (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
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









