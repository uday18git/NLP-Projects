import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 #64 # how many independent sequences will we process in parallel?
block_size = 128#256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
eval_iters = 200
n_layer = 6
n_head = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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
# single head of self-attention
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
# multi head attention is just a aggregation of multiple heads running in parallel
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # projection is just linear transformation of the output
        out = self.dropout(self.proj(out))
        return out
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd),# inner layer should be multiplied by 4 according to the paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),# multiplication factor should be 4 in the inner layer
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
# super simple bigram model
class GPTLanguageModel(nn.Module):

    def __init__(self):# removed vocab_size because its a global variable
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)# we dont want to go directly to vocab size o/p , well put a intermediate layer of n_embd
        self.position_embedding_table = nn.Embedding(block_size, n_embd)# we want to go from block_size to n_embd
        # self.sa_head = Head(n_embd)
        # self.sa_heads = MultiHeadAttention(4,n_embd//4) # i.e 4 heads of 8-dimensional self-attention
        # self.ffwd = FeedFoward(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer) ]) # we are applying feed forward and multi head attention many many times
        self.ln_f = nn.LayerNorm(n_embd) # right before the end of the transformer
        self.lm_head = nn.Linear(n_embd, vocab_size)# we want to go from n_embd to vocab_size

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        # B is the batch size, T is the sequence length
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C) this c and below c are not equal c is the embedding size here
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C) so here broadcasting takes place 1,T,C is created then broadcasted through B
        # this currently wont help because we have a bigram model and it is translation invariant , as we work on the self attention block this will be helpful
        # now see
        # x = self.sa_heads(x) # applied one head of self attention (B,T,C), simplest way to add self attention
        # x = self.ffwd(x) # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size) here c is the vocab size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits,loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, "M parameters")
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# changes log
# after introducing a intermediate layer -> train loss 2.5006, val loss 2.5117
# so far we have taken indices and we have encoded them based on the identities of the tokens.
# next we will encode them also using thier position in the sequence
# we applied a single head the train loss ->2.4 val loss ->2.44
# we applied multi head the train loss ->2.2424 val loss -> 2.2696
# when we added a feed forward train loss ->2.2179 val loss ->2.2359 , better
# when we add many blocks of feed forward and multi head attention , it does not do that well because now our network is becoming deep and we need to add resnets and dropout and norm
# when we apply resnets with the blocks train loss -> 1.9727 val loss ->2.0686 ,much better , and we also see that it is overfitting a little bit so we add layer norm
# layer norm is similar to batch norm
# spoiler it is very complicated lol
# in call function of batch norm we change the 0 to 1 to make layer norm
# we use xmean= x.mean(dim=1, keepdim=True)
# we use xvar= x.var(dim=1, keepdim=True) we normalize the rows
# in paper we see that the layer norm is applied before the feed forward and after the multi head attention but we are applying it after the feed forward and before the multi head attention it is nowadays common to do so.
# after adding layer norm train loss ->1.9883 val loss -> 2.0828 idk why
# we reach 1.2614 train loss and 1.5291 val loss when trained after scaling the model