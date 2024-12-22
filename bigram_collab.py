import torch
import torch
import torch.nn as nn
from torch.nn import functional as F

# Constants
torch.manual_seed(1337)
batch_size = 64  # how many independent sequences to process in parallel
block_size = 256 # What is the max context length?
eval_iters= 500 
max_iters = 5000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 3e-4
n_embd = 384
num_heads = 6
head_size =  64
dropout = 0.2
# ---------

# !wget https://raw.githubusercontent.com/joydeep049/ShakespeareGPT/master/input.txt

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l: ''.join([itos[n] for n in l])

data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train_data = data[0:n]
val_data = data[n:]
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
  context = x[:t+1]
  target = y[t]
  print(f"when input is {context} the target: {target}")

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+1+block_size] for i in ix])
  return x,y

xb,yb = get_batch('train')
print("inputs:")
print(xb.shape)
print(xb)

print("targets:")
print(yb.shape)
print(yb)

for b in range(batch_size):
  for t in range(block_size):
    context = xb[b, :t+1] #Our input to transformer
    target = yb[b,t]
    print(f"When context is {context.tolist()} target is {target}")

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

class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd,n_embd * 4),
      nn.ReLU(),
      nn.Linear(n_embd * 4, n_embd),
    )

  def forward(self,x):
    return self.net(x)


class Head(nn.Module):

  def __init__(self, head_size):
    super().__init__()
    self.query = nn.Linear(n_embd, head_size, bias = False)
    self.key = nn.Linear(n_embd, head_size, bias = False)
    self.value = nn.Linear(n_embd, head_size, bias= False)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    B,T,C = x.shape

    q = self.query(x)
    k = self.key(x)

    tril = torch.tril(torch.ones((T,T)))
    wei = q @ k.transpose(-2, -1) # (B,T,head_size) x (B, head_size, T) --> (B,T,T)
    wei = wei * head_size**-0.5  # Scaled Attention to preserve unit variance

    wei = wei.masked_fill(tril == 0, float('-inf')) # Decoder block. Present cannot talk to future
    wei = F.softmax(wei, dim = -1) # (B,T,T)
    wei = self.dropout(wei)

    v = self.value(x)
    out = wei @ v # (B,T,T) x (B,T,head_size) --> (B,T,head_size)

    return out

class MultiHeadAttention(nn.Module):

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim = -1)
    out = self.dropout(self.proj(out))
    return out

class Block(nn.Module):
  def __init__(self, n_embd, num_heads):
    super().__init__()
    self.head_size = n_embd//num_heads
    self.multi_head_attention = MultiHeadAttention(num_heads, self.head_size) # Communication
    self.ffwd = FeedForward(n_embd) # Contemplation
    self.layerN1 = nn.LayerNorm(n_embd)
    self.layerN2 = nn.LayerNorm(n_embd)

  def forward(self,x):
    x = x + self.multi_head_attention(self.layerN1(x)) # Residual Connection by Adding
    x = x + self.ffwd(self.layerN2(x)) # Residual Connection by Adding 
    return x

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    # self.attention_heads = MultiHeadAttention(4, n_embd//4) # 4 heads of 8-dimensional self-attention
    self.blocks = nn.Sequential(
      Block(n_embd, num_heads=4),
      Block(n_embd, num_heads=4),
      Block(n_embd, num_heads=4),
      nn.LayerNorm(n_embd),
    ) # (B,T,n_embd)
    # self.ffwd = FeedForward(n_embd)
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B,T = idx.shape

    token_embd = self.token_embedding_table(idx) # (B,T,n_embd)
    pos_embd = self.position_embedding_table(torch.arange(T, device= device)) #(T,C)
    x = token_embd + pos_embd # (B,T,C)
    x = self.blocks(x)
    # x = self.attention_heads(x)
    # x = self.ffwd(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) #(B,T,vocab_size)

    if(targets == None):
      loss2 = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss2 = F.cross_entropy(logits,targets)

    return logits,loss2

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits , loss = self(idx_cond)
      logits = logits[:,-1,:] #(B,C)
      probs = F.softmax(logits,dim = 1) #(B,C)
      idx_next = torch.multinomial(probs,num_samples = 1) #(B,1)
      idx = torch.cat((idx,idx_next), dim = 1) # (B, T+1)
    return idx


model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

for steps in range(max_iters):
  xb,yb = get_batch('train')

  if(steps % eval_iters == 0):
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  #evaluate the loss
  logits , loss = m(xb,yb)

  # Training
  optimizer.zero_grad(set_to_none = True)
  loss.backward()
  optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


