import torch
import torch
import torch.nn as nn
from torch.nn import functional as F

# Constants
torch.manual_seed(1337)
batch_size = 4  # how many independent sequences to process in parallel
block_size = 8  # What is the max context length?
eval_iters= 300 
max_iters = 3000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e2
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

class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx) #(B,T,C)

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
      logits , loss = self(idx)
      logits = logits[:,-1,:] #(B,C)
      probs = F.softmax(logits,dim = 1) #(B,C)
      idx_next = torch.multinomial(probs,num_samples = 1) #(B,1)
      idx = torch.cat((idx,idx_next), dim = 1) # (B, T+1)
    return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

for steps in range(max_iters):
  xb,yb = get_batch('train')

  if(steps % eval_iters == 0):
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  #evaluate the loss
  logits , loss = m(xb,yb)
  optimizer.zero_grad(set_to_none = True)
  loss.backward()
  optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
