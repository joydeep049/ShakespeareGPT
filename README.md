# ShakespeareGPT

A Generatively Pre-Trained Transformer based on the paper "Attention is all you need"
Trained On Shakespeare's famous 'Coriolanus'.


## Mathematical Trick In Self-Attention

1. In our bigram model, the next token is predicted based on only the previous token.
But here, we want our current token to be affected by all of the previous tokens in that particular sequence.

So, we create a bag of words list:

```bash
# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B,T,C))
for b in range(B):
  for t in range(T):
    xprev = x[b, :t+1] #(t,C)
    xbow[b,t] = torch.mean(xprev,0)
```
where we want our current token value `x[b,t]` to reflect the context of all
the previous values in the sequence `x[b, : t+1]`

<b?>Conceptual Example:</b>
Let's assume you have a sequence of words, and you want each word's representation to include information about all the previous words in the sequence.

For instance, if your sequence is ["I", "like", "pizza"], the model would compute:

<ul> For "I" (t=0), the BoW is just the embedding of "I".
<ul> For "like" (t=1), the BoW is the mean of the embeddings of "I" and "like".
<ul> For "pizza" (t=2), the BoW is the mean of the embeddings of "I", "like", and "pizza".

This approach ensures that the representation of each word/token incorporates information from all the words/tokens that have come before it, which can be beneficial for capturing the overall context in tasks like language modeling.

2. Matrix Multiplication as Aggregation

```bash
torch.manual_seed(42)
a = torch.tril(torch.ones(3,3))
a = a / torch.sum(a,1,keepdim = True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
```

This is a pretty neat trick to calculate the mean by using matrix multiplication.
The output was something like:

```bash
a =
tensor([[1., 0., 0.],
        [1., 1., 0.],
        [1., 1., 1.]])

a after change = 
tensor([[1.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000],
        [0.3333, 0.3333, 0.3333]])
b =
tensor([[2., 7.],
        [6., 4.],
        [6., 5.]])
c =
tensor([[2.0000, 7.0000],
        [4.0000, 5.5000],
        [4.6667, 5.3333]])

```



