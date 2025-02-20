# GPT Language Model with PyTorch

This repository provides a comprehensive guide to building, training, and generating text with a GPT-style language model implemented from scratch using PyTorch. Whether you're new to transformers or looking to deepen your understanding of self-attention mechanisms, this project offers hands-on experience with state-of-the-art techniques in natural language processing.

![image](https://github.com/user-attachments/assets/0c8e4914-0217-47fa-bbb8-bc8acfffaf31)


## Introduction

The GPT Language Model in this project is a character-level transformer designed to predict the next token in a sequence. It demonstrates key concepts such as:

- **Token and Positional Embeddings:** Mapping input characters to high-dimensional vectors.
- **Self-Attention Mechanism:** Allowing the model to focus on different parts of the input sequence.
- **Transformer Blocks:** Stacking attention and feed-forward layers with residual connections and layer normalization.
- **Text Generation:** Sampling from the learned probability distribution to generate novel text.

By following this tutorial, you’ll learn how each component is built and how they work together to train a language model.

---

## Prerequisites

Before you begin, ensure that you have the following installed:

- **Python 3.7+**
- **PyTorch** (installation instructions available [here](https://pytorch.org/get-started/locally/))
- **NumPy**

Also, prepare a text dataset (for example, a file containing Shakespeare's works) and save it as `input.txt` in an accessible directory.

---

## Installation

Install the required packages using `pip`:

```bash
pip install torch numpy
```
---

## Hyperparameters & Imports

This section sets up the environment and defines the core hyperparameters for the GPT language model. The code snippet below imports the necessary modules and specifies key parameters such as batch size, context length, training iterations, and model dimensions.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# Hyperparameters
batch_size = 64        # Number of sequences processed in parallel
block_size = 256       # Maximum context length for predictions
max_iters = 5000       # Total number of training iterations
eval_interval = 500    # Frequency of evaluation (in iterations)
learning_rate = 3e-4   # Learning rate for the optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200       # Number of iterations for loss estimation

# Model dimensions
n_embd = 384           # Embedding dimension for token representations
n_head = 6             # Number of self-attention heads
n_layer = 6            # Number of transformer blocks
dropout = 0.2          # Dropout rate for regularization

# Set random seed for reproducibility
torch.manual_seed(1337)
```

## Data Loading and Preprocessing
```python
with open(r'C:\Users\吴童\Documents\ai\nanogpt\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```
- **stoi/itos:**
- stoi (string-to-integer) converts characters to indices.
- itos (integer-to-string) performs the reverse.
- **Encoder and Decoder:**
- encode converts a string into a list of integers.
- decode converts a list of integers back into a string.

## Data Loading
```python
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```
- **Purpose:**
-Samples random batches of subsequences from the training or validation set.
- **How it Works:**
-torch.randint selects random starting indices.
-For each index, it extracts a sequence of length block_size for x (input) and the next sequence (shifted by one) for y (target).
-Moves the batches to the chosen device (CPU or GPU).

## Loss Estimation
```python
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
```
- **Purpose:**
- Evaluates the model on both training and validation sets without updating parameters.
- **Key Points:**
- The @torch.no_grad() decorator disables gradient calculations to speed up evaluation.
- model.eval() sets the model to evaluation mode (e.g., dropout is disabled).
- Runs eval_iters batches to compute an average loss for each split.
- Finally, the model is set back to training mode with model.train().

## Attention Mechanism
```python
class Head(nn.Module):
    """ One head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # Scaled dot-product
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```
- **Key Components:**
- **Linear Projections:**
-Three separate linear layers compute the key, query, and value representations from the input.
- **Causal Mask (tril):**
-self.tril is a lower-triangular matrix ensuring that a token can only attend to previous tokens (maintaining autoregressive property).
- **Forward Pass:**
-Computes keys and queries.
-Computes scaled dot-product attention scores.
-Applies the causal mask to prevent “peeking” into the future.
-Uses softmax to obtain attention weights, applies dropout, and finally aggregates values based on these weights

- **MultiHeadPurpose:**
-Runs several attention heads in parallel.
- **How it Works:**
-Instantiates multiple Head objects (each processing a portion of the embedding space).
-Concatenates their outputs.
-Projects back to the original embedding size with a linear layer and applies dropout.

## Feed-Forward Network
```python
class FeedFoward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```
- **Purpose:**
- Implements a two-layer feed-forward network that expands the dimensionality by a factor (here, 4×) and then reduces it back.
- **Non-Linearity and Dropout:**
- ReLU introduces non-linearity.
- Dropout regularizes the network.

## Transformer Block
```python
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
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
```
- **Components:**
-**Self-Attention:**
-Processes the input through multi-head self-attention after applying layer normalization.
- **Feed-Forward:**
-Processes the normalized output through a feed-forward network.
- **Residual Connections:**
-The original input is added back to the output after each sub-layer (self-attention and feed-forward), which helps with gradient flow and stabilizes training.

## The GPT Language Model
```python
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```
- **Components**
- **Token Embeddings:**
-Maps each token index to a vector representation.
- **Position Embeddings:**
-Provides the model with information about token positions (since transformers are permutation-invariant).
- **Transformer Blocks:**
-A stack of n_layer transformer blocks processes the sum of token and positional embeddings.
- **Final Layer Norm and LM Head:**
-The output is normalized and then projected to a vector of size vocab_size to produce logits for each token.

```python
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)        # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
```
- **Steps:**
- **Embedding:**
-The input token indices (idx) are converted into embeddings and added to the corresponding positional embeddings.
- **Transformer Processing:**
-The combined embeddings are fed through the transformer blocks.
- **Projection:**
-After layer normalization, the final linear layer produces logits over the vocabulary.
- **Loss Computation:**
-If target labels are provided, the model reshapes the logits and computes cross-entropy loss.

```python
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # Only the last token's logits
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```
- **Generation Loop:**
-For each new token to be generated:
- **Context Cropping:**
-Only the last block_size tokens are used as context.
- **Prediction:**
-The model produces logits for the next token.
- **Sampling:**
-A probability distribution is computed via softmax and a token is sampled.
- **Appending:**
-The sampled token is concatenated to the existing sequence.

```python
model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

- **Model Initialization:**
-The GPT model is created, moved to the appropriate device, and its total parameter count is printed.
- **Optimizer:**
-AdamW is used for training.
- **Training Iterations:**
-Every eval_interval iterations, the loss on both train and validation splits is evaluated.
- **For each iteration:**
-A batch is sampled.
-The forward pass computes the logits and loss.
-Gradients are computed (loss.backward()) and the optimizer updates the weights.

```python
def interactive_generate():
    """
    Interactive text generation using the same token logic as original code
    """
    while True:
        try:
            user_input = input("Enter max tokens to generate (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
                
            max_tokens = int(user_input)
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated_text = decode(m.generate(context, max_new_tokens=max_tokens)[0].tolist())
            
            print("\nGenerated text:")
            print("-" * 50)
            print(generated_text)
            print("-" * 50 + "\n")
            
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    interactive_generate()
```

- **Interactive Loop:**
- Prompts the user for the number of tokens to generate.
- Uses a zero token as the initial context.
- Calls the model’s generate function to produce new tokens.
- Decodes the generated token indices back into text and displays it.
- Provides error handling for invalid input and graceful exit on interruption.
