import torch
from model import ModelConfig, GPTLanguageModel

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
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

# -------------------
# Get model config
modelconf = ModelConfig()
batch_size = 64 # how many independent sequences will we process in parallel?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# -------------------

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - modelconf.block_size, (batch_size,))
    x = torch.stack([data[i:i+modelconf.block_size] for i in ix])
    y = torch.stack([data[i+1:i+modelconf.block_size+1] for i in ix])
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

model = GPTLanguageModel(modelconf)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---------------------

# Define the path for saving the checkpoint
checkpoint_path = 'ckpt.pt'

# Function to save model and configuration
def save_model_checkpoint(model, model_config, path):
    # Get the model state dictionary
    model_weights = model.state_dict()
    
    # Prepare the model configuration
    model_args = {
        'block_size': model_config.block_size,
        'vocab_size': model_config.vocab_size,
        'n_layer': model_config.n_layer,
        'num_heads': model_config.num_heads,
        'n_embd': model_config.n_embd,
        'dropout': model_config.dropout
    }
    
    # Create a checkpoint dictionary
    checkpoint = {
        'model_args': model_args,
        'model': model_weights
    }
    
    # Save the checkpoint
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

# Save the model checkpoint
save_model_checkpoint(model, modelconf, checkpoint_path)

# ---------------------

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

