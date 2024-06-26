# lm_trainer.py

"""
Training script for LM models, compatible with Transformer.

Supports both single-GPU debug mode and distributed data parallel (DDP) training.

Usage:

- Single GPU (debug mode):
  $ python lm_trainer.py --batch_size=32 --compile=False

- DDP training (4 GPUs on 1 node):
  $ torchrun --standalone --nproc_per_node=4 lm_trainer.py

- DDP training (4 GPUs across 2 nodes):
  - Master node (IP: 123.456.123.456):
    $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 lm_trainer.py
  - Worker node:
    $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 lm_trainer.py
"""

# add root folder to path
import sys
sys.path.append('..')

import os
import time
import random
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models.lm import LM
from models.transformer.transformer import TransformerConfig

import matplotlib.pyplot as plt
from tqdm import tqdm


## DISABLE COMPILATION WARNINGS
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


# to train with specific GPU, set visible gpu here
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# -----------------------------------------------------------------------------
# Configuration (defaults for babyGPT on tiny_shakespeare)
# -----------------------------------------------------------------------------

# I/O
out_dir = 'out_tiny_shakespeare_transformer' # this is where the checkpoints will be saved
eval_interval = 100
log_interval = 20
eval_iters = 10
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume'

# Wandb logging
wandb_log = False
wandb_project = 'babyGPT-char'
wandb_run_name = 'tiny-shakespeare-transformer'

# Data
dataset = 'tiny-shakespeare'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context length (also necessary for generation)

# Model
model_type = 'transformer'  # 'transformer' or 'mamba' or 's4
n_layer = 6
d_model = 288
bias = False

# transformer specific
n_head = 6
dropout = 0.2
use_flash_attention = True

# mamba specific
expand_factor = 2

# mamba and s4 specific
d_state = 16 # state dimension, set higher (x4 for S4)
use_cuda = True
# mamba: choose True if you can (mamba-ssm installed). else, fallbacks to mamba.py (https://github.com/alxndrTL/mamba.py)
# s4: using S4D block from s4d.py (https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py)
dt_min= 1e-3
dt_max= 1e-1

# Optimizer
learning_rate = 1e-3
max_iters = 2000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 100
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually

# DDP settings
backend = 'nccl'

# System
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# smnist sampling parameters
dataset = 'smnist'
n_bits = 8
image_size = 16 # resize from (28, 28) to (image_size, image_size)

# Seed for permutation
permutation_seed = 1234

use_separator = True # if True, use a special separator token to separate images in the dataset
is_continous = False # if True, the 2D MNIST will be flattened to 1D contiguously, i.e. 28x28 image will be flattened to 784 image by alternate reversed row-wise concatenation, i.e. 1st row, with reversed 2nd row, with 3rd row, with reversed 4th row, and so on. # if False, the 2D MNIST will be flattened to 1D by row-wise concatenation, i.e. 1st row, with 2nd row, with 3rd row, and so on.

sample_digit = 0 # sample this digit
n_ctx_pixels = image_size*image_size//2 # number of pixels in context

# set no. of samples to generate
num_samples = 4
start = '\n' # starting set of characters for generation
max_new_tokens = 500  # max number of tokens to generate

# Generation parameters
temperature = 2  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
top_k = 5  # retain only the top_k most likely tokens, clamp others to 0 probability

# checkpoint storage dir
ext_storage_dir = '/storage/hdd1/model/SKK_CHKPT'


# midi generation parameters
ext_data_dir = '/storage/hdd1/data/midi_dataset/raw_data/maestro-v2-chunked'
# this already contains the corresponding tokenizer to be used for this dataset (tokenizer.json file in the 'tokenizer' dir)

# -----------------------------------------------------------------------------
# Load configuration overrides
# -----------------------------------------------------------------------------

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # For logging

# -----------------------------------------------------------------------------
# Initialize training setup
# -----------------------------------------------------------------------------

# Distributed Data Parallel (DDP) setup
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"Tokens per iteration: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

train_data = np.memmap(os.path.join(ext_data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(ext_data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(ext_data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
meta_vocab_size = meta['vocab_size']
print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
# encoder and decoder for char-level model
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# set save directory to save model checkpoints
ckpt_save_dir = os.path.join(out_dir, 'ckpt')
# check if the directory exists: if it does, empty the directory, if not create it
if os.path.exists(ckpt_save_dir):
    print(f"Emptying directory {ckpt_save_dir}")
    for f in os.listdir(ckpt_save_dir):
        os.remove(os.path.join(ckpt_save_dir, f))
else:
    os.makedirs(ckpt_save_dir)





# get the context for generation
start_ids = encode(start)
context_xs = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]) # shape (1, len(start))
# stack this context to num_samples size
context_idx = context_xs.repeat(num_samples, 1) # shape (num_samples, len(start))
# we use this during generation loop



# set save directory to save the generated samples
text_save_dir = os.path.join(out_dir, 'samples')
# check if the directory exists: if it does, empty the directory, if not create it
if os.path.exists(text_save_dir):
    print(f"Emptying directory {text_save_dir}")
    for f in os.listdir(text_save_dir):
        os.remove(os.path.join(text_save_dir, f))
else:
    os.makedirs(text_save_dir)

# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------


# poor man's data loader
def get_batch(split, seed=0):
    data = train_data if split == 'train' else val_data

    # set the seed
    torch.manual_seed(seed)

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Initialize model and optimizer
# -----------------------------------------------------------------------------

iter_num = 0
best_val_loss = 1e9

# Model initialization
if model_type == 'transformer':
    model_config = TransformerConfig(
        d_model=d_model, n_layers=n_layer, n_heads=n_head, dropout=dropout, bias=bias, flash=use_flash_attention, max_len=block_size
    )
else:
    raise ValueError(f"Unsupported model type: {model_type}")

model = LM(model_config, vocab_size=meta_vocab_size)

if init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    # fix the keys of the state dictionary, sometimes they get unwanted prefixes
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# Print no. of params and model summary
n_params = model.get_num_params()
print(f"Model has {round(n_params / 1e6, 1)} M parameters")

# Move model to device
model.to(device)

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# Compile the model (optional)
if compile:
    print("Compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# Wrap model into DDP container (if applicable)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == dtype))

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

# Helpers

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(seed):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, seed)
            with ctx:
                logits = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff in [0, 1]
    return min_lr + coeff * (learning_rate - min_lr)

# Logging using WandB
# set directory to store wandb runs
wandb_dir = os.path.join(ext_storage_dir, 'wandb')
os.makedirs(wandb_dir, exist_ok=True)
if wandb_log and master_process:
    import wandb
    wandb.init(dir=wandb_dir, project=wandb_project, name=wandb_run_name, config=config)

# Training loop
X, Y = get_batch('train', iter_num)

t0 = time.time()
local_iter_num = 0

# todo: debug this
if not compile:
    raw_model = model.module if ddp else model
else:
    raw_model = model._orig_mod

while True:
    # Set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate loss and save checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss(iter_num)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"Saving checkpoint to {ckpt_save_dir}")
                torch.save(checkpoint, os.path.join(ckpt_save_dir, 'ckpt.pt'))
    
    if iter_num % 25 == 0:
        # generate samples using context_idx as input
        model.eval()
        with torch.no_grad():
            with ctx:
                generated_idx = model.generate(
                    idx=context_idx, block_size=block_size, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, spl_token=None
                )
        model.train()
        
        generated_text = ""
        for i in range(num_samples):
            generated_text += f'Generated Text #{i}\n' + decode(generated_idx[i].tolist()) + '\n\n\n'
        
        # save the generated text to a .txt file
        with open(os.path.join(text_save_dir, f'sample_text_{iter_num}.txt'), 'w') as f:
            f.write(generated_text)

    if iter_num == 0 and eval_only:
        break

    # Forward Backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits= model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train', iter_num + micro_step)
        scaler.scale(loss).backward()

    # Clip gradients
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Update optimizer and scaler
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    # Termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()