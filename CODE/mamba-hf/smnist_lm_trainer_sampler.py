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
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Resize
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models.lm import LM, MambaCustomConfig

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
model_type = 'mamba'  # 'mamba'
n_layer = 6
d_model = 288
expand_size = 2
state_size = 16

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

# Define separator token
separator_token = 2**n_bits # Assuming pixel intensities are in range 0-2^n_bits-1
# 0-2^n_bits-1 are reserved as tokens for pixel intensities and 2**n_bits is reserved for special separator token

# load mnist samples for validation
train_val_split_seed = 1337  # Seed for train/val split

# check if data is already downloaded
if not os.path.exists(os.path.join(ext_data_dir, 'MNIST')):
    set_download = True
else:
    set_download = False

# define the transforms: convert to tensor and resize
transform = Compose([
    ToTensor(),
    Resize(image_size)
])

train_dataset = MNIST(ext_data_dir, train=True, transform=transform, download=set_download)
val_dataset = MNIST(ext_data_dir, train=False, transform=transform, download=set_download)

# filter out all the samples with sample_digit
train_dataset = list(filter(lambda x: x[1] == sample_digit, train_dataset))
val_dataset = list(filter(lambda x: x[1] == sample_digit, val_dataset))

# select `num_samples` random indices from (0, len(val_dataset))
test_indices = random.sample(range(len(val_dataset)), num_samples)

# Store original images for plotting
original_images = [val_dataset[i][0].numpy().squeeze() for i in test_indices]

# -----------------------------------------------------------------------------
# Generate Permutations
# -----------------------------------------------------------------------------

def get_inv_perm(perm):
    return np.argsort(perm)

'''
# RANDOM
# Generate random permutation and inverse permutation
def get_permutation(seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(image_size * image_size)
    inv_perm = np.argsort(perm)
    return perm, inv_perm

# random permutation
perm, inv_perm = get_permutation(permutation_seed)
'''


# IDENTITY
# Generate identity permutation
perm = np.arange(image_size*image_size)
inv_perm = get_inv_perm(perm)


'''
# CONTINUOUS
# Generate permutation for is_continous=True
def get_continuous_permutation(image_size):
    perm_continuous = []
    for i in range(image_size):
        for j in range(image_size):
            # Even rows: normal order
            if i % 2 == 0:
                index = i * image_size + j
            # Odd rows: reversed order
            else:
                index = i * image_size + (image_size - 1 - j)
            perm_continuous.append(index)
    return np.array(perm_continuous)

perm = get_continuous_permutation(image_size)
inv_perm = get_inv_perm(perm)
'''

'''
# HILBERT
# Generate permutation for hilbert curve
from hilbert import decode

def hilbert_flatten(array):
    """
    Flattens a multi-dimensional array using the Hilbert curve.

    Parameters:
    array (ndarray): The input multi-dimensional array.

    Returns:
    ndarray: The flattened array.

    """
    D = array.ndim
    S = np.arange(np.array(array.shape).prod())
    L = decode(S, D, 8).T.tolist()

    return array[tuple(L)]

# Store range(image_size*image_size) in a 2D array
A = np.array(range(image_size*image_size)).reshape(image_size, image_size)
# flatten this according to hilbert curve
A_hilbert_flattened = hilbert_flatten(A)

perm = []
for i in range(image_size*image_size):
    # find the index of i in A_hilbert_flattened
    idx = np.where(A_hilbert_flattened == i)
    perm.append(idx[0][0])
perm = np.array(perm)
inv_perm = get_inv_perm(perm)
'''

# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------

# Preprocess data
def preprocess_data(dataset, seed=train_val_split_seed):
    all_data = []

    for image, _ in tqdm(dataset):
        image = image.numpy()
        image = image.squeeze()

        # quantise the image
        image = (image * (2**n_bits - 1)).astype(np.uint16)

        if is_continous:
            # Flatten 2D image to 1D
            flattened_image = np.concatenate([image[i] if i%2 == 0 else np.flip(image[i], axis=0) for i in range(image.shape[0])])
        else:
            flattened_image = image.flatten()
        
        # Apply permutation
        flattened_image = flattened_image[perm] 
        
        # Append separator token
        if use_separator:
            flattened_image = np.append(separator_token, flattened_image)

        all_data.append(flattened_image)
    
    # return (n, image_size*image_size) array
    return np.array(all_data)

train_data = preprocess_data(train_dataset)
val_data = preprocess_data(val_dataset)

# vocabulary including null token
if use_separator:
    meta = {
        'vocab_size': 2**n_bits + 1,  # Include separator token in vocab
        'itos': {i: i for i in range(2**n_bits + 1)},  # Identity mapping for pixel intensities and separator
        'stoi': {i: i for i in range(2**n_bits + 1)},
    }
else:
    meta = {
        'vocab_size': 2**n_bits,  # Exclude separator token from vocab
        'itos': {i: i for i in range(2**n_bits)},  # Identity mapping for pixel intensities and null token
        'stoi': {i: i for i in range(2**n_bits)},
    }

# set the tokenizers
meta_vocab_size = meta['vocab_size']
print(f"Found vocab_size = {meta_vocab_size}")
stoi, itos = meta['stoi'], meta['itos']
encode = lambda x: [stoi[tok] for tok in x]
decode = lambda x: [itos[tok] for tok in x]


# set the sampling parameters
max_new_tokens = image_size*image_size - n_ctx_pixels # number of tokens to be generated in each sample

# define test_images from val_data using test_indices
test_images = val_data[test_indices]

# store the test context to plot later and the context_xs
test_plots = []
context_xs = []

for test_image in test_images:
    if use_separator:
        context = test_image[:n_ctx_pixels + 1]
    else:
        context = test_image[:n_ctx_pixels]

    # store this to plot the context part of the image later
    if use_separator:
        test_image = test_image[1:]
    
    context_ids = encode(context)
    context_x = (torch.tensor(context_ids, dtype=torch.long, device=device)[None, ...])

    test_plots.append(test_image)
    context_xs.append(context_x)

# batch the context_xs
context_xs = torch.cat(context_xs, dim=0)

# set save directory to save sample images
fig_save_dir = os.path.join(out_dir, 'samples')
# check if the directory exists: if it does, empty the directory, if not create it
if os.path.exists(fig_save_dir):
    print(f"Emptying directory {fig_save_dir}")
    for f in os.listdir(fig_save_dir):
        os.remove(os.path.join(fig_save_dir, f))
else:
    os.makedirs(fig_save_dir)

# set save directory to save model checkpoints
ckpt_save_dir = os.path.join(out_dir, 'ckpt')
# check if the directory exists: if it does, empty the directory, if not create it
if os.path.exists(ckpt_save_dir):
    print(f"Emptying directory {ckpt_save_dir}")
    for f in os.listdir(ckpt_save_dir):
        os.remove(os.path.join(ckpt_save_dir, f))
else:
    os.makedirs(ckpt_save_dir)

# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------


# poor man's data loader
def get_batch(split, seed=0):
    # set the seed
    torch.manual_seed(seed)

    data = train_data if split == 'train' else val_data
    ix = torch.randint(data.shape[0], (batch_size,))

    # reset the seed
    torch.manual_seed(seed+1)
    
    # select a random index between 1 and data.shape[1]
    if use_separator:
        jx = torch.randint(1, data.shape[1]-1, (1,)).item()
    
    X, Y = [], []

    for idx in ix:
        xi = data[idx,:][0:jx]
        yi = data[idx,:][1:jx+1] # shift by 1
        
        # check size of xi, if more than block_size select the last block_size tokens
        if len(xi) >= block_size:
            xi = xi[-block_size:]
            yi = yi[-block_size:]
        
        X.append(xi)
        Y.append(yi)

    X = torch.tensor(np.array(X), dtype=torch.int64)
    Y = torch.tensor(np.array(Y), dtype=torch.int64)

    if device_type == 'cuda':
        X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)
    else:
        X, Y = X.to(device), Y.to(device)
    
    return X, Y

# -----------------------------------------------------------------------------
# Initialize model and optimizer
# -----------------------------------------------------------------------------

iter_num = 0
best_val_loss = 1e9

# Model initialization
if model_type == 'mamba':
    model_config = MambaCustomConfig(d_model=d_model, n_layers=n_layer, expand_size=expand_size, state_size=state_size, vocab_size=meta_vocab_size)
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

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

# Helpers
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == dtype))

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
                logits = model(X).logits
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
        # generate samples using context_xs as input
        model.eval()
        with torch.no_grad():
            with ctx:
                generated_idx = model.generate(
                    idx=context_xs,
                    block_size=block_size,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    spl_token=separator_token
                )
        
        # decode the generated indices
        generated_images = []
        for i in range(num_samples):
            generated_tokens = decode(generated_idx[i].tolist())
            generated_image = np.array(generated_tokens)
            
            if use_separator:
                generated_image = generated_image[1:]
            
            # apply inverse permutation
            generated_image = generated_image[inv_perm]

            # reshape the image
            generated_image = generated_image.reshape(image_size, image_size)

            # normalize the image
            generated_image = generated_image / (2**n_bits - 1)
            generated_images.append(generated_image)

        # plot and save
        fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4*num_samples))

        for i, ax in enumerate(axes):
            # Plot original image
            ax[0].imshow(original_images[i], cmap='gray')
            ax[0].set_title('Original Image')
            ax[0].axis('off')

            # Plot generated image
            ax[1].imshow(generated_images[i], cmap='gray')
            ax[1].set_title('Generated Image')
            ax[1].axis('off')

        # save the generated image to the checkpoint directory
        plt.savefig(os.path.join(fig_save_dir, f'sample_{iter_num}.png'))

        plt.close()

        model.train()


    if iter_num == 0 and eval_only:
        break

    # Forward Backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits= model(X).logits
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