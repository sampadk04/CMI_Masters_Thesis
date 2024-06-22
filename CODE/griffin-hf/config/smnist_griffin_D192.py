import os

ext_storage_dir = '/storage/hdd1/model/smnist_chkpt'

out_dir = 'smnist_griffin'
eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 10
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves (False)
always_save_checkpoint = False

wandb_log =  False # override via command line if you like
wandb_project = 'smnistGPT-char-test'
# wandb_run_name = 'smnist-mamba-cuda'

dataset = 'smnist'
image_size = 16 # resize from (28, 28) to (image_size, image_size)

# sampling parameters
sample_digit = 0 # sample this digit
# n_ctx_pixels = image_size*image_size//2  # number of pixels in context
n_ctx_pixels = 0

num_samples = 4 # number of samples to generate

temperature = 1  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
top_k = 3  # retain only the top_k most likely tokens, clamp others to 0 probability

n_bits = 3
use_separator = True # if True, use a special separator token to separate images in the dataset
is_continous = False # if True, the 2D MNIST will be flattened to 1D contiguously, i.e. 28x28 image will be flattened to 784 image by alternate reversed row-wise concatenation, i.e. 1st row, with reversed 2nd row, with 3rd row, with reversed 4th row, and so on. # if False, the 2D MNIST will be flattened to 1D by row-wise concatenation, i.e. 1st row, with 2nd row, with 3rd row, and so on.
data_dir = os.path.join('data', dataset)

gradient_accumulation_steps = 1

batch_size = 512
block_size = 256 # context of up to block_size previous characters
# also try with 128, 192, 256, 384

# baby GPT model :)
model_type = 'griffin'
n_layer = 6
d_model = 192
n_heads = 8
local_attention_window = image_size
dropout = 0.0

dtype = 'float32'

# rename out_dir and run_name to reflect the model type
out_dir = f'{out_dir}_I{image_size}_B{n_bits}_N{n_layer}_D{d_model}_L{block_size}'
wandb_run_name = out_dir

# set the outdir to be within the external storage directory
out_dir = os.path.join(ext_storage_dir, out_dir)

# set learning rates
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 1000
# iters: 2k, 3k, 4k, 6k
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is smol
warmup_iters = min(100, max_iters//10) # not super necessary potentially

# set it to False if torch._dynamo error persists
compile = False