import os

ext_storage_dir = '/storage/hdd1/model/SKK_CHKPT'

ext_data_dir = '/storage/hdd1/data/MNIST_raw'

out_dir = 'smnist_transformer_VANILLA'
eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 10 
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves (False)
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'smnistGPT-char'
# wandb_run_name = 'smnist_transformer'

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

gradient_accumulation_steps = 1

batch_size = 512
block_size = 256 # context of up to block_size previous characters
# also try with 128, 192, 256, 384

# baby GPT model :)
model_type = 'transformer'
n_layer = 4
n_head = 8
d_model = 192
dropout = 0.1
use_flash_attention = True

# rename out_dir and run_name to reflect the model type
out_dir = f'{out_dir}_L{block_size}_IDENTITY'
wandb_run_name = out_dir

# set the outdir to be within the external storage directory
out_dir = os.path.join(ext_storage_dir, dataset, out_dir)

# set learning rates
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 1000
# iters: 2k, 3k, 4k, 6k
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = min(100, max_iters//10) # not super necessary potentially

compile = True