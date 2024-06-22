import os

ext_storage_dir = '/storage/hdd1/model/SKK_CHKPT'

ext_data_dir = '/storage/hdd1/data/tiny_shakespeare'

out_dir = 'ts_transformer_VANILLA'
eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 10 
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves (False)
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'tsGPT-char'
# wandb_run_name = 'ts_transformer'

dataset = 'tiny_shakespeare'

num_samples = 2 # number of samples to generate
start = '\n' # starting set of characters for generation
max_new_tokens = 500 # max number of tokens to generate

temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
top_k = 200  # retain only the top_k most likely tokens, clamp others to 0 probability

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
out_dir = f'{out_dir}_D{d_model}'
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