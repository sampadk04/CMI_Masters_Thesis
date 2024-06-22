import os

# main project directory

ext_storage_dir = '/storage/hdd1/model/SKK_CHKPT'

ext_data_dir = '/storage/hdd1/data/midi_dataset/raw_data/maestro-v2-chunked'
# this already contains the corresponding tokenizer to be used for this dataset (tokenizer.json file in the 'tokenizer' dir)

out_dir = 'midi_mamba'
eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 25 
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves (False)
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'midiGPT-char'
# wandb_run_name = 'ts_transformer'

dataset = 'midi'

num_samples = 2 # number of samples to generate
max_new_tokens = 512 # max number of tokens to generate

temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
top_k = 20  # retain only the top_k most likely tokens, clamp others to 0 probability

gradient_accumulation_steps = 1

batch_size = 256
block_size = 256 # context of up to block_size previous characters
# also try with 128, 192, 256, 384

# baby GPT model :)
model_type = 'mamba'
n_layer = 8
d_model = 192
expand_size = 2
state_size = 32

dtype = 'float32'

# rename out_dir and run_name to reflect the model type
out_dir = f'{out_dir}_D{d_model}'
wandb_run_name = out_dir

# set the outdir to be within the external storage directory
out_dir = os.path.join(ext_storage_dir, dataset, out_dir)

# set learning rates
learning_rate = 5e-4 # with baby networks can afford to go a bit higher
max_iters = 1000
# iters: 2k, 3k, 4k, 6k
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = min(100, max_iters//10) # not super necessary potentially

compile = False