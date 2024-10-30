
from llminfra import *

def trainV3():
    train(
    enc_input_path = 'tinystories_train_encoded.npy',
    ckpt_folder = './tinystories/Basic',

    # model
    vocab_size = 10000,
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
    attn_pdrop = None, 
    residual_pdrop = None,

    # optimizer
    lr_min = 2e-5, 
    lr_max = 5e-4,
    T_w = 5000,
    T_c = 75000,
    weight_decay = 0.1, 
    betas = (0.9, 0.99), 
    eps = 1e-8,
    
    # training setting:
    load_ckpt = None,
    valid_enc_input = 'tinystories_valid_encoded.npy',
    valid_interval = 1000,
    batch_size = 16,
    save_interval = 100000,
    max_grad_l2norm = None,
    proc_token_limit=327_680_000,
    device = 'mps'
    )


if __name__ == "__main__":
    trainV3()