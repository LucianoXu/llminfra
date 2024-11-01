
from llminfra import *
from datasets import load_dataset

def trainV1():
    train(
    ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train"),
    ckpt_folder = './ckpt/wikija/V1',

    # model
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
    attn_pdrop = None, 
    residual_pdrop = None,

    # optimizer
    lr_min = 2e-4, 
    lr_max = 4e-3,
    T_c = 37500,
    weight_decay = 0.1, 
    betas = (0.9, 0.99), 
    eps = 1e-8,
    
    # training setting:
    ckpt_name = 'none',
    batch_size = 32,
    save_interval = 100000,
    max_grad_l2norm = 1.0,
    proc_token_limit=327_680_000,
    device = 'mps'
    )


if __name__ == "__main__":
    trainV1()