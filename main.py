
from llminfra import *
from datasets import load_dataset

tok = tiktoken.get_encoding('cl100k_base')
vocab_size = tok.max_token_value + 1

def trainV6():
    train(
    ds = load_dataset("roneneldan/TinyStories", split='train'),
    ckpt_folder = './ckpt/tinystories/V6',

    # model
    model=Llama3(
        vocab_size = vocab_size,
        context_length = 256,
        num_layers = 4,
        dim = 512,
        num_heads = 16,
        d_ff = 2048,
        device='cpu'
    ),

    context_length = 256,

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
    save_interval = 37500,
    max_grad_l2norm = 1.0,
    # proc_token_limit=327_680_000,
    proc_token_limit=None
    )


if __name__ == "__main__":
    trainV6()