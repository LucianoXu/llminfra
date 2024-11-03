
from llminfra import *
from datasets import load_dataset

tok = tiktoken.get_encoding('cl100k_base')
vocab_size = tok.max_token_value + 1

def trainV2():
    train(
    ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train").shuffle(),
    ckpt_folder = './ckpt/wikija/V2',

    # model
    model=Llama3(
        vocab_size = vocab_size,
        context_length = 512,
        num_layers = 10,
        dim = 768,
        num_heads = 16,
        d_ff = 3072,
        device='cuda'
    ),

    context_length = 512,

    # optimizer
    lr_min = 1e-4, 
    lr_max = 3e-4,
    T_c = 50000,
    weight_decay = 0.05, 
    betas = (0.9, 0.99), 
    eps = 1e-8,
    
    # training setting:
    load_version_name = 'latest',
    batch_size = 9,
    accumulation_step = 8,
    save_interval = 50000,
    max_grad_l2norm = 1.0,
    # proc_token_limit=327_680_000,
    proc_token_limit=None
    )


if __name__ == "__main__":
    trainV2()
