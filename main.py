
from llminfra import *
from datasets import load_dataset

tok = tiktoken.get_encoding('cl100k_base')
vocab_size = tok.max_token_value + 1

def trainV1():
    args = Llama3ModelArgs()
    args.vocab_size = vocab_size
    train(
    ds = load_dataset("Skylion007/openwebtext", split="train").shuffle(),
    ckpt_folder = './ckpt/owt/V1',

    # model
    model=Llama3(
        model_args=args,
        device='cuda'
    ),

    context_length = args.context_length,

    # optimizer
    lr_min = 1e-4, 
    lr_max = 3e-4,
    T_c = 70000,
    weight_decay = 0.05, 
    betas = (0.9, 0.99), 
    eps = 1e-8,
    
    # training setting:
    load_version_name = 'latest',
    batch_size = 7,
    accumulation_step = 11,
    save_interval = 70000,
    max_grad_l2norm = 1.0,
    # proc_token_limit=327_680_000,
    proc_token_limit=None
    )


if __name__ == "__main__":
    trainV1()
