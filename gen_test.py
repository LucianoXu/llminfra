from typing import Type
from elab import ELab, generate, generate_batched
from llminfra import *
import tiktoken
from torch import nn

tok = tiktoken.get_encoding('cl100k_base')
vocab_size = tok.max_token_value + 1

def gen_test(    
    model: nn.Module,
    ckpt_folder,
    prompt = "Once upon a time",
    context_length = 256,
    T = 0.6,
    p_threshold = 0.95,
):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    lab = ELab(ckpt_folder, '50000', model=model)

    prompt_ids = tokenizer.encode(prompt)
    res = generate(model, prompt_ids, max_len=context_length, EOT_id = tokenizer.eot_token, T=T, p_threshold=p_threshold, include_prompt=True, show_progress=True)
    res = tokenizer.decode(res)

    print(res)


def gen_test_batched(        
    model: nn.Module,
    ckpt_folder,
    prompts = ["Once upon a time", "In a galaxy far far away", "It was a dark and stormy night", "It was the best of times, it was the worst of times"],
    context_length = 256,
    T = 0.6,
    p_threshold = 0.95,
):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    lab = ELab(ckpt_folder, 'latest', model=model)

    prompt_ids = [tokenizer.encode(p) for p in prompts]
    res = generate_batched(model, prompt_ids, max_len=context_length, EOT_id = tokenizer.eot_token, T=T, p_threshold=p_threshold, include_prompt=True, show_progress=True)
    res = [tokenizer.decode(r) for r in res]

    for r in res:
        print(r)
        print()

if __name__ == "__main__":
    gen_test(
        model = Llama3(
            vocab_size = vocab_size,
            context_length = 512,
            num_layers = 4,
            dim = 512,
            num_heads = 16,
            d_ff = 2048,
            device='cuda'
        ),
        ckpt_folder='./ckpt/wikija/V1',
        context_length=512,
        prompt="埼玉県南東部の市。県庁所在地。平成13年（2001）浦和、大宮、与野の3市が合併して成立。平成15年（2003）指定都市。平成17年（2005）に岩槻市を編入。人口122.3万（2010）。",

        T=0.4,
        p_threshold=0.95
    )

    # gen_test_batched(
    #     model = Llama3(
    #         vocab_size = vocab_size,
    #         context_length = 256,
    #         num_layers = 4,
    #         dim = 512,
    #         num_heads = 16,
    #         d_ff = 2048,
    #         device='cpu'
    #     ),
    #     ckpt_folder='./ckpt/tinystories/V6'
    # )

