from elab import ELab, generate, generate_batched
from llminfra import TransformerLM
import tiktoken

def gen_test(    
    ckpt_folder = './ckpt/tinystories/V5',
    # model
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    model = TransformerLM(
        vocab_size=tokenizer.max_token_value + 1,
        context_length=context_length,
        num_layers=num_layers,
        dim=dim,
        num_heads=num_heads,
        d_ff=d_ff
    )
    lab = ELab(ckpt_folder, 'latest', model=model)

    prompt = "Once upon a time"
    prompt_ids = tokenizer.encode(prompt)
    res = generate(model, prompt_ids, max_len=256, EOT_id = tokenizer.eot_token, T=0.6, p_threshold=0.95, include_prompt=True, show_progress=True)
    res = tokenizer.decode(res)

    print(res)



def gen_test_batched(    
    ckpt_folder = './ckpt/tinystories/V5',
    # model
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    model = TransformerLM(
        vocab_size=tokenizer.max_token_value + 1,
        context_length=context_length,
        num_layers=num_layers,
        dim=dim,
        num_heads=num_heads,
        d_ff=d_ff
    )
    lab = ELab(ckpt_folder, 'latest', model=model)

    prompt = ["Once upon a time", "In a galaxy far far away", "It was a dark and stormy night", "It was the best of times, it was the worst of times"]
    prompt_ids = [tokenizer.encode(p) for p in prompt]
    res = generate_batched(model, prompt_ids, max_len=256, EOT_id = tokenizer.eot_token, T=0.6, p_threshold=0.95, include_prompt=True, show_progress=True)
    res = [tokenizer.decode(r) for r in res]

    for r in res:
        print(r)
        print()

if __name__ == "__main__":
    # gen_test()
    gen_test_batched()

