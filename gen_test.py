from typing import Type
from elab import ELab, generate, generate_batched
from llminfra import *
import tiktoken
from torch import nn

tok = tiktoken.get_encoding('cl100k_base')
vocab_size = tok.max_token_value + 1

def gen_test(    
    model: nn.Module,
    ckpt_folder: str,
    version_name: str,
    prompt = "Once upon a time",
    context_length = 256,
    T = 0.6,
    p_threshold = 0.95,
    repitition_penalty = None
):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    lab = ELab(ckpt_folder, version_name, model=model)

    prompt_ids = tokenizer.encode(prompt)
    res = generate(model, prompt_ids, 
        max_len=context_length, 
        EOT_id = tokenizer.eot_token, 
        T=T, p_threshold=p_threshold, repitition_penalty=repitition_penalty,
        include_prompt=True, show_progress=True)
    res = tokenizer.decode(res)

    print(res)


def gen_test_batched(        
    model: nn.Module,
    ckpt_folder: str,
    version_name: str,
    prompts = ["Once upon a time", "In a galaxy far far away", "It was a dark and stormy night", "It was the best of times, it was the worst of times"],
    context_length = 256,
    T = 0.6,
    p_threshold = 0.95,
):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    lab = ELab(ckpt_folder, version_name, model=model)

    prompt_ids = [tokenizer.encode(p) for p in prompts]
    res = generate_batched(model, prompt_ids, max_len=context_length, EOT_id = tokenizer.eot_token, T=T, p_threshold=p_threshold, include_prompt=True, show_progress=True)
    res = [tokenizer.decode(r) for r in res]

    for r in res:
        print(r)
        print()

if __name__ == "__main__":
    args = Llama3ModelArgs()
    args.vocab_size = vocab_size
    gen_test(
        model = Llama3(
            model_args=args,
            device='cuda'
        ),
        ckpt_folder = './ckpt/wikija/V2',
        version_name = 'latest',
        context_length=512,
        prompt="埼玉県南東部の市。県庁所在地。平成13年（2001）浦和、大宮、与野の3市が合併して成立。平成15年（2003）指定都市。平成17年（2005）に岩槻市を編入。人口122.3万（2010）。",
        # prompt="「神の子どもたちはみな踊る」は、日本の大人気作家村上春樹が書いた小説。",
        # prompt="日本における桜の開花の進行を示す概念で、各地の気象条件により異なる開花時期を元に作成される。通常、南の九州地方から始まり、北海道に至るまで各地で次第に開花する。多くのメディアや観光サイトで桜前線の情報が提供され、花見の予定を立てる指標ともなる。",
        # prompt="日本の東京都にあった城で、江戸時代の徳川将軍家の本拠地として知られる。江戸時代には政治と文化の中心地として栄え、多くの武士や職人が集まった。現在は皇居として一部が公開されており、庭園などの観光地としても人気を博している。",
        # prompt="アイドルマスターは2005から始まったアイドル育成ゲーム。",

        T=0.6,
        p_threshold=0.95,
        repitition_penalty=1.0
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

