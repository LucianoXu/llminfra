
from typing import Iterable, Optional, Callable, Type
import torch
from torch.optim.optimizer import Optimizer


import math

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    
def grad_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    norms : list[float] = []

    for p in parameters:
        if p.grad is None:
            continue
        
        norms.append(torch.norm(p.grad.data, p=2).item())


    l2_norm = torch.norm(torch.tensor(norms))

    if l2_norm > max_l2_norm:
        coef = max_l2_norm / (l2_norm + 1e-6)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data *= coef

    
import numpy as np
import random
def np_loader(x: np.ndarray, batch_size: int, context_length: int, device: str):
    r = [random.randint(0, x.size - context_length - 1) for _ in range(batch_size)]
    inputs = np.stack([x[id:id+context_length] for id in r])
    targets = np.stack([x[id+1:id+context_length+1] for id in r])

    return torch.tensor(inputs, dtype=torch.long, device = device), torch.tensor(targets, dtype=torch.long, device = device)
    


def save_checkpoint(model: torch.nn.Module, 
                    optimizer: Optimizer, 
                    iteration: int, 
                    out):
    
    obj = {
        'model_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(obj, out)

def load_checkpoint(src, 
                    model: torch.nn.Module, 
                    optimizer: Optimizer):
    obj = torch.load(src)

    model.load_state_dict(obj['model_dict'])
    optimizer.load_state_dict(obj['optimizer_state'])
    return obj['iteration']



from .model import *
import os
import pathlib
from torch.utils.tensorboard.writer import SummaryWriter
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from torch.optim.adamw import AdamW

def train(
        ds: Dataset,
        ckpt_folder: str,
        tokenizer_path: str,

        # model
        vocab_size: int,
        context_length: int,
        num_layers: int,
        dim: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float|None, 
        residual_pdrop: float|None,

        # optimizer
        lr_min: float, 
        lr_max: float,
        T_c: int,
        weight_decay, 
        betas: tuple[float, float], 
        eps = 1e-8,
        
        # training setting:
        load_ckpt: Optional[str] = None,
        valid_enc_input: Optional[str] = None,
        valid_interval: int = 1000,
        batch_size: int = 8,
        save_interval: int = 10000,
        max_grad_l2norm: Optional[float] = None,
        proc_token_limit: Optional[int] = None,
        device = 'cpu',

        # other setting
        model_type: Type[torch.nn.Module] = TransformerLM
        ):
    
    ckpt_folder_path = pathlib.Path(ckpt_folder)
    
    model = model_type(
        vocab_size,
        context_length,
        num_layers,
        dim,
        num_heads,
        d_ff,
        attn_pdrop, 
        residual_pdrop
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr = lr_max, betas = betas, weight_decay=weight_decay,
        eps = eps
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = T_c, T_mult = 1, eta_min = lr_min) # type: ignore

    writer = SummaryWriter(ckpt_folder_path, flush_secs=5, max_queue=1)

    if load_ckpt is not None:
        t = load_checkpoint(load_ckpt, model, optimizer)
    else:
        t = 1


    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<|endoftext|>"))
    tokenizer.enable_truncation(max_length=context_length+1)

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    def load_examples(next_batch):
        enc = torch.tensor([enc.ids for enc in tokenizer.encode_batch_fast(next_batch['text'])], dtype=torch.long, device=device)
        inputs, targets = enc[:, :-1], enc[:, 1:]
        return inputs, targets


    if valid_enc_input is not None:
        valid_data = np.load(valid_enc_input)
        valid_loss = None

    criterion = torch.nn.CrossEntropyLoss()

    try:
        while True:
            optimizer.zero_grad()
            
            # set the learning rate
            scheduler.step(t)

            # get the learning rate
            lr = optimizer.param_groups[0]['lr']

            inputs, targets = load_examples(next(iter(dataloader)))

            logits = model(inputs)

            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()

            if max_grad_l2norm:
                grad_clip(model.parameters(), max_grad_l2norm)

            optimizer.step()


            print(f"{ckpt_folder}\tStep {t}\ttokens: {t * context_length * batch_size:,}\tlr: {lr:.7f}\tloss: {loss.item():.3f}")

            log_loss = {
                'train': loss.item()
            }

            if t % save_interval == 0:
                print("Saving checkpoint...")
                save_checkpoint(model, optimizer, t, ckpt_folder_path / f"{t}.pth")

            if valid_enc_input is not None and t % valid_interval == 0:
                print("computing validation loss...")
                model.eval()

                valid_loss = 0.
                rounds = 256
                with torch.no_grad():
                    for _ in tqdm(range(rounds), desc='Validating'):
                        inputs, targets = np_loader(valid_data, batch_size, context_length, device)

                        logits = model(inputs)

                        valid_loss += cross_entropy_loss(logits, targets).item() # type: ignore
                avg_loss = valid_loss / rounds

                print('Validation Loss: ', avg_loss)

                log_loss['validation'] = avg_loss
                model.train()

            writer.add_scalars('loss(step)', log_loss, t)
            writer.add_scalars('loss(token)', log_loss, t * context_length * batch_size)
            writer.flush()

            if proc_token_limit is not None and t * context_length * batch_size > proc_token_limit:
                raise KeyboardInterrupt()

            t += 1
    
    except KeyboardInterrupt:
        print("Saving checkpoint...")
        save_checkpoint(model, optimizer, t, ckpt_folder_path / f"{t}.pth")
        
    
    


