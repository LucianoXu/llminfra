
from typing import Iterable, Literal, Optional, Callable, Type
import random
from tqdm import tqdm
    
import numpy as np

import tiktoken

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.optimizer import Optimizer

import elab
from .models import *
from elab import ELab


def np_loader(x: np.ndarray, batch_size: int, context_length: int, device: str):
    r = [random.randint(0, x.size - context_length - 1) for _ in range(batch_size)]
    inputs = np.stack([x[id:id+context_length] for id in r])
    targets = np.stack([x[id+1:id+context_length+1] for id in r])

    return torch.tensor(inputs, dtype=torch.long, device = device), torch.tensor(targets, dtype=torch.long, device = device)
        

def get_batched_inputs_targets(encodings: list[list[int]], context_length: int, PADDING_ID: int):
    '''
    Transform a list of encodings into a batch of inputs and targets for self-supervised learning.
    '''

    # calculate the sequence length of this batch
    encoding_len_max = max([len(encoding) for encoding in encodings])
    batch_seq_len = min(encoding_len_max, context_length)

    input_ids = []
    targets = []
    
    for encoding in encodings:
        encoding_len = len(encoding)
        
        # select a random part in the text
        if encoding_len == 0:
            input_id = [PADDING_ID] * batch_seq_len
            target = [PADDING_ID] * batch_seq_len

        elif encoding_len <= batch_seq_len:
            padding_number = batch_seq_len - (encoding_len - 1)
            input_id = encoding[:encoding_len-1] + [PADDING_ID] * padding_number
            target = encoding[1:encoding_len] + [PADDING_ID] * padding_number

        else:
            id_start = random.randint(0, encoding_len - batch_seq_len - 1)
            input_id = encoding[id_start: id_start + batch_seq_len]
            target = encoding[id_start + 1: id_start + batch_seq_len + 1]

        input_ids.append(input_id)
        targets.append(target)
        
    return input_ids, targets



def train(
        ds: Dataset,
        ckpt_folder: str,

        model: torch.nn.Module,

        context_length: int,

        # optimizer
        lr_min: float, 
        lr_max: float,
        T_c: int,
        weight_decay, 
        betas: tuple[float, float], 
        eps = 1e-8,
        
        # training setting:
        ckpt_name: str|Literal['latest', 'none'] = 'none',
        batch_size: int = 8,
        save_interval: int = 10000,
        max_grad_l2norm: Optional[float] = None,
        proc_token_limit: Optional[int] = None
        ):
    
    # build tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    PADDING_ID=tokenizer.eot_token

    # get device
    device = next(model.parameters()).device

    # build optimizer
    optimizer = AdamW(
        model.parameters(),
        lr = lr_max, betas = betas, weight_decay=weight_decay,
        eps = eps
    )

    # create/load the checkpoint
    lab = ELab(
        ckpt_folder, 
        ckpt_name=ckpt_name,
        model = model,
        optimizer = optimizer,
        default_states={
            't': 1,
            'proc_token': 0
        }
    )

    t: int = lab.states['t']
    proc_token: int = lab.states['proc_token']
    
    # set the learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = T_c, T_mult = 1, eta_min = lr_min) # type: ignore

    # create the tensorboard writer
    writer = SummaryWriter(ckpt_folder, flush_secs=5, max_queue=1)

    # create the dataloader
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)    # type: ignore
    
    # create the loss function
    criterion = torch.nn.CrossEntropyLoss()

    try:
        for batch in tqdm(dataloader):            
            # set the learning rate
            scheduler.step(t)

            # get the learning rate
            lr = optimizer.param_groups[0]['lr']

            encodings = [tokenizer.encode(text, allowed_special='all') for text in batch['text']]

            inputs, targets = get_batched_inputs_targets(encodings, context_length, PADDING_ID)
            inputs = torch.tensor(inputs, dtype=torch.long, device = device)
            targets = torch.tensor(targets, dtype=torch.long, device = device)

            logits = model(inputs)

            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()

            # calculate and log the raw gradient norm
            raw_grad_norm = elab.get_grad_norm(model)

            if max_grad_l2norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_l2norm)

            optimizer.step()
            optimizer.zero_grad()

            proc_token += context_length * batch_size


            print(f"{ckpt_folder}\tStep {t}\ttokens: {proc_token:,}\tlr: {lr:.7f}\tloss: {loss.item():.3f}")

            log_loss = {
                'train': loss.item()
            }

            if t % save_interval == 0:
                lab.states['t'] = t
                lab.states['proc_token'] = proc_token
                lab.save('f"{t}.pth"')

            writer.add_scalars('loss(step)', log_loss, t)
            writer.add_scalars('loss(token)', log_loss, proc_token)
            writer.add_scalar('raw_grad_norm', raw_grad_norm, t)
            writer.flush()

            if proc_token_limit is not None and proc_token > proc_token_limit:
                raise KeyboardInterrupt()

            t += 1
    
    except KeyboardInterrupt:
        lab.states['t'] = t
        lab.states['proc_token'] = proc_token
        lab.save(f"{t}.pth")
        
    
    


