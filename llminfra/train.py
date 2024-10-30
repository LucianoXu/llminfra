
from typing import Iterable, Optional, Callable, Type
import torch
from torch.optim.optimizer import Optimizer

def cross_entropy_loss(logits: torch.FloatTensor, targets: torch.LongTensor) -> torch.Tensor:
    max_logit = logits.max(dim=-1, keepdim=True).values
    sub_logits = logits - max_logit

    loss_matrix = torch.gather(sub_logits, -1, targets.unsqueeze(-1)).squeeze(-1) - torch.log(torch.exp(sub_logits).sum(-1))

    return -torch.mean(loss_matrix)




class AdamW(Optimizer):
    def __init__(self, params, lr: float, weight_decay, betas: tuple[float, float], eps = 1e-8):
        defaults = {'lr': lr, 'beta1': betas[0], 'beta2': betas[1], 'lam': weight_decay}
        super().__init__(params, defaults)
        self.epsilon = eps

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            lam = group['lam']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.data

                if "m" not in state:
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['t'] = 1
                
                m, v, t = state['m'], state['v'], state['t']
                
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                lr_t = lr * (1-beta2**t)**0.5 / (1 - beta1**t)

                p.data -= lr_t * m / (torch.sqrt(v) + self.epsilon)

                p.data -= lr * lam * p.data

                state['t'] = t + 1
                

        return loss


import math
def cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return t/T_w * alpha_max
    
    elif t <= T_c:
        return alpha_min + 0.5 * (1. + math.cos((t-T_w)/(T_c - T_w) * math.pi)) * (alpha_max - alpha_min)
    
    else:
        return alpha_min
    
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

def train(
        enc_input_path: str,
        ckpt_folder: str,

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
        T_w: int, 
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
        lr_min, weight_decay, betas, eps
    )

    writer = SummaryWriter(ckpt_folder_path, flush_secs=5, max_queue=1)

    if load_ckpt is not None:
        t = load_checkpoint(load_ckpt, model, optimizer)
    else:
        t = 1

    data = np.load(enc_input_path, mmap_mode='r')

    if valid_enc_input is not None:
        valid_data = np.load(valid_enc_input)
        valid_loss = None

    try:
        while True:
            optimizer.zero_grad()
            
            # set the learning rate
            lr = cosine_schedule(t, 
                alpha_max = lr_max,
                alpha_min = lr_min, 
                T_w = T_w, 
                T_c = T_c)
            
            for p in optimizer.param_groups:
                p['lr'] = lr

            inputs, targets = np_loader(data, batch_size, context_length, device)

            logits = model(inputs)

            loss = cross_entropy_loss(logits, targets)  # type: ignore
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
        
    
    


