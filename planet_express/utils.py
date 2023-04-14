import math
from collections import namedtuple
from typing import Dict

import numpy as np
import torch
import random
import torch.nn as nn
import os

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from torch.utils.data import DistributedSampler
import loguru


DDPState = namedtuple("DDPState", ("is_ddp", "rank", "local_rank", "device", "is_master_process", "world_size"))

class Lion(torch.optim.Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.
        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

def get_optimizer_class(name):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam
    elif name == "adamw":
        return torch.optim.AdamW
    elif name == "sgd":
        return torch.optim.SGD
    elif name == "lion":
        return Lion
    else:
        raise ValueError(f"Optimizer {name} is not supported")

def get_activation(activation):
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "none":
        return nn.Identity()
    else:
        raise ValueError('Activation function not recognized')

def get_dtype(dtype):
    dtype = dtype.lower()
    if dtype == "float32":
        return torch.float32
    elif dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError('Dtype not recognized')

def get_lr_scheduler(args, optimizer: torch.optim.Optimizer):
    if args.lr_scheduler == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)
    elif args.lr_scheduler == "cosine":
        def get_lr(it):
            # 1) linear warmup for warmup_iters steps
            if it < args.warmup_steps:
                return it / args.warmup_steps
            # 2) if it > lr_decay_iters, return min learning rate
            if it > args.decay_iterations:
                return args.min_lr_frac
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - args.warmup_steps) / (args.decay_iterations - args.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            return args.min_lr_frac + coeff * (1 - args.min_lr_frac)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

def setup_logger(log_path: str):
    loguru.logger.add(
        log_path,
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        backtrace=True,
        enqueue=True,
        diagnose=True,
    )
    return loguru.logger

def get_ddp_state() -> DDPState:
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        master_process = True
        device = "cuda"
        world_size = 1
    return DDPState(
        is_ddp=ddp,
        rank=ddp_rank,
        local_rank=ddp_local_rank,
        is_master_process=master_process,
        device=device,
        world_size=world_size,
    )


def get_dataloader_sampler(ddp_state, dataset):
    if ddp_state.is_ddp:
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=ddp_state.world_size,
            rank=ddp_state.rank,
        )
    else:
        sampler = None
    return sampler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def wrap_model(model, ddp_state, device):
    model = model.to(device)
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    if ddp_state.is_ddp:
        model = DDP(
            model,
            device_ids=[ddp_state.local_rank],
            output_device=ddp_state.local_rank,
            find_unused_parameters=True,
        )
    return model

def unwrap_state_dict(state_dict: Dict) -> Dict:
    """Unwrap a state dict that was wrapped by DDP"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict
