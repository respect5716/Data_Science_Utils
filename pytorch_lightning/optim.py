from typing import Dict

import torch
from transformers import get_scheduler

OPTIM_DICT = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW
}


def help_optimizer(optim_name: str = None):
    if optim_name in OPTIM_DICT:
        help(OPTIM_DICT[optim_name])
    else:
        print(f'choose optimizer from {list(OPTIM_DICT.keys())}')
        
        
def get_optimizer(optim_name: str, params, **kwargs):
    optimizer = OPTIM_DICT.get(optim_name.lower())
    if optimizer is None:
        raise ValueError
        
    optimizer = optimizer(params, **kwargs)
    return optimizer
