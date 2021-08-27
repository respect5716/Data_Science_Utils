from typing import Any
import pytorch_lightning as pl


DEFUALT_ARGS = {
    'batch_size': 32,
    
    'trainer': {
        'accelerator': 'ddp',
        'gpus': -1,
        'max_steps': 1000, # or use max_epochs (max_epochs is prioritized than max_steps)
        'log_every_n_steps': 10,
        'val_check_interval': 1000,
    },
    
    'optimizer': {
        'name': 'adamw',
        'params': {
            'lr': 5e-5,
            'eps': 1e-8,
            'weight_decay': 1e-5,
        }
    },
    
    'scheduler': {
        'name': 'linear',
        'warmup_size': 0,
    }
    
}


class AttributeDict(pl.utilities.parsing.AttributeDict):
    def __init__(self, data):
        for k, v in data.items():
            self.__setattr__(k, v)
    
    def __setattr__(self, key: str, val: Any) -> None:
        if isinstance(val, dict):
            val = AttributeDict(val)
        self[key] = val
        
