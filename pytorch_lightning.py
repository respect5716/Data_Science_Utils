# util functions for pytorch_lightning
import os
import pytorch_lightning as pl
from typing import Optional, Any, Dict, List, Callable
from transformers import get_scheduler


class AttributeDict(pl.utilities.parsing.AttributeDict):
    def __init__(self, data):
        for k, v in data.items():
            self.__setattr__(k, v)
    
    def __setattr__(self, key: str, val: Any) -> None:
        if isinstance(val, dict):
            val = AttributeDict(val)
        self[key] = val
        

OPTIM_DICT = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW
}


def help_optimizer(optim_name = None):
    if optim_name in OPTIM_DICT:
        help(OPTIM_DICT[optim_name])
    else:
        print(f'choose optimizer from {list(OPTIM_DICT.keys())}')
        
        
def get_optimizer(optim_name, params, **kwargs):
    optimizer = OPTIM_DICT.get(optim_name.lower())
    if optimizer is None:
        raise ValueError
        
    optimizer = optimizer(params, **kwargs)
    return optimizer
    
    
class Module(pl.LightningModule):
    """Base module for pytorch_lightning
    """
    
    def check_hyperparameters(self, hparams):
        assert 'trainer' in hparams
        assert 'optimizer' in hparams
        
        
    def save_hyperparameters(self, hparams):
        self.check_hyperparameters(hparams)
        for k, v in hparams.items():
            self.hparams[k] = v
    
    def param_groups(self):
        no_decay = ["bias", "bn", "ln", "norm"]
        param_groups = [
            {
                # apply weight decay
                "params": [p for n, p in self.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
                "weight_decay": self.hparams.optimizer.params.get('weight_decay', 0.0),
            },
            {
                # not apply weight decay
                "params": [p for n, p in self.named_parameters() if any(nd in n.lower() for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return param_groups
    
    @property
    def num_train_batches(self):
        num_batches = len(self.train_dataloader())

        if type(self.trainer.limit_train_batches) == float:
            num_batches = int(num_batches * self.trainer.limit_train_batches)
        else:
            num_batches = min(self.trainer.limit_train_batches, num_batches)
        
        return num_batches
    
    @property
    def num_devices(self):
        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)
        return num_devices
    
    @property
    def num_training_steps(self):
        if self.trainer.max_steps:
            return self.trainer.max_steps
        
        num_devices = 1 if self.trainer.distributed_backend == 'dp' else self.num_devices
        effective_accum = self.trainer.accumulate_grad_batches * num_devices     
        num_training_steps = self.num_train_batches // effective_accum * self.trainer.max_epochs
        return num_training_steps
    
    @property
    def num_warmup_steps(self):
        warmup_size = self.hparams.scheduler.get('warmup_size', 0)
        num_warmup_steps = int(self.num_training_steps * warmup_size) if type(warmup_size) == float else warmup_size
        return num_warmup_steps
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.hparams.optimizer.name, self.param_groups(), **self.hparams.optimizer.params)
        
        if 'scheduler' in self.hparams:
            scheduler = get_scheduler(self.hparams.scheduler.name, optimizer, self.num_warmup_steps, self.num_training_steps)
            return [optimizer], [scheduler]
        
        return optimizer
    
    def step(self, batch, mode):
        raise NotImplementedError("Implement the step method.")
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')
        
    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'valid')
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')
    
    
    
class CheckpointCallback(pl.Callback):
    def __init__(self, ckpt_dir: str, metric: str = 'valid/loss_epoch', mode: str = 'min'):
        self.ckpt_dir = ckpt_dir
        self.metric = metric
        self.mode = mode
        self.best_metric = 1e10 if mode == 'min' else -1e10 
        
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
    
    def should_save(self, curr_metric):
        if self.mode == 'min':
            return curr_metric < self.best_metric
        else:
            return curr_metric > self.best_metric
    
    def save_checkpoint(self, model):
        raise NotImplementedError("Implement save checkpoint method.")
    
    def on_validation_epoch_end(self, trainer, model):
        curr_metric = float(trainer.callback_metrics[self.metric])
        if self.should_save(curr_metric):
            print(f'STEP: {trainer.global_step:06d} | {self.metric}: {curr_metric:.3f} | model saved ({self.best_metric:.3f} -> {curr_metric:.3f})')
            self.save_checkpoint(model)
            self.best_metric = curr_metric
        else:
            print(f'STEP: {trainer.global_step:06d} | {self.metric}: {curr_metric:.3f} | model not saved (best metric: {self.best_metric:.3f})')

            
            
class MessageCallback(pl.Callback):
    def __init__(self, title: str, message_fn: Callable, metrics: List[str]):
        self.title = title
        self.message_fn = message_fn
        self.metrics = metrics

    def on_train_start(self, trainer, pl_module):
        self.message_fn(f'{self.title} train started!!')

    def on_train_end(self, trainer, pl_module):
        self.message_fn(f'{self.title} train ended!!')

    def on_epoch_end(self, trainer, pl_module):
        message = f'{self.title} log\n'
        message += f'step: {trainer.global_step:06d}\n'
        for k in self.metrics:
            v = trainer.callback_metrics.get(k, 'none')
            v = v if type(v) == str else f'{v:.3f}'
            message += f'{k}: {v}\n'
        
        message += '=' * 50
        self.message_fn(message)
