# util functions for pytorch_lightning
import os
import pytorch_lightning as pl
from typing import Optional, Any, Dict, List, Callable
from .optim import get_optimizer, get_scheduler

    
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
    
 
