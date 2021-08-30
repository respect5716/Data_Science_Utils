from typing import List, Callable

import pytorch_lightning as pl


class CheckpointCallback(pl.Callback):
    def __init__(self, ckpt_path: str, metric: str = 'valid/loss_epoch', mode: str = 'min'):
        self.ckpt_path = ckpt_path
        self.metric = metric
        self.mode = mode
        self.best_metric = 1e10 if mode == 'min' else -1e10 

    def should_save(self, curr_metric):
        if self.mode == 'min':
            return curr_metric < self.best_metric
        else:
            return curr_metric > self.best_metric

    def on_validation_epoch_end(self, trainer, model):
         curr_metric = float(trainer.callback_metrics[self.metric])
        
        if self.mode == 'always':
            trainer.save_checkpoint(self.ckpt_path)
            print(f'STEP: {trainer.global_step:06d} | {self.metric}: {curr_metric:.3f} | model saved')
       
        elif self.should_save(curr_metric):
            print(f'STEP: {trainer.global_step:06d} | {self.metric}: {curr_metric:.3f} | model saved ({self.best_metric:.3f} -> {curr_metric:.3f})')
            trainer.save_checkpoint(self.ckpt_path)
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
