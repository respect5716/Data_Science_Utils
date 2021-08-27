from typing import List, Callable

import pytorch_lightning as pl


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
