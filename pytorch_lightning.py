# util functions for pytorch_lightning
import pytorch_lightning as pl


class Module(pl.LightningModule):
    """Base module for pytorch_lightning
    """
    
    @property
    def param_groups(self):
        no_decay = ["bias", "bn", "ln", "norm"]
        param_groups = [
            {
                # apply weight decay
                "params": [p for n, p in self.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
                "weight_decay": self.hparams.get('weight_decay', 0.0),
            },
            {
                # not apply weight decay
                "params": [p for n, p in self.named_parameters() if any(nd in n.lower() for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return param_groups
    
    @property
    def num_training_steps(self):
        assert 'batch_size' in self.hparams, 'batch size is required in hparams'
        assert 'epoch_size' in self.hparams, 'epoch size is required in hparams'
        
        effective_batch_size = self.hparams.batch_size * self.trainer.accumulate_grad_batches * self.trainer.num_gpus
        num_training_steps = int(len(self.train_dataloader.dataloader.dataset) / effective_batch_size * self.hparams.epoch_size)
        return num_training_steps
    
    @property
    def num_warmup_steps(self):
        warmup_ratio = self.hparams.get('warmup_ratio', 0.)
        num_warmup_steps = int(self.num_training_steps * warmup_ratio)
        return num_warmup_steps
    
    def step(self, batch):
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
        curr_metric = trainer.callback_metrics[self.metric]
        if self.should_save(curr_metric):
            print(f'STEP: {trainer.global_step:06d} | {self.metric}: {curr_metric:.3f} | model saved ({self.best_metric:.3f} -> {curr_metric:.3f})')
            self.save_checkpoint(model)
            self.best_metric = curr_metric
        else:
            print(f'STEP: {trainer.global_step:06d} | {self.metric}: {curr_metric} | model not saved (best metric: {self.best_metric:.3f})')
