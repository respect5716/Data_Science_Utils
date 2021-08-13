# util functions for pytorch_lightning
import pytorch_lightning as pl


class Module(pl.LightningModule):
    """Base module for pytorch_lightning
    """
    
    @property
    def param_groups(self):
        no_decay = ["bias", "bn", "ln", "Norm"]
        param_groups = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.get('weight_decay', 0.0),
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
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
