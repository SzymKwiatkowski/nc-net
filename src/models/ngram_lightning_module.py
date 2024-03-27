import torch.linalg
from lightning import pytorch as pl
import torchmetrics
from models.ngram_model import NGramLanguageModeler
import torch.nn as nn


class NGramLightningModule(pl.LightningModule):
    def __init__(self,
                 lr: float,
                 lr_patience: int,
                 lr_factor: float):
        super().__init__()

        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.model = NGramLanguageModeler(
            100,
            2,
            10
        )
        metrics = torchmetrics.MetricCollection([
            torchmetrics.MeanAbsoluteError(),
            torchmetrics.MeanAbsolutePercentageError(),
            torchmetrics.MeanSquaredError(),
        ])

        self.train_metrics = metrics.clone('train_')
        self.val_metrics = metrics.clone('val_')
        self.test_metrics = metrics.clone('test_')
        self.loss_function = nn.NLLLoss()
        self.save_hyperparameters()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        loss = self.loss_function(outputs, labels)

        self.test_metrics.update(outputs, labels)

        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        self.log_dict(self.train_metrics)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)

        self.val_metrics.update(outputs, labels)

        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(self.val_metrics)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)

        self.test_metrics.update(outputs, labels)

        self.log('test_loss', loss, prog_bar=True)
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), betas=(0.91, 0.9999),
                                      lr=self.lr, weight_decay=0.1, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.lr_patience,
                                                               factor=self.lr_factor)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_MeanAbsoluteError',
        }
