import torch.linalg
from lightning import pytorch as pl
import torchmetrics
from models.controller_model import ControllerNetworkModel

from models.controller_models import BaseModels


class ControllerModel(pl.LightningModule):
    def __init__(self,
                 lr: float,
                 lr_patience: int,
                 lr_factor: float,
                 extraction_points_count: int,
                 num_dense_neurons: int):
        super().__init__()

        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.loss_function = torch.nn.MSELoss()
        # network = getattr(BaseModels, model)

        input_size = (extraction_points_count + 1) * 8
        self.model = ControllerNetworkModel(
            input_size=input_size,
            output_size=5,
            num_dense_neurons=num_dense_neurons
        )

        metrics = torchmetrics.MetricCollection([
            torchmetrics.MeanAbsoluteError(),
            torchmetrics.MeanAbsolutePercentageError(),
            torchmetrics.MeanSquaredError(),
        ])

        self.train_metrics = metrics.clone('train_')
        self.val_metrics = metrics.clone('val_')
        self.test_metrics = metrics.clone('test_')
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
