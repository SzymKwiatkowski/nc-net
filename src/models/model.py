"""Lightning module for controller"""
import torch.linalg
from lightning import pytorch as pl
import torchmetrics
from models.controller_model import ControllerNetworkModel
from models.controller_sc_model import ControllerScNetworkModel
from models.rbf.rbf_network import RbfNetwork
from models.rbf.rbf import poisson_two


# pylint: disable=W0221, R0902
class ControllerModel(pl.LightningModule):
    """Class for lightning module for controller"""
    def __init__(self,
                 module_config: dict,
                 network_config: dict):
        super().__init__()

        self.lr = module_config["lr"]
        self.lr_factor = module_config["lr_factor"]
        self.lr_patience = module_config["lr_patience"]
        self.extraction_points_count = module_config["extraction_points_count"]
        self.loss_function = torch.nn.MSELoss()

        if network_config["type"] == "RBF":
            self.model = RbfNetwork(
                [
                    network_config["input_size"],
                    network_config["input_size"],
                    network_config["input_size"],
                    1
                ],
                [
                    network_config["num_dense_neurons"],
                    network_config["num_dense_neurons"],
                    network_config["num_dense_neurons"] // 2]
                ,
                poisson_two
            )
        elif network_config["type"] == "skip_connection":
            self.model = ControllerScNetworkModel(
                input_size=network_config["input_size"],
                output_size=network_config["output_size"],
                num_dense_neurons=network_config["num_dense_neurons"]
            )
        else:
            self.model = ControllerNetworkModel(
                input_size=network_config["input_size"],
                output_size=network_config["output_size"],
                num_dense_neurons=network_config["num_dense_neurons"]
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

    def training_step(self, batch, _batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        loss = self.loss_function(outputs, labels)

        self.test_metrics.update(outputs, labels)

        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        self.log_dict(self.train_metrics)
        return loss

    def validation_step(self, batch, _batch_idx) -> None:
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)

        self.val_metrics.update(outputs, labels)

        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(self.val_metrics)

    def test_step(self, batch, _batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)

        self.test_metrics.update(outputs, labels)

        self.log('test_loss', loss, prog_bar=True)
        self.log_dict(self.test_metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), betas=(0.91, 0.997), eps=1e-7,
                                      lr=self.lr, weight_decay=8e-3, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.lr_patience,
                                                               factor=self.lr_factor)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_MeanAbsoluteError',
        }
