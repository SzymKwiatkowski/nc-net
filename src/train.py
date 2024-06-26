"""Training module"""
from pathlib import Path
import argparse
import os

import lightning.pytorch as pl
from lightning.pytorch.loggers import NeptuneLogger, TensorBoardLogger

from datamodules.controller import ControllerDataModule
from models.model import ControllerModel
from utils.helpers import load_config


# pylint: disable=W0105
def train(args):
    """
    :param args: parsed arguments
    :rtype: None
    """
    # Overrides used graphic card
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_dir = str(args.data_dir)
    config = load_config(args.config)
    token = config['config']['NEPTUNE_API_TOKEN']
    project = config['config']['NEPTUNE_PROJECT']
    """
    Expected model types are:
    - RBF
    - skip_connection
    - DenseRBF
    - None, which is default bare fully connected network
    """
    model_type = None

    if args.use_neptune:
        logger = NeptuneLogger(
            project=project,
            api_token=token)
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    pl.seed_everything(42, workers=True)

    datamodule = ControllerDataModule(
        data_path=Path(data_dir),
        batch_size=64,
        num_workers=4,
        model_type=model_type,
        extraction_points_count=args.extraction_points_count,
    )

    model = ControllerModel(
        module_config={
            "lr": 1.1e-3,
            "lr_patience": 3,
            "lr_factor": 0.5,
            "extraction_points_count": args.extraction_points_count,
            "loss": "MSE"
        },
        network_config={
            'input_size': datamodule.n_features,
            'output_size': datamodule.n_targets,
            'num_dense_neurons': args.dense_neurons,
            'type': model_type,
        }
    )

    model.hparams.update(datamodule.hparams)

    model_summary_callback = pl.callbacks.ModelSummary(max_depth=-1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=f'points-{args.extraction_points_count}'+'-{epoch}-{val_MeanAbsoluteError:.5f}',
        mode='min',
        monitor='val_MeanAbsoluteError',
        verbose=True,
        save_last=True)
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_MeanAbsoluteError',
        mode='min',
        patience=args.patience
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[model_summary_callback, checkpoint_callback, early_stop_callback, lr_monitor],
        accelerator='cuda',
        max_epochs=args.epochs,
        limit_train_batches=690,
    )

    trainer.fit(model=model, train_dataloaders=datamodule.train_dataloader(),
                val_dataloaders=datamodule.val_dataloader())

    results = trainer.test(model=model, ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)

    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='train',
        description='train neural controller network',
        epilog='')
    # Points to root project dir
    parser.add_argument('-c', '--config', action='store', default='../config.yaml')
    # Points to root project dir
    parser.add_argument('-d', '--data_dir', action='store', default='../data')
    # Using neptune
    parser.add_argument('-n', '--use_neptune', action='store', default=False)
    # Max training epochs
    parser.add_argument('-e', '--epochs', action='store', default=70,
                        type=int, help='Specified number of maximum epochs')
    parser.add_argument('-ep', '--extraction-points-count', action='store', default=6,
                        type=int, help='Specified count of points from trajectory to be used')
    parser.add_argument('-dn', '--dense-neurons', action='store', default=512, type=int)
    parser.add_argument('-p', '--patience', action='store', default=10,
                        type=int, help='Specified count of points from trajectory')

    args_parsed = parser.parse_args()

    train(args_parsed)
