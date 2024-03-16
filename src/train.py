from pathlib import Path
import argparse
import os

import lightning.pytorch as pl
from lightning.pytorch.loggers import NeptuneLogger, TensorBoardLogger

from datamodules.controller import ControllerDataModule
from models.model import ControllerModel
from utils.helpers import load_config


def train(args):
    # Overrides used graphic card
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config_file = args.config
    max_epochs = args.epochs
    use_neptune = args.use_neptune
    data_dir = str(args.data_dir)
    config = load_config(config_file)
    token = config['config']['NEPTUNE_API_TOKEN']
    project = config['config']['NEPTUNE_PROJECT']

    logger = None
    if use_neptune:
        logger = pl.loggers.NeptuneLogger(
            project=project,
            api_token=token)

    else:
        logger = TensorBoardLogger(save_dir="logs/")

    pl.seed_everything(42, workers=True)
    patience = 25

    datamodule = ControllerDataModule(
        data_path=Path(data_dir),
        batch_size=32,
        num_workers=4,
        train_size=0.8
    )

    model = ControllerModel(
        lr=2.55e-5,
        lr_patience=5,
        lr_factor=0.5
    )

    model.hparams.update(datamodule.hparams)

    model_summary_callback = pl.callbacks.ModelSummary(max_depth=-1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='{epoch}-{val_MeanAbsoluteError:.5f}', mode='min',
                                                       monitor='val_MeanAbsoluteError', verbose=True, save_last=True)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_MeanAbsoluteError', mode='min', patience=patience)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[model_summary_callback, checkpoint_callback, early_stop_callback, lr_monitor],
        accelerator='cuda',
        max_epochs=max_epochs
    )

    trainer.fit(model=model, train_dataloaders=datamodule.train_dataloader(),
                val_dataloaders=datamodule.val_dataloader())
    results = trainer.test(model=model, ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)

    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    # Points to root project dir
    parser.add_argument('-c', '--config', action='store', default='../config.yaml')
    # Points to root project dir
    parser.add_argument('-d', '--data_dir', action='store', default='../data/sample_data.csv')
    # Using neptune
    parser.add_argument('-n', '--use_neptune', action='store', default=False)
    # Max training epochs
    parser.add_argument('-e', '--epochs', action='store', default=50,
                        type=int, help='Specified number of maximum epochs')
    args_parsed = parser.parse_args()
    train(args_parsed)
