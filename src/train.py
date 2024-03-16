import pickle
from pathlib import Path
import argparse
import yaml
import os

import lightning.pytorch as pl

from datamodules.controller import ControllerDataModule
from models.model import ControllerModel


def load_config(path: Path) -> dict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config_file = args.config
    max_epochs = args.epochs
    config = load_config(config_file)
    token = config['config']['NEPTUNE_API_TOKEN']
    logger = pl.loggers.NeptuneLogger(
        project='szymkwiatkowski/nc-net',
        api_token=token)

    pl.seed_everything(42, workers=True)
    patience = 25

    datamodule = ControllerDataModule(
        data_path=Path('data'),
        batch_size=32,
        num_workers=4,
        train_size=0.8
    )
    model = ControllerModel(
        lr=2.55e-5,
        lr_patience=5,
        lr_factor=0.5,
        model='controller',
    )

    model.hparams.update(datamodule.hparams)

    model_summary_callback = pl.callbacks.ModelSummary(max_depth=-1)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='{epoch}-{val_MeanAbsoluteError:.5f}', mode='min',
                                                       monitor='val_MeanAbsoluteError', verbose=True, save_last=True)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_MeanAbsoluteError', mode='min', patience=patience)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        logger=logger,
        devices=1,
        callbacks=[model_summary_callback, checkpoint_callback, early_stop_callback, lr_monitor],
        accelerator='cuda',
        strategy="ddp",
        max_epochs=max_epochs
    )

    trainer.fit(model=model, datamodule=datamodule)
    predictions = trainer.predict(model=model, ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)

    results = {}
    for prediction in predictions:
        for embedding, identifier in zip(*prediction):
            results[identifier] = embedding.tolist()

    with open('results.pickle', 'wb') as file:
        pickle.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-c', '--config', action='store', default='config.yaml')
    parser.add_argument('-e', '--epochs', action='store', default=50,
                        type=int, help='Specified number of maximum epochs')
    args = parser.parse_args()
    train(args)
