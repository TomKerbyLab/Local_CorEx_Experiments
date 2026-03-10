import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from config import conf as config
import shutil
from model import MLPClassifier
from data import MNISTDataModule
import torch

def main():
    conf = config['classifier']
    data_module = MNISTDataModule(conf, os.getcwd())
    classifier = MLPClassifier(conf=conf)
    if conf['start_with_init_weights']:
        print("Loading saved intialized weights...")
        classifier.load_state_dict(torch.load(conf['save_initial_weights_path']))
        
    logger = TensorBoardLogger(conf['save_path'], name=conf['logger']['version'])
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        dirpath=conf['save_path'],
        filename="mnist_clf_{epoch:03d}-{val_loss:.4f}",
        save_last=False,
        mode='min'
    )
    early_stop_cb = EarlyStopping(monitor='val_loss', patience=conf['patience'], mode='min')
    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])
    shutil.copy('config.py', conf['save_path']+'/config.py')
    trainer = pl.Trainer(
        devices=conf['trainer']['devices'],
        max_epochs=conf['trainer']['max_epochs'],
        accelerator=conf['trainer']['accelerator'],
        strategy=conf['trainer']['strategy'],
        logger=logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=conf['trainer']['refresh_rate']),
            checkpoint_callback,
            early_stop_cb
        ]
    )
    trainer.fit(classifier, data_module)

    trainer.test(ckpt_path='best', datamodule=data_module)

if __name__ == "__main__":
    main()              