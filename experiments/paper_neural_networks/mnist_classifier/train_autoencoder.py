import os
import shutil
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from config import conf as config
from model import MLPClassifier, Autoencoder
from data import MNISTDataModule
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from nn_utils import build_decoder_from_classifier, build_encoder_from_classifier

def main():
    # Set up the data module
    conf = config['classifier']
    ae_conf = config['autoencoder']
    data_module = MNISTDataModule(conf, os.getcwd())
    
    # Load the classifier from checkpoint (built using MLPClassifier)
    classifier = MLPClassifier.load_from_checkpoint(ae_conf['load_classifier_path'], conf=conf)
    
    # Build the encoder by extracting all layers from the classifier
    new_encoder = build_encoder_from_classifier(classifier)
    new_encoder.requires_grad_(False)  # Freeze the encoder during autoencoder training

    # Build the decoder as the mirror of the classifier's encoder.
    new_decoder = build_decoder_from_classifier(classifier)
    
    # Create an autoencoder using its configuration, and override its encoder/decoder.
    autoencoder = Autoencoder(conf=ae_conf)
    autoencoder.encoder = new_encoder
    autoencoder.decoder = new_decoder

    # Set up logging, checkpointing, and early stopping.
    logger = TensorBoardLogger(ae_conf['save_path'], name=ae_conf['logger']['version'])
    checkpoint_callback = ModelCheckpoint(
        monitor="val_MeanSquaredError",
        save_top_k=1,
        dirpath=ae_conf['save_path'],
        filename="mnist_ae_{epoch:03d}-{val_loss:.4f}",
        save_last=False,
        mode='min'
    )
    early_stop_cb = EarlyStopping(monitor='val_MeanSquaredError', patience=ae_conf['patience'], mode='min')
    
    if not os.path.exists(ae_conf['save_path']):
        os.makedirs(ae_conf['save_path'])
    shutil.copy('config.py', os.path.join(ae_conf['save_path'], 'autoencoder_config.py'))
    
    trainer = pl.Trainer(
        devices=ae_conf['trainer']['devices'],
        max_epochs=ae_conf['trainer']['max_epochs'],
        accelerator=ae_conf['trainer']['accelerator'],
        strategy=ae_conf['trainer']['strategy'],
        logger=logger,
        callbacks=[TQDMProgressBar(refresh_rate=ae_conf['trainer']['refresh_rate']), checkpoint_callback, early_stop_cb]
    )
    
    trainer.fit(autoencoder, data_module)
    trainer.test(ckpt_path='best', datamodule=data_module)

if __name__ == "__main__":
    main()
