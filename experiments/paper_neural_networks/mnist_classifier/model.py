import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy
from torchmetrics.regression import MeanSquaredError

class MLPClassifier(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.dropout = nn.Dropout(conf['drop_out_p'])
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.LogSoftmax(dim=1)
        
        metrics = MetricCollection([Accuracy(task="multiclass", num_classes=10)])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics   = metrics.clone(prefix='val_')
        self.test_metrics  = metrics.clone(prefix='test_')

        hidden_layers = conf['hidden_layers']
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if conf.get('use_batch_norm', False) else None
        
        input_dim = 28 * 28  # MNIST image size flattened
        for i, hidden_dim in enumerate(hidden_layers):
            self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
            if self.batch_norms is not None:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        self.fc_out = nn.Linear(input_dim, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
                
            # For the first hidden layer, optionally skip activation if specified
            if i == 0 and self.conf.get('no_act_1st_layer', False):
                pass  # no activation on first layer
            else:
                x = F.relu(x)
            x = self.dropout(x)
        out = self.fc_out(x)
        out = self.softmax(out)
        return out

    def predict_with_hidden(self, x):
        x = x.view(x.size(0), -1)
        hidden_outputs = []
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            if i == 0 and self.conf.get('no_act_1st_layer', False):
                pass
            else:
                x = F.relu(x)
            x = self.dropout(x)
            hidden_outputs.append(x)
        out = self.fc_out(x)
        out = self.softmax(out)
        return out, hidden_outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.train_metrics.update(logits, y)
        return {"loss": loss}

    def on_training_epoch_end(self, outputs):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.val_metrics.update(logits, y)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics.update(logits, y)

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self.test_metrics.reset()

    def predict_step(self, batch, batch_idx):
        return self(batch[0])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.conf['lr'])
        return optimizer

    def on_train_start(self):
        if self.conf.get('save_initial_weights', False):
            torch.save(self.state_dict(), self.conf['save_initial_weights_path'])

class Autoencoder(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.input_dim = conf.get("input_dim", 28 * 28)
        self.drop_p = conf.get("drop_out_p", 0.0)
        self.use_bn = conf.get("use_batch_norm", False)
        
        # Build Encoder
        encoder_layers = []
        current_dim = self.input_dim
        for hidden_dim in conf["encoder_layers"]:
            encoder_layers.append(nn.Linear(current_dim, hidden_dim))
            if self.use_bn:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            if self.drop_p > 0.0:
                encoder_layers.append(nn.Dropout(self.drop_p))
            current_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        latent_dim = current_dim  # The last encoder dimension

        # Build Decoder:
        # Use provided 'decoder_layers' or mirror the encoder_layers if not provided.
        if "decoder_layers" in conf:
            decoder_config = conf["decoder_layers"]
        else:
            # Mirror the encoder layers: e.g. if encoder_layers = [200, 100] then decoder_layers = [200]
            decoder_config = list(reversed(conf["encoder_layers"][:-1]))
        
        decoder_layers = []
        current_dim = latent_dim
        for hidden_dim in decoder_config:
            decoder_layers.append(nn.Linear(current_dim, hidden_dim))
            if self.use_bn:
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            if self.drop_p > 0.0:
                decoder_layers.append(nn.Dropout(self.drop_p))
            current_dim = hidden_dim
        # Final layer to reconstruct the input
        decoder_layers.append(nn.Linear(current_dim, self.input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Loss function and metrics
        self.loss_fn = nn.MSELoss()
        metrics = MetricCollection([MeanSquaredError()])
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics   = metrics.clone(prefix="val_")
        self.test_metrics  = metrics.clone(prefix="test_")
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.train_metrics.update(x_hat, x)
        return {"loss": loss}
    
    def on_training_epoch_end(self, outputs):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self.train_metrics.reset()
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.val_metrics.update(x_hat, x)
    
    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self.val_metrics.reset()
    
    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics.update(x_hat, x)
    
    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, sync_dist=True)
        self.test_metrics.reset()
    
    def predict_step(self, batch, batch_idx):
        return self(batch[0])
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.conf['lr'])
        return optimizer