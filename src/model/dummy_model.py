import logging

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Hooks https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks


class Identity(nn.Module):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.identity = nn.Identity()

        # Set all submodules to eval mode
        for module in self.children():
            module.eval()

    def forward(self, x, y=None, shuffle=True):
        return x, y


class DummyModel(L.LightningModule):
    def __init__(self, input_shape, num_classes=2):
        super().__init__()
        self.save_hyperparameters()

        # Layer that just returns the input
        # self.layer = Identity()

        # simple linear autoencoder
        self.model = nn.Sequential(
            nn.Linear(input_shape[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_shape[-1]),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, x)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
