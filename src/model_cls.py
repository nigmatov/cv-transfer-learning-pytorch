import torch, torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet18

class ClsModel(pl.LightningModule):
    def __init__(self, num_classes=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.model = m
        self.crit = nn.CrossEntropyLoss()

    def forward(self, x): return self.model(x)

    def training_step(self, batch, _):
        x,y = batch
        logits = self(x)
        loss = self.crit(logits, y)
        acc = (logits.argmax(1)==y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x,y = batch
        logits = self(x)
        loss = self.crit(logits, y)
        acc = (logits.argmax(1)==y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
