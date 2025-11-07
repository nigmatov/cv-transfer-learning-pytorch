import argparse, os, pytorch_lightning as pl
from src.datamodule import make_loaders
from src.model_cls import ClsModel

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/sample")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()

train_loader, val_loader = make_loaders(args.data, args.batch)
model = ClsModel(num_classes=2)
trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto", devices="auto", log_every_n_steps=5)
trainer.fit(model, train_loader, val_loader)
os.makedirs("outputs", exist_ok=True)
trainer.save_checkpoint("outputs/cls.ckpt")
print("Saved to outputs/cls.ckpt")
