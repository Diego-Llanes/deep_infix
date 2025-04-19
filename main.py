import torch
import skeletonkey as sk
from torch.optim import AdamW
from torch.utils.data import DataLoader

from functools import partial
import os
from pathlib import Path

from runner.runner import Runner
from dataset.dataset import InfixEquivalanceDataset
from models.linear import Linear
from utils import make_parents
from viz import plot_losses


@sk.unlock("configs/config.yaml")
def main(cfg):
    model: Linear = sk.instantiate(cfg.model)
    optimizer: AdamW = sk.instantiate(cfg.optimizer, params=model.parameters())

    # Datasets
    train_ds: InfixEquivalanceDataset = sk.instantiate(cfg.train_dataset)
    dev_ds: InfixEquivalanceDataset = sk.instantiate(
        cfg.dev_dataset,
        operator=train_ds.operator,  # make sure the dev ds is the same fn
    )

    # DataLoaders
    train_dl: DataLoader
    dev_dl: DataLoader
    train_dl, dev_dl = map(
        partial(DataLoader, batch_size=cfg.batch_size), (train_ds, dev_ds)
    )

    # Runners
    train_runner: Runner
    dev_runner: Runner
    train_runner, dev_runner = map(
        partial(
            Runner,
            model=model,
            optimizer=optimizer,
            critereon=torch.nn.functional.mse_loss,
        ),
        (train_dl, dev_dl),
    )

    train_losses, dev_losses = [], []
    epochs = list(range(1, cfg.epochs + 1))
    for epoch in epochs:
        print(f"Training Epoch: {epoch}")
        train_losses.append(train_runner.run_epoch())
        print(f"Dev Epoch: {epoch}")
        dev_losses.append(dev_runner.run_epoch(train=False))
        print("-" * os.get_terminal_size()[0])

        plot_losses(
            epochs=epochs[:epoch],
            train_losses=train_losses,
            dev_losses=dev_losses,
            save_path=make_parents(Path(f"{cfg.logdir}/{cfg.run_name}_loss.html")),
            title=f'Learning the "{cfg.train_dataset.operator}" Operator',
        )


if __name__ == "__main__":
    main()
