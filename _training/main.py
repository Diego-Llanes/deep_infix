import torch
import skeletonkey as sk
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from functools import partial
import os
from pathlib import Path

from .runner.runner import Runner
from .dataset.dataset import InfixEquivalanceDataset
from .models.linear import Linear
from .utils import make_parents
from .viz import plot_losses
from .color import Fore, Back, Style


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

    scheduler = ReduceLROnPlateau(optimizer, "min")

    train_losses, dev_losses = [], []
    epochs = list(range(1, cfg.epochs + 1))
    best_dev_loss = float("inf")
    last_lr = cfg.optimizer.lr
    bad_epochs = 0

    for epoch in epochs:
        screen_width = os.get_terminal_size()[0]

        print(f"{f'{Fore.green}Training Epoch: {epoch}{Fore.reset}':^{screen_width}}")
        train_losses.append(train_runner.run_epoch())

        print(f"{f'{Fore.yellow}Dev Epoch: {epoch}{Fore.reset}':^{screen_width}}")
        dev_loss = dev_runner.run_epoch(train=False)
        dev_losses.append(dev_loss)

        scheduler.step(dev_loss)
        new_lr = scheduler.get_last_lr()[-1]
        if last_lr != new_lr:
            print(
                f"{f'{Fore.yellow}Learning rate updated: {last_lr} -> {new_lr}{Fore.reset}':^{screen_width}}"
            )
            last_lr = new_lr

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            bad_epochs = 0
            torch.save(
                model.state_dict(),
                make_parents(Path(f"{cfg.logdir}/{cfg.run_name}_best_model.pt")),
            )
            title = (
                f'Learning the "{cfg.train_dataset.operator}" Operator'
                if not cfg.train_dataset.operator is None
                else f"Learning all Python Infix Operators"
            )
            print(f"{f'{Fore.blue}New best dev loss!!🎉{Fore.reset}':^{screen_width}}")
            plot_losses(
                epochs=epochs[:epoch],
                train_losses=train_losses,
                dev_losses=dev_losses,
                save_path=make_parents(Path(f"{cfg.logdir}/{cfg.run_name}_loss.html")),
                title=title,
            )
        else:
            bad_epochs += 1
            print(
                f"{f'{Fore.red}Epochs since improvement: {bad_epochs}/{cfg.patience}{Fore.reset}':^{screen_width}}"
            )

        if bad_epochs >= cfg.patience:
            print(
                f"{f'{Style.bold}{Style.underline}{Back.red}Patience exceeded stopping training!{Back.reset}{Style.reset}{Style.reset}':^{screen_width}}"
            )
            break

        print("-" * screen_width)


if __name__ == "__main__":
    main()
