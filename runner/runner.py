from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module

from typing import Tuple, Optional


class Runner:

    def __init__(
        self,
        dataloader: DataLoader,
        model: Module,
        optimizer: Optimizer,
        critereon: callable,
    ) -> None:
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.critereon = critereon

    def run_epoch(
        self,
        train=True,
        epoch: Optional[int] = None,
    ) -> Tuple[float]:

        desc = "loss: _" if epoch is None else f"Epoch: {epoch}, loss: _"
        self.model.train(train)

        with tqdm(total=len(self.dataloader), desc=desc) as pbar:
            cum_loss = 0.0
            for x, y in self.dataloader:
                self.optimizer.zero_grad()
                y_hat = self.model(x)
                loss = self.critereon(y_hat, y)
                if train:
                    loss.backward()
                    self.optimizer.step()
                cum_loss += loss
                desc = f"loss: {loss:0.2f}" if epoch is None else f"Epoch: {epoch}, loss: {loss:0.2f}"
                pbar.desc = desc
                pbar.update()

            return (cum_loss / len(self.dataloader)).item()

    def __repr__(self) -> str:
        return f"Runner: (\n\t{',\n\t'.join("%s: %s" % item for item in vars(self).items())}\n)"
