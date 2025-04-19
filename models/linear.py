import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


class Linear(Module):
    def __init__(
        self,
        insize: int = 2,
        outsize: int = 1,
        classification: bool = True,
    ) -> None:
        super().__init__()
        self.classification = classification
        self.linear = nn.Linear(
            in_features=insize,
            out_features=outsize,
            bias=True,
        )

    @typechecked
    def forward(
        self,
        x: TensorType["batch", "insize"],
    ):
        x = self.linear(x)
        if self.classification:
            x = F.sigmoid(x)

        return x
