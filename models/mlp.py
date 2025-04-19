import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import List


patch_typeguard()  # use before @typechecked


class MLP(Module):
    def __init__(
        self,
        insize: int = 2,
        hidden_sizes: List[int] = [64, 64],
        outsize: int = 1,
        classification: bool = True,
        activation_fn: callable = F.relu,
    ) -> None:
        super().__init__()
        assert (
            len(hidden_sizes) != 0
        ), f"Please don't use an MLP if you want a linear function...\n{len(hidden_sizes)=}"
        self.classification = classification
        self.activation_fn = activation_fn

        self.in_layer = nn.Linear(insize, hidden_sizes[0])
        if len(hidden_sizes) > 1:
            self.hidden_layers = nn.Sequential(
                *[
                    nn.Linear(hs_1, hs_2)
                    for hs_1, hs_2 in zip(hidden_sizes[:-1], hidden_sizes[1:])
                ]
            )
        else:
            self.hidden_layers = []

        self.out_layer = nn.Linear(hidden_sizes[-1], outsize)

    @typechecked
    def forward(
        self,
        x: TensorType["batch", "insize"],
    ):
        x = self.activation_fn(self.in_layer(x))

        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))

        x = self.out_layer(x)

        if self.classification:
            x = F.sigmoid(x)

        return x
