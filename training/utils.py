import torch
from torchtyping import TensorType

from typing import Union
from pathlib import Path


def to_tensor(*args) -> TensorType["args"]:
    return torch.Tensor([x for x in args])


def make_parents(path: Union[Path, str]) -> Path:
    """
    Just tries to resovle a path if the directories don't exist yet :)
    """
    if not path.parent.exists():
        try:
            path.parent.mkdir(parents=True)
        except Exception as e:
            print(f"Unable to resolve path for making a figure\n{e}")

    return path
