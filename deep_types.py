from typing import Optional, Union
from pathlib import Path
from collections import OrderedDict

from deep_infix.training.utils import to_tensor
from deep_infix.training.models.mlp import MLP

from ._deep_infix import DeepInfix


"""
weights_path: logs/first_run/test_best_model.pt

model:
  _target_: models.mlp.MLP
  insize: 3 # left, right, operator_encoding
  hidden_sizes: [64]
  outsize: 1
  classification: true
"""

_WEIGHTS_PATH = Path(__file__).parent / "saved_weights/og_weights/test_best_model.pt"
_MODEL = MLP(
    insize=3,
    hidden_sizes=[64],
    outsize=1,
    classification=True,
)
_DEEP_INFIX = DeepInfix(weights_path=_WEIGHTS_PATH, model=_MODEL)


class DeepInt(int):

    def __new__(
        cls,
        value,
        deep_infix: DeepInfix = _DEEP_INFIX,
        verbose=False,
    ):
        obj = super().__new__(cls, value)
        deep_infix.verbose = verbose
        obj._deep_infix = deep_infix
        return obj

    def __eq__(self, other: int) -> bool:
        return self._deep_infix(int(self), int(other), "==")

    def __ne__(self, other: int) -> bool:
        """
        More accurate to use a different stratagey than `not self.__eq__()`,
        trust me bro
        """
        return self._deep_infix(int(self), int(other), "!=")

    def __lt__(self, other: int):
        return self._deep_infix(int(self), int(other), "<")

    def __le__(self, other: int):
        return self._deep_infix(int(self), int(other), "<=")

    def __gt__(self, other: int):
        return self._deep_infix(int(self), int(other), ">")

    def __ge__(self, other: int):
        return self._deep_infix(int(self), int(other), ">=")


class DeepFloat(float):

    def __new__(
        cls,
        value,
        deep_infix: DeepInfix = _DEEP_INFIX,
        verbose=False,
    ):
        obj = super().__new__(cls, value)
        deep_infix.verbose = verbose
        obj._deep_infix = deep_infix
        return obj

    def __eq__(self, other: float) -> bool:
        return self._deep_infix(float(self), float(other), "==")

    def __ne__(self, other: float) -> bool:
        """
        More accurate to use a different stratagey than `not self.__eq__()`,
        trust me bro
        """
        return self._deep_infix(float(self), float(other), "!=")

    def __lt__(self, other: float):
        return self._deep_infix(float(self), float(other), "<")

    def __le__(self, other: float):
        return self._deep_infix(float(self), float(other), "<=")

    def __gt__(self, other: float):
        return self._deep_infix(float(self), float(other), ">")

    def __ge__(self, other: float):
        return self._deep_infix(float(self), float(other), ">=")
