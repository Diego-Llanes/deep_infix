import torch
import torch.nn as nn

from typing import Optional, Union, List
from pathlib import Path
from collections import OrderedDict
import os

from deep_infix.training.dataset.dataset import (
    InfixEquivalanceDataset,
    list_encoding_func,
    scale_pair,
)
from deep_infix.training.utils import to_tensor


class DeepInfix:
    def __init__(
        self,
        weights_path: Optional[Union[Path, str]],
        model: nn.Module,
        equivililance_operators: List[str] = [
            "<",
            ">",
            "==",
            "!=",
            "<=",
            ">=",
        ],
        encoding_func: callable = list_encoding_func,
        norm_func: callable = scale_pair,
        verbose=False,
    ) -> None:
        assert weights_path.exists(), "Make sure the weights are downloaded"

        weights: OrderedDict = torch.load(weights_path)
        model.load_state_dict(weights)

        self.model = model
        self.equivililance_operators = equivililance_operators
        self.encoding_func = encoding_func
        self.norm_func = norm_func
        self.verbose = verbose

    def _verbose_message(
        self,
        lhs: Union[int, float],
        rhs: Union[int, float],
        operator: str,
        correct_answer: bool,
    ) -> None:
        msgs = [
            f'Asking a neural network if "{lhs} {operator} {rhs}"',
            f"We think it was {correct_answer}!",
        ]

        max_len = max(len(m) for m in msgs)
        box_w = max_len + 4  # 2 spaces padding + 2 border chars

        print("+" + "-" * (box_w - 2) + "+")
        for m in msgs:
            print("| " + m.ljust(max_len) + " |")
        print("+" + "-" * (box_w - 2) + "+")

    def __call__(
        self,
        lhs: Union[int, float],
        rhs: Union[int, float],
        operator: str,
    ) -> bool:
        _lhs, _rhs, _ = scale_pair(lhs, rhs)
        enc_operator = list_encoding_func(operator, self.equivililance_operators)
        x = to_tensor(_lhs, _rhs, enc_operator).unsqueeze(0)

        correct_answer = bool(round(self.model(x).item()))

        if self.verbose:
            self._verbose_message(lhs, rhs, operator, correct_answer)
        return correct_answer
