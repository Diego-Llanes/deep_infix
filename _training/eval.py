import torch
import torch.nn as nn
from torchtyping import TensorType
import skeletonkey as sk

from collections import OrderedDict
from typing import Any, Optional, List, Type
import os

from dataset.dataset import InfixEquivalanceDataset, list_encoding_func, scale_pair
from utils import to_tensor


def get_input(
    prompt: str,
    expected_type: Type,
    options: Optional[List[Any]] = None,
) -> Any:
    while True:
        raw = input(prompt)
        try:
            val = expected_type(raw)
        except (ValueError, TypeError):
            print(f"Please try again, expected a {expected_type.__name__}.")
            continue

        if options and val not in options:
            print(f"Please try again, your options are: {options}")
            continue

        return val


def test_model(model: nn.Module, operators: List[Any]):
    lhs = get_input("LHS: ", float)
    rhs = get_input("RHS: ", float)
    operator = get_input("operator: ", str, options=operators)

    expr = f'Expression: "{lhs} {operator} {rhs}"'
    print(expr)

    lhs, rhs, _ = scale_pair(lhs, rhs)
    operator = list_encoding_func(operator, operators)

    x = to_tensor(lhs, rhs, operator).unsqueeze(0)
    print("we predict the answer is....")
    print(f"{expr}: {bool(round(model(x).item()))}")
    print("Were we right? be honest")
    width = os.get_terminal_size()[0]
    print("-" * width)


@sk.unlock("configs/eval.yaml")
def main(cfg) -> None:
    model: nn.Module = sk.instantiate(cfg.model)
    weights: OrderedDict = torch.load(cfg.weights_path, weights_only=False)
    ds = InfixEquivalanceDataset()

    model.load_state_dict(weights)
    width = os.get_terminal_size()[0]
    print("-" * width)
    test_model(model, ds.infix_equivalances)
    while input("continue (y/n)") != "n":
        test_model(model, ds.infix_equivalances)


if __name__ == "__main__":
    main()
