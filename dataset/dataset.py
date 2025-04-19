import torch
from torch.utils.data import Dataset
from torchtyping import TensorType, patch_typeguard

from typing import Tuple, Union
import random



def scale_pair(left: float, right: float, lo: float = -1.0, hi: float = 1.0,) -> Tuple[float, float, callable]:
    """
    Affinely map {left, right} so min→lo and max→hi, preserving their spacing.
    returns:
    - (scaled left, scaled right, inverse_fn)
    """
    mn, mx = min(left, right), max(left, right)

    if mx == mn:
        # both equal → return midpoint
        mid = (lo + hi) / 2
        return mid, mid, lambda y: mn

    scale = (hi - lo) / (mx - mn)
    return (
        (left - mn) * scale + lo,
        (right - mn) * scale + lo,
        lambda val: (val - lo) / scale + mn,
    )


class InfixEquivalanceDataset(Dataset):
    """
    This was going to be all infix, but I am still thinking about how to handle
    things like XOR, AND, etc.
    """

    def __init__(
        self,
        operator: str = "==",
        equal_p: float = 0.2,
        _len: int = 1_000,  # 1_000 /feels right/
        _norm_func: callable = scale_pair,
        _min_num: int = torch.iinfo(torch.int32).min,
        _max_num: int = torch.iinfo(torch.int32).max,
    ) -> None:
        infix_equivalances = [
            "<",
            ">",
            "==",
            "!=",
            "<=",
            ">=",
        ]
        if not operator in infix_equivalances:
            raise ValueError(
                f"Please use one of the following infix equivalence operators: \"{'\", \"'.join(infix_equivalances)}\"."
            )
        self.operator = operator
        self.norm_func = _norm_func
        self._len = _len

        self._min_num = _min_num
        self._max_num = _max_num
        self.equal_p = equal_p

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[TensorType["feature"], TensorType[1]]:
        left = random.randint(self._min_num, self._max_num)
        if random.random() < self.equal_p:
            right = left
        else:
            right = random.randint(self._min_num, self._max_num)

        left, right, inv = self.norm_func(left, right)

        def torch_tensor(*args) -> TensorType["args"]:
            return torch.Tensor([x for x in args])

        x = torch_tensor(left, right)
        y = torch_tensor(float(eval(f"{left}{self.operator}{right}")))

        return x, y


    def __len__(
        self,
    ) -> int:
        return self._len

    def __repr__(self) -> str:
        return f"InfixEquivalanceDataset: (\n\t{',\n\t'.join("%s: %s" % item for item in vars(self).items())}\n)"

if __name__ == "__main__":

    ds = InfixEquivalanceDataset(
        operator="==",
        _len=1_000,
        _norm_func=scale_pair,
        equal_p=0.5
    )

    for i in range(15):
        print(ds[0])

    print(f"{len(ds)=}")
