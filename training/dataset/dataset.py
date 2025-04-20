import torch
from torch.utils.data import Dataset
from torchtyping import TensorType, patch_typeguard

from typing import Tuple, Union, Optional, List, Any
import random

from ..utils import to_tensor



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


def list_encoding_func(selection: Any, options: List[Any]) -> float:
    # Try to kee the value centered around zero
    return float(options.index(selection)) - (0.5 * len(options))


class InfixEquivalanceDataset(Dataset):
    """
    This was going to be all infix, but I am still thinking about how to handle
    things like XOR, AND, etc.
    """

    def __init__(
        self,
        operator: Optional[str] = None,
        encode_operator: bool = True,
        equal_p: float = 0.2,
        _len: int = 1_000,  # 1_000 /feels right/
        _norm_func: callable = scale_pair,
        _min_num: int = torch.iinfo(torch.int32).min,
        _max_num: int = torch.iinfo(torch.int32).max,
    ) -> None:
        self.infix_equivalances = [
            "<",
            ">",
            "==",
            "!=",
            "<=",
            ">=",
        ]
        if (not operator is None) and (not operator in self.infix_equivalances):
            raise ValueError(
                f"Please use one of the following infix equivalence operators: \"{'\", \"'.join(self.infix_equivalances)}\"."
            )
        self.operator = operator
        self.norm_func = _norm_func
        self._len = _len

        self._min_num = _min_num
        self._max_num = _max_num
        self.equal_p = equal_p

        self.encode_operator = encode_operator

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


        operator = self.operator if not self.operator is None else random.choice(self.infix_equivalances)

        if self.encode_operator:
            # Append an encoding that represents the operator 
            x = to_tensor(
                left,
                right,
                list_encoding_func(operator, self.infix_equivalances),
            )
        else:
            x = to_tensor(left, right)

        # The label is just applying the infix operator
        y = to_tensor(float(eval(f"{left}{operator}{right}")))

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
