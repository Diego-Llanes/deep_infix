# DeepInfix

This is a deep learning library to learn all the infix operations in python.

Right now we are working on equivalence types i.e. `==`, `<=`, `!=`, etc.

## Getting Started

You can install everything you need for the project by running `uv sync`, once you have all the packages you need, you can run the main entry point with `uv run main.py --flag==value`.

The easiest way to change configuration values is editing the configuration file (`configs/config.yaml`) with your favorite text editor (`nvim`).

## Example

Here is how you would use our package.

```python
from deep_infix.deep_types import DeepInt, DeepFloat

val_1, val_2 = DeepInt(2, verbose=True), DeepInt(3, verbose=True)
print(f"{(val_1 == val_2)=}")

print("\n")

e, pi = DeepFloat(2.71828), DeepFloat(3.14159)
print(f"{(e == pi)=}")

print("\n")

print(f"{(e != pi)=}")
```

The following is the output
```
~/Documents/personal_cs
❯ python test.py
+-------------------------------------+
| Asking a neural network if "2 == 3" |
| We think it was False!              |
+-------------------------------------+
(val_1 == val_2)=False


(e == pi)=False


(e != pi)=True

~/Documents/personal_cs
❯

```
You can finally be *mostly* confident, if two floats are equal, thanks to deep learning.
We are working on releasing a model card to demonstrate the failure cases of our work.
Calling all scientists, we need your help.
