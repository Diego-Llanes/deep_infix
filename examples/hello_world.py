from deep_infix.deep_types import DeepInt, DeepFloat

val_1, val_2 = DeepInt(2, verbose=True), DeepInt(3, verbose=True)
print(f"{(val_1 == val_2)=}")

print("\n")

e, pi = DeepFloat(2.71828), DeepFloat(3.14159)
print(f"{(e == pi)=}")

print("\n")

print(f"{(e != pi)=}")
