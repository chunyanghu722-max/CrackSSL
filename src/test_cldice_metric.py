import numpy as np
from baseline_utils import _cldice_one

# identical straight line -> clDice should be 1
gt = np.zeros((64, 64), dtype=bool)
gt[32, 10:54] = True
pred = gt.copy()
s1 = _cldice_one(pred, gt)

# empty vs non-empty -> 0
empty = np.zeros_like(gt)
s2 = _cldice_one(empty, gt)

# both empty -> 1
s3 = _cldice_one(empty, empty)

print("identical line clDice:", s1)
print("empty vs crack clDice:", s2)
print("both empty clDice:", s3)

assert abs(s1 - 1.0) < 1e-8
assert abs(s2 - 0.0) < 1e-8
assert abs(s3 - 1.0) < 1e-8
print("clDice metric self-test: OK")
