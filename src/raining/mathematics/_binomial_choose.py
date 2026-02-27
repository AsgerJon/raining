"""
The 'binomialChoose' function computes the binomial coefficient, which is
the number of ways to choose 'k' elements from a set of 'n' elements without
regard to the order of selection. It is denoted as 'C(n, k)' or 'n choose k'.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING

from . import factorial

if TYPE_CHECKING:  # pragma: no cover
  pass


def _binomialChoose(n: int, k: int) -> int:
  if k < 0 or k > n:
    return 0
  if k == 0 or k == n:
    return 1
  return factorial(n) // (factorial(k) * factorial(n - k))


#  Just use the built-in math.comb
#  Perhaps a future release will implement something as fast, but for now,
#  using the above implementation is too slow.
from math import comb

binomialChoose = comb
