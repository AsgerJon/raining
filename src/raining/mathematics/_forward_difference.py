"""
The 'forwardDifference' function computes the forward difference in an
infinite convergent series.

It requires a function mapping from natural numbers to real numbers. This
function specifies factors for the terms in the series.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING

from worktoy.examples.mathematics import binomialChoose
from worktoy.waitaminute.mathematics import DomainError

if TYPE_CHECKING:  # pragma: no cover
  from typing import Callable

  Func = Callable[[int], complex]
  Delta = Callable[[int, int], complex]


def forwardDifference(func: Func) -> Delta:
  coefficientCache: dict[int, tuple[int, ...]] = {}

  def getCoefficients(m: int) -> tuple[int, ...]:
    """Return tuple of binomial coefficients (m choose j), j=0..m."""
    try:
      return coefficientCache[m]
    except KeyError:
      pass

    coefficients = [0] * (m + 1)
    c = 1
    for j in range(m + 1):
      coefficients[j] = c
      if j != m:
        c = (c * (m - j)) // (j + 1)
    out = (*coefficients,)
    coefficientCache[m] = out
    return out

  def delta(m: int, n: int, ) -> complex:
    if m < 0:
      raise DomainError(m, 0, float('inf'))
    coefficients = getCoefficients(m)
    out = 0
    for j, c in enumerate(coefficients):
      term = c * func(n + j)
      out += (-term if (j & 1) else term)
    return out

  return delta
