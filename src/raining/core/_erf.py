"""The 'erf' function represents the error function."""
#  AGPL-3.0 license
#  Copyright (c) 2024 Asger Jon Vistisen
from __future__ import annotations

import sys

from numba import njit

from raining.core import pi

eps = sys.float_info.epsilon


@njit
def erf(x: float) -> float:
  """erf returns the error function at x."""
  out = 0
  term = 0
  den = 1
  for i in range(63):
    if i:
      den *= i
    term = x ** (2 * i + 1) / (2 * i + 1) / den
    if i % 2:
      out -= term
    else:
      out += term
    if term < eps:
      break
    print(x, i, term)
  return out * 2 / pi ** 0.5


@njit
def erfc(x: float) -> float:
  """erfc returns the complementary error function at x."""
  return 1 - erf(x)


@njit
def erfinv(x: float) -> float:
  """erfinv returns the inverse error function at x."""
  if x < -1 or x > 1:
    return float('nan')
  if x < 0:
    return -erfinv(-x)
  out = 0
  term = 0
  for i in range(63):
    term = 2 ** (2 * i + 1) * x ** (2 * i + 1) / (2 * i + 1) / pi
    out += term
    if term < eps * abs(out) and i:
      break
  return out


@njit
def erfcinv(x: float) -> float:
  """erfcinv returns the inverse complementary error function at x."""
  return erfinv(1 - x)
