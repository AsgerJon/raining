"""
The 'erf' function computes the Gauss error function. For details, see:

References
----------
Abramowitz, M. & Stegun, I. A. (1964).
*Handbook of Mathematical Functions*, Section 7.1.
National Bureau of Standards.

NIST Digital Library of Mathematical Functions.
https://dlmf.nist.gov/7.2
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING

from . import pi, exp

if TYPE_CHECKING:  # pragma: no cover
  from typing import Any, TypeAlias, Type, Union, Self


def _erf(x: float) -> float:
  term = 1  # must be > 1e-16 to start loop, but otherwise is overridden.
  result = x  # x is the n=0 term
  n = 1
  den = 1
  num = x

  while abs(term) > 1e-16:
    den *= (n / (2 * n - 1) * (2 * n + 1))
    num *= -x * x
    term = num / den
    result += term
    n += 1

  return 2 * result * pi ** (-0.5)


class _ErfTerm:
  __slots__ = ('n', 'T', 'x', 'x_2')

  def __init__(self, x: float, ) -> None:
    if abs(x) < 1e-08:
      raise ZeroDivisionError
    self.x, self.x_2 = x, 1 / x / x

  def __iter__(self, ) -> Self:
    self.n = 0
    self.T = 1
    return self

  def __next__(self, ) -> float:
    prevT = float(self.T)
    self.T *= -(2 * self.n + 1) / 2 * self.x_2
    if abs(self.T) < 1e-16:
      raise StopIteration
    if abs(self.T) > abs(prevT):
      self.T = 0  # Triggers StopIteration next
    self.n += 1
    return prevT


def _erfc(x: float) -> float:
  """Asymptotic expansion"""
  if isinstance(x, int):
    return _erfc(float(x))
  try:
    out = exp(-x * x) * pi ** (-0.5) / x * sum((*_ErfTerm(x),))
  except ZeroDivisionError:
    out = (1 - _erf(x))
  if isinstance(x, float):
    return out.real
  if isinstance(x, complex):
    if abs(x.imag) < 1e-16:
      return out.real
  return out


def erf(x: float) -> float:
  """Compute the Gauss error function.

  Parameters
  ----------
  x : float
    The input value for which to compute the error function.

  Returns
  -------
  float
    The value of the error function at the given input.

  References
  ----------
  Abramowitz, M. & Stegun, I. A. (1964).
  *Handbook of Mathematical Functions*, Section 7.1.
  National Bureau of Standards.

  NIST Digital Library of Mathematical Functions.
  https://dlmf.nist.gov/7.2
  """
  if x < 0:
    return -_erf(-x)
  if x < 3.5:
    return _erf(x)
  return 1 - _erfc(x)
