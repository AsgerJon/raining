"""
The 'log' function computes the natural logarithm of real and complex
numbers.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from logging import warn
from typing import TYPE_CHECKING

from . import forwardDifference, eta, atan2
from ...utilities import maybe
from ...waitaminute import TypeException
from ...waitaminute.mathematics import DomainError, ConvergenceError, \
  SingularityError

if TYPE_CHECKING:  # pragma: no cover
  from typing import Union, TypeAlias

  Real: TypeAlias = Union[int, float]
  Number: TypeAlias = Union[int, float, complex]

log2: float = eta(1.0 + 0j).real
logSqrt2: float = log2 / 2
sqrt2: float = 2 ** 0.5
invSqrt2: float = 1 / sqrt2


def _shiftDown(x: Real, s: int = None) -> tuple[Number, Number]:
  s = maybe(s, 0)  # null coalescence, s = s ?? 0
  if x > sqrt2:
    return _shiftDown(x / sqrt2, s + 1)
  if x < 1:
    return _shiftDown(x * sqrt2, s - 1)
  return x, s


def _log(x: Real) -> float:
  if abs(x) < 1e-16:
    raise SingularityError(x, log)
  if x < 0:
    raise DomainError(x, 0, float('inf'))
  if abs(x - sqrt2) < 1e-16:
    return logSqrt2
  xP, s = _shiftDown(x)
  u = xP - 1
  a = lambda _n: u ** (_n + 1) / (_n + 1)
  delta = forwardDifference(a)
  term = 1.
  m = 0
  out = 0.
  den = 1.
  while abs(term) > 1e-16 or m < 4:
    num = delta(m, 0)
    den *= 2
    term = num / den
    out += term
    m += 1
    if m > 100:
      raise ConvergenceError(100)
  return out + s * logSqrt2


def log(z: Number) -> Number:
  if not isinstance(z, (int, float, complex)):
    raise TypeException('z', z, int, float, complex)
  if abs(z) < 1e-16:
    raise SingularityError(z, log)
  if isinstance(z, (int, float)):
    if z < 0:
      return log(z + 0j)
    return _log(z)
  r = abs(z)
  t = atan2(z.imag, z.real)
  return _log(r) + 1j * t
