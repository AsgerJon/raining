"""
The 'exp' function computes exponential function, that is, the inverse of
the natural logarithm.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING

from ...utilities import maybe
from ...waitaminute import TypeException
from . import cos, sin
from ...waitaminute.mathematics import ConvergenceError

if TYPE_CHECKING:  # pragma: no cover
  from typing import Union, TypeAlias

  Real: TypeAlias = Union[int, float]
  Number: TypeAlias = Union[int, float, complex]


def _shiftDown(x: Real, n: int = None) -> tuple[Number, int]:
  """
  This function shifts *positive* real numbers near 0.
  """
  if x < 0:
    raise ValueError("""Received negative 'x' argument: '%s'.""" % x)
  n = maybe(n, 0)  # null coalescence, n = n ?? 0
  if x > 1e-04:
    return _shiftDown(x / 2, n + 1)
  return x, n


def _exp(x: Real) -> Real:
  if x < 0:
    return 1 / _exp(-x)
  xP, n = _shiftDown(x)
  term = 1.
  out = 1.
  m = 1
  while abs(term) > 1e-16 or m < 4:
    term *= xP / m
    out += term
    m += 1
    if m > 100:
      raise ConvergenceError(100)
  return out ** (2 ** n)


def exp(x: Number) -> Number:
  if isinstance(x, (int, float)):
    return _exp(float(x))
  if isinstance(x, complex):
    return exp(x.real) * (cos(x.imag) + 1j * sin(x.imag))
  raise TypeException('x', x, int, float, complex)


e = exp(1)
