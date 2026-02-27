"""
The 'logGamma' function computes the natural logarithm of the gamma
function. Please note, that this nested function is much easier to
implement than the gamma function directly.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from logging import warn
from typing import TYPE_CHECKING

from . import log, BernoulliNumber, pi
from ...waitaminute.mathematics import DomainError

if TYPE_CHECKING:  # pragma: no cover
  from typing import Self

_asymptotic_threshold = 12.0


def _shiftUp(z: complex) -> tuple[complex, complex]:
  m = max([0, int(_asymptotic_threshold + 1 - z.real)])
  r = 0
  for k in range(m):
    r += log(z + k)
  return z + m, r + 0j


def _correction(z: complex) -> complex:
  n = 1
  term = 1 + 0j
  out = 0j
  while abs(term) > 1e-16:
    num = BernoulliNumber[2 * n]
    den = 2 * n * (2 * n - 1) * z ** (2 * n - 1)
    prevTemp = term
    term = num / den
    if n > 4 and abs(term) > abs(prevTemp):
      break
    out += term
    n += 1
    if n > 100:
      raise DomainError(z, 0, float('inf'))
  return out


def logGamma(z: complex) -> complex:
  zP, r = _shiftUp(z)
  s = _correction(zP)
  return (zP - 0.5) * log(zP) - zP + log(2 * pi) / 2 - r + s
