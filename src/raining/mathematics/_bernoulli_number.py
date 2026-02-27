"""
The 'BernoulliNumber' function computes the Bernoulli numbers on the
'__getitem__' allowing syntax like: 'BernoulliNumber[10]' to compute the
10th Bernoulli number.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from . import factorial
from ...waitaminute.mathematics import DomainError


class _MetaBernoulliNumber(type):

  def __getitem__(cls, n: int) -> float:
    if n < 0:
      raise DomainError(n, 0, float('inf'))
    if not n:
      return 1
    if n == 1:
      return -1 / 2
    if n % 2:
      return 0
    num = factorial(n + 1)
    out = 0
    for k in range(n):
      den = factorial(k) * factorial(n + 1 - k)
      out += num * cls[k] / den
    return -out / (n + 1)


class BernoulliNumber(metaclass=_MetaBernoulliNumber):
  pass
