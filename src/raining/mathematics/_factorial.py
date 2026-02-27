"""
The '_cuteFactorial' function computes the _cuteFactorial of a
non-negative integer.
"""
#  AGPL-3.0 license
#  Copyright (c) 2025-2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING

from ...utilities import textFmt
from ...waitaminute import TypeException
from ...waitaminute.mathematics import NumberValueError, DomainError

if TYPE_CHECKING:  # pragma: no cover
  from typing import Any, TypeAlias, Type, Union

  Numerical: TypeAlias = Union[int, float]


def _factorial(n: Numerical) -> int:
  if isinstance(n, float):
    if float.is_integer(n):
      return factorial(int(n), )
    raise TypeException('n', n, int)
  if n < 0:
    raise DomainError(n, 0, float('inf'))
  if n == 0 or n == 1 or n == 2:
    return n or 1
  out = 1
  for i in range(2, int(n) + 1):
    out *= i
  return out


#  Just use the much faster built-in factorial function from the math
#  module, which is implemented in C and optimized for performance. The
#  above implementation is too slow.
import math

factorial = math.factorial
