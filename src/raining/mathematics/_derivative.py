"""
The 'derivative' function takes a mapping from complex to complex and a
value and it returns a numerical approximation of the derivative of the
function at the given value using the limit definition of the derivative.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
  from typing import TypeAlias, Callable

  Func: TypeAlias = Callable[[complex], complex]


def derivative(func: Func, z: complex) -> complex:
  """Returns a numerical approximation of the derivative of 'func' at 'z'."""
  h = 1e-08
  return (func(z + h * 1j) - func(z - h * 1j)) / (2 * h * 1j)
