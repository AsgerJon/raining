"""
TestExp tests the 'exp' function implemented in the
'worktoy.examples.mathematics' package.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING

from worktoy.examples.mathematics._exp import _exp
from . import MathTest

if TYPE_CHECKING:  # pragma: no cover
  from typing import Any, TypeAlias, Type, Union


class TestExp(MathTest):

  def test_exp(self, ) -> None:
    """Tests the 'exp' function."""
    n = 4
    xMin, xMax = -96, 96
    linSamples = self.linSpace(xMin, xMax, n)
    randSamples = self.randFloats(n, xMin, xMax)
    samples = (*linSamples, *randSamples)
    for z in samples:
      self.assertAlmostEqual(_exp(z), _exp(z / 2) * _exp(z / 2))
