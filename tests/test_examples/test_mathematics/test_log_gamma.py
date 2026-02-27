"""
TestLogGamma tests the 'logGamma' function implemented in the
'worktoy.examples.mathematics' package.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from math import lgamma
from typing import TYPE_CHECKING

from worktoy.examples.mathematics import logGamma, log, pi

from . import MathTest

if TYPE_CHECKING:  # pragma: no cover
  from typing import Any, TypeAlias, Type, Union


class TestLogGamma(MathTest):

  def test_log_gamma(self, ) -> None:
    """Tests the logGamma function."""

    samples = self.randFloats(4, 0.1, 100.)
    for z in samples:
      self.assertAlmostEqual(logGamma(z + 1), logGamma(z) + log(z))

  def test_special_values(self) -> None:
    """
    Testing the special values of the 'logGamma' function:
    - logGamma(1) should be 0, since Gamma(1) = 1.
    - logGamma(0.5) should be log(sqrt(pi)), since Gamma(0.5) = sqrt(pi).
    """

    self.assertAlmostEqual(logGamma(1), 0)
    self.assertAlmostEqual(logGamma(0.5), log(pi) / 2)

  def test_by_reference(self) -> None:
    """
    This method tests random values against the 'math.lgamma' function
    from the Python standard library, which computes the logarithm of the
    absolute value of the gamma function.
    """

    n = 4

    logSamples = self.logSpace(1, 1000, n)
    linSamples = self.linSpace(0, 1000, n)
    randSamples = self.randFloats(n, 0.1, 1000.)
    samples = (*logSamples, *linSamples, *randSamples)
    for z in samples:
      if not z:
        continue
      try:
        reference = lgamma(z.real)
        tested = logGamma(z)
      except ZeroDivisionError:
        continue
      else:
        self.assertAlmostEqual(reference, tested)
