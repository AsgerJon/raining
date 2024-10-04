"""TestErf tests the error function implementation.  """
#  AGPL-3.0 license
#  Copyright (c) 2024 Asger Jon Vistisen
from __future__ import annotations

import sys
from random import random

from raining.core import erf, erfc, erfinv, erfcinv
from unittest import TestCase


class TestErf(TestCase):
  """TestErf tests the error function implementation.  """

  def setUp(self) -> None:
    """This method sets up the test fixture before exercising it."""
    for i in range(7):
      print(i, 'erf', erf(i))
    self.limit = sys.float_info.epsilon
    self.unitValues = [-1 + 2 * random() for _ in range(1, 23)]
    self.maxLim = 2  # Beyond this limit, 1 is an acceptable return
    seedSample = [random() for _ in range(1, 23)]
    minVal, spanVal = -self.maxLim, 2 * self.maxLim
    self.values = sorted([minVal + spanVal * i for i in seedSample])

  def test_complementary(self, ) -> None:
    """Test that the complementary error function is correct."""
    for value in self.values:
      left = 1 - erf(value)
      right = erfc(value)
      lim = self.limit * max(abs(left), abs(right))
      if left == left and right == right:
        self.assertAlmostEqual(left, right, delta=lim)
    for value in self.unitValues:  # Testing inverses
      left = erfinv(1 - value)
      right = erfcinv(value)
      lim = self.limit * max(abs(left), abs(right))
      if left == left and right == right:
        self.assertAlmostEqual(left, right, delta=lim)

  def test_boundaries(self) -> None:
    """Tests the values at zero and at max value."""
    lim = self.limit * self.maxLim
    left = erf(0)
    right = 0
    self.assertAlmostEqual(left, right, delta=lim)
    left = erfc(0)
    right = 1
    self.assertAlmostEqual(left, right, delta=lim)
    left = erf(self.maxLim)
    right = 1
    self.assertAlmostEqual(left, right, delta=lim)
    left = erfc(self.maxLim)
    right = 0
    self.assertAlmostEqual(left, right, delta=lim)
    left = erf(-self.maxLim)
    right = -1
    self.assertAlmostEqual(left, right, delta=lim)
    left = erfc(-self.maxLim)
    right = 2
    self.assertAlmostEqual(left, right, delta=lim)
    left = erfinv(-0.999)
    self.assertLess(left, -self.maxLim / 2)
    left = erfcinv(0.999)
    self.assertLess(left, -self.maxLim / 2)
    left = erfinv(0.999)
    self.assertGreater(left, self.maxLim / 2)
    left = erfcinv(-0.999)
    self.assertGreater(left, self.maxLim / 2)
    left = erfinv(0)
    right = 0
    self.assertAlmostEqual(left, right, delta=lim)
    left = erfcinv(1)
    right = 0
    self.assertAlmostEqual(left, right, delta=lim)
