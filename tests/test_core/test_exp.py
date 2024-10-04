"""TestExp tests the exponential function"""
#  AGPL-3.0 license
#  Copyright (c) 2024 Asger Jon Vistisen
from __future__ import annotations
import sys
from random import random
from unittest import TestCase

from raining.core import exp

eps = sys.float_info.epsilon


class TestExp(TestCase):
  """TestExp tests the exponential function"""

  def setUp(self, ) -> None:
    """This method sets up the test fixture before exercising it."""
    self.sampleValues = [(2 ** i) * random() for i in range(-16, 16)]

  def test_exp(self) -> None:
    """Testing that the exponential function behaves as expected."""
    for value in self.sampleValues:
      left = exp(value + 1)
      right = exp(value) * exp(1)
      limit = eps ** 0.5 * max(abs(left), abs(right))
      self.assertAlmostEqual(left, right, delta=limit)
      self.assertAlmostEqual(exp(-value), 1 / exp(value), delta=limit)
      