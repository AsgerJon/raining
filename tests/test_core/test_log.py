"""TestLog tests that log behaves correctly"""
#  AGPL-3.0 license
#  Copyright (c) 2024 Asger Jon Vistisen
from __future__ import annotations

import sys
import math
from random import random
from unittest import TestCase

from raining.core import exp, log

eps = sys.float_info.epsilon


class TestLog(TestCase):
  """TestLog tests that log behaves correctly"""

  def setUp(self, ) -> None:
    """This method sets up the test fixture before exercising it."""
    self.sampleValues = [2 ** i * random() for i in range(1, 23)]

  def test_log(self) -> None:
    """Testing that the log function behaves as expected."""
    for value in self.sampleValues:
      left = math.log(value)
      right = log(value)
      limit = 1e-03
      self.assertAlmostEqual(left, right, delta=limit)
