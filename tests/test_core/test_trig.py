"""TestTrig tests trigonometric functions"""
#  AGPL-3.0 license
#  Copyright (c) 2024 Asger Jon Vistisen
from __future__ import annotations
import sys
from unittest import TestCase
from raining.core import pi, sin, cos, sec, csc, cot, tan

eps = sys.float_info.epsilon


class TestTrig(TestCase):
  """TestTrig tests trigonometric functions"""

  def test_id(self, ) -> None:
    """Testing that sin(x)**2 + cos(x)**2 == 1"""
    for i in range(1024):
      x = -pi / 4 + i / 1023 * pi / 2
      left = cos(x) ** 2 + sin(x) ** 2
      right = 1
      loss = (left - right) ** 2
      limit = eps ** 0.5
      if limit < loss:
        print(i, x, x * pi)
      self.assertLess(loss, limit)

  def test_vals(self) -> None:
    """Testing known values"""
    limit = eps ** 0.5
    self.assertAlmostEqual(cos(pi / 4), sin(pi / 4), delta=limit)
    self.assertAlmostEqual(sin(pi / 4), 2 ** 0.5 / 2, delta=limit)
    self.assertAlmostEqual(cos(pi / 4), 2 ** 0.5 / 2, delta=limit)
    self.assertAlmostEqual(sin(pi / 6), 1 / 2, delta=limit)
    self.assertAlmostEqual(cos(pi / 6), 3 ** 0.5 / 2, delta=limit)
    self.assertAlmostEqual(sin(pi / 3), 3 ** 0.5 / 2, delta=limit)
    self.assertAlmostEqual(cos(pi / 3), 1 / 2, delta=limit)
    self.assertAlmostEqual(sin(pi / 2), 1, delta=limit)
    self.assertAlmostEqual(cos(pi / 2), 0, delta=limit)
    self.assertAlmostEqual(sin(pi), 0, delta=limit)
    self.assertAlmostEqual(cos(pi), -1, delta=limit)
    self.assertAlmostEqual(sin(3 * pi / 2), -1, delta=limit)
    self.assertAlmostEqual(cos(3 * pi / 2), 0, delta=limit)
    self.assertAlmostEqual(sin(2 * pi), 0, delta=limit)
    self.assertAlmostEqual(cos(2 * pi), 1, delta=limit)

  def test_tan(self) -> None:
    """Testing the tangent function"""
    limit = eps ** 0.5
    self.assertAlmostEqual(tan(0), 0, delta=limit)
    self.assertAlmostEqual(tan(pi / 4), 1, delta=limit)
    self.assertAlmostEqual(tan(pi / 6), 3 ** 0.5 / 3, delta=limit)
    self.assertAlmostEqual(tan(pi / 3), 3 ** 0.5, delta=limit)
    self.assertAlmostEqual(tan(pi / 2), float('inf'), delta=limit)
    self.assertAlmostEqual(tan(pi), 0, delta=limit)
    self.assertAlmostEqual(tan(3 * pi / 2), float('inf'), delta=limit)
    self.assertAlmostEqual(tan(2 * pi), 0, delta=limit)

  def test_cot(self, ) -> None:
    """Testing the cotangent function"""
    limit = eps ** 0.5
    self.assertAlmostEqual(cot(0), float('inf'), delta=limit)
    self.assertAlmostEqual(cot(pi / 4), 1, delta=limit)
    self.assertAlmostEqual(cot(pi / 6), 3 ** 0.5, delta=limit)
    self.assertAlmostEqual(cot(pi / 3), 3 ** 0.5 / 3, delta=limit)
    self.assertAlmostEqual(cot(pi / 2), 0, delta=limit)
    self.assertAlmostEqual(cot(pi), float('inf'), delta=limit)
    self.assertAlmostEqual(cot(3 * pi / 2), 0, delta=limit)
    self.assertAlmostEqual(cot(2 * pi), float('inf'), delta=limit)

  def test_csc(self, ) -> None:
    """Testing the cosecant function"""
    limit = eps ** 0.5
    self.assertAlmostEqual(csc(0), float('inf'), delta=limit)
    self.assertAlmostEqual(csc(pi / 4), 2 ** 0.5, delta=limit)
    self.assertAlmostEqual(csc(pi / 6), 2, delta=limit)
    self.assertAlmostEqual(csc(pi / 2), 1, delta=limit)
    self.assertAlmostEqual(csc(pi), float('inf'), delta=limit)
    self.assertAlmostEqual(csc(3 * pi / 2), -1, delta=limit)
    self.assertAlmostEqual(csc(2 * pi), float('inf'), delta=limit)

  def test_sec(self, ) -> None:
    """Testing the secant function"""
    limit = eps ** 0.5
    self.assertAlmostEqual(sec(0), 1, delta=limit)
    self.assertAlmostEqual(sec(pi / 4), 2 ** 0.5, delta=limit)
    self.assertAlmostEqual(sec(pi / 3), 2, delta=limit)
    self.assertAlmostEqual(sec(pi / 2), float('inf'), delta=limit)
    self.assertAlmostEqual(sec(pi), -1, delta=limit)
    self.assertAlmostEqual(sec(3 * pi / 2), float('inf'), delta=limit)
    self.assertAlmostEqual(sec(2 * pi), 1, delta=limit)
