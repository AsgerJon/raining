"""TestHyperbolic tests the hyperbolic functions"""
#  AGPL-3.0 license
#  Copyright (c) 2024 Asger Jon Vistisen
from __future__ import annotations

import sys

from random import random
from unittest import TestCase
from raining.core import sinh, cosh, tanh, coth, sech, csch
from raining.core import arccosh, arccoth, arccsch, arcsech, arcsinh, arctanh

eps = sys.float_info.epsilon


class TestHyperbolic(TestCase):
  """TestHyperbolic tests the hyperbolic functions"""

  def setUp(self) -> None:
    """Sets up the values"""
    self.sampleValues = [10 * random() for _ in range(31)]
    self.limit = eps ** 0.25
    self.positiveValues = [abs(2 ** 31 * random()) + 1 for _ in
                           range(31)]  # Positive values for functions
    # needing positive inputs
    self.nonZeroValues = [2 ** 31 * random() - 2 ** 30 for _ in range(31) if
                          random() - 0.5]  # Non-zero values for functions
    # needing non-zero inputs
    self.values = [2 ** 31 * random() - 2 ** 30 for _ in
                   range(31)]  # General values for functions with wider
    # domains

  def test_id(self) -> None:
    """Testing that the hyperbolic sine and cosine functions satisfies the
    expected identities"""
    for value in self.sampleValues:
      left = cosh(value) ** 2 - sinh(value) ** 2
      right = 1
      loss = (left - right) ** 2
      if left is float('inf') or float('nan'):
        continue
      self.assertLess(loss, self.limit)

  def test_edges(self) -> None:
    """Testing correct handling of inf and 0"""
    self.assertAlmostEqual(sech(float('inf')), 0, delta=self.limit)
    self.assertAlmostEqual(cosh(float('inf')),
                           float('inf'),
                           delta=self.limit)
    self.assertAlmostEqual(csch(float('inf')), 0, delta=self.limit)
    self.assertAlmostEqual(sinh(float('inf')),
                           float('inf'),
                           delta=self.limit)
    self.assertAlmostEqual(coth(float('inf')), 1, delta=self.limit)
    self.assertAlmostEqual(tanh(float('inf')), 1, delta=self.limit)
    self.assertAlmostEqual(coth(-float('inf')), -1, delta=self.limit)
    self.assertAlmostEqual(tanh(-float('inf')), -1, delta=self.limit)

  def test_asymptoticBehavior(self) -> None:
    """Tests the asymptotic behavior of tanh at large magnitudes."""
    self.assertAlmostEqual(tanh(1e10), 1.0, delta=self.limit)
    self.assertAlmostEqual(tanh(-1e10), -1.0, delta=self.limit)

  def test_reciprocalRelations(self) -> None:
    """Tests reciprocal relationships between functions."""
    for value in self.sampleValues:
      limit = self.limit * tanh(value)
      if tanh(value) != 0:
        self.assertAlmostEqual(coth(value), 1 / tanh(value), delta=limit)
      if cosh(value) != 0:
        limit = self.limit * cosh(value)
        self.assertAlmostEqual(sech(value), 1 / cosh(value), delta=limit)

  def test_hyperbolicIdentities(self) -> None:
    """Tests more complex hyperbolic identities."""
    for value in self.sampleValues:
      limit = self.limit * 2 * max(cosh(value * 2), sinh(2 * value))
      self.assertAlmostEqual(cosh(2 * value),
                             2 * cosh(value) ** 2 - 1, delta=limit)
      self.assertAlmostEqual(sinh(2 * value),
                             2 * sinh(value) * cosh(value), delta=limit)

  def test_monotonicity(self) -> None:
    """Tests that sinh and tanh are monotonic functions."""
    sortedValues = sorted(self.sampleValues)
    for i in range(len(sortedValues) - 1):
      self.assertLessEqual(sinh(sortedValues[i]), sinh(sortedValues[i + 1]))
      self.assertLessEqual(tanh(sortedValues[i]), tanh(sortedValues[i + 1]))

  def test_arcsinh_values(self) -> None:
    """Tests for known values"""
    vals = []
    while len(vals) < 32:
      vals.append(-10 + 20 * random())
    vals.sort()
    for val in vals:
      result = sinh(val)
      left = arcsinh(result)
      right = val
      lim = self.limit * max(abs(left), abs(right))
      self.assertAlmostEqual(left, right, delta=lim)

  def test_arccosh_values(self) -> None:
    """Tests arccosh with values ensuring arccosh(cosh(x)) = x, x >= 1."""
    for val in [1 + 9 * random() for _ in range(32)]:
      result = cosh(val)
      left, right = arccosh(result), val
      lim = self.limit * max(abs(left), abs(right))
      self.assertAlmostEqual(left, right, delta=lim)

  def test_arctanh_values(self) -> None:
    """Tests arctanh with values ensuring arctanh(tanh(x)) = x, |x| < 1."""
    for val in sorted([-1 + 2 * random() for _ in range(32)]):
      result = tanh(val)
      left, right = arctanh(result), val
      lim = self.limit * max(abs(left), abs(right))
      self.assertAlmostEqual(left, right, delta=lim)

  def test_arccoth_values(self) -> None:
    """Tests arccoth with values ensuring arccoth(coth(x)) = x, |x| > 1."""
    for val in sorted([-10 + 9 * random() if random() < 0.5 else
                       1 + 9 * random() for _ in range(32)]):
      result = coth(val)
      left, right = arccoth(result), val
      lim = self.limit * max(abs(left), abs(right))
      self.assertAlmostEqual(left, right, delta=lim)

  def test_arccsch_values(self) -> None:
    """Tests arccsch with values ensuring arccsch(csch(x)) = x, x != 0."""
    for val in sorted([-10 + 20 * random() for _ in range(32) if
                       random() - 0.5]):
      result = csch(val)
      if val:
        left, right = arccsch(result), val
        lim = self.limit * max(abs(left), abs(right))
        self.assertAlmostEqual(left, right, delta=lim)

  def test_arcsech_values(self) -> None:
    """Tests arcsech with values ensuring arcsech(sech(x)) = x, 0 < x <=
    1."""
    for val in sorted([0.01 + 0.99 * random() for _ in range(32)]):
      result = sech(val)
      left, right = arcsech(result), val
      lim = self.limit * max(abs(left), abs(right))
      self.assertAlmostEqual(left, right, delta=lim)
