"""
TestCos tests the implementation of 'cosine' in the
'worktoy.examples.mathematics' package.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

import os
from math import cos as refCos
from math import sin as refSin
from typing import TYPE_CHECKING

from pyperclip import copy

from worktoy.examples.mathematics import pi
from worktoy.examples.mathematics import cos
from worktoy.examples.mathematics import sin

from . import MathTest

if TYPE_CHECKING:  # pragma: no cover
  from typing import Any, TypeAlias, Type, Union


class TestCos(MathTest):

  def setUp(self, ) -> None:
    super().setUp()
    self.randSamples = self.randFloats(-1000, 1000, 4)
    self.edgeSamples = [-2 * pi, ]
    while self.edgeSamples[-1] < 2 * pi:
      self.edgeSamples.append(self.edgeSamples[-1] + pi / 4)

  def test_by_reference(self, ) -> None:
    """Tests the 'cos' function by reference to known values."""

  def test_triangle_identity(self) -> None:
    """
    Testing that for any x, cos(x), sin(x) and 1 form a right triangle.
    """
    for x in self.randSamples:
      self.assertAlmostEqual(cos(x) ** 2 + sin(x) ** 2, 1)

  def test_losses(self) -> None:
    """Tests the losses of the 'cos' function."""
    first = self.randFloat(.0001, 0.001)
    last = first + 2 * pi
    samples = self.linSpace(first, last, 4)
    lines = []
    infoSpec = """cos(%.3f) ** 2 + sin(%.3f) ** 2 = %.3E"""
    for x in samples:
      line = infoSpec % (x, x, sin(x) ** 2 + cos(x) ** 2)
      lines.append(line)
