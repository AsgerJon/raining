"""
TestEulerMascheroni tests the Euler-Mascheroni constant.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING

from worktoy.examples.mathematics import eulerMascheroni, derivative, \
  eta, log
from . import MathTest

if TYPE_CHECKING:  # pragma: no cover
  from typing import Any, TypeAlias, Type, Union


class TestEulerMascheroni(MathTest):

  def test_euler_mascheroni(self, ) -> None:
    """Tests the Euler-Mascheroni constant."""
