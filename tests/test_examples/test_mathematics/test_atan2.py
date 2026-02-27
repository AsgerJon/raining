"""
TestAtan2 tests the 'atan2' function implemented in the
'worktoy.examples.mathematics' package.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING

from worktoy.examples.mathematics import atan2, pi

from . import MathTest

if TYPE_CHECKING:  # pragma: no cover
  from typing import Any, TypeAlias, Type, Union


class TestAtan2(MathTest):

  def test_atan2(self, ) -> None:
    """Tests the 'atan2' function."""

    self.assertLessEqual(atan2(-1, 0.1), 0)
    self.assertFalse(atan2(0.0, 0.0))
