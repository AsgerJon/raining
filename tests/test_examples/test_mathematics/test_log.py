"""
TestLog tests the 'log' function implemented in the
'worktoy.examples.mathematics' package.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING

from worktoy.examples.mathematics._log import _log
from . import MathTest

if TYPE_CHECKING:  # pragma: no cover
  from typing import Any, TypeAlias, Type, Union


class TestLog(MathTest):

  def test_log(self, ) -> None:
    """Tests the 'log' function."""

    self.assertAlmostEqual(2 * _log(0.8), _log(0.64))
