"""
TestErf tests the implementation of the error function (erf) in the
'worktoy.examples.mathematics' package.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from math import erf as cheats

from pyperclip import copy

from worktoy.examples.mathematics import erf
from . import MathTest

if TYPE_CHECKING:  # pragma: no cover
  from typing import Any, TypeAlias, Type, Union


class TestErf(MathTest):
  """
  The 'TestErf' class contains unit tests for the error function (erf)
  implementation in the 'worktoy.examples.mathematics' package. It verifies
  the correctness of the erf function against known values and properties.
  """

  def test_values(self) -> None:
    """Test the complementary error function property."""
