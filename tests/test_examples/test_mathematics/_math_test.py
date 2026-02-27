"""
MathTest subclasses 'worktoy.work_test.BaseTest' and provides the base
class for test classes in the 'tests.test_examples.test_mathematics' package.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING

from worktoy.work_test import BaseTest

if TYPE_CHECKING:  # pragma: no cover
  from typing import Any, TypeAlias, Type, Union


class MathTest(BaseTest):
  """
  The 'MathTest' class serves as the base test class for all test cases in
  the 'tests.test_examples.test_mathematics' package. It inherits from
  'worktoy.work_test.BaseTest' and provides common setup and utilities for
  testing mathematical functions and algorithms.
  """
