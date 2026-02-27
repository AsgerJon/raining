"""
TestBernoulliNumber tests the 'BernoulliNumber' class from the
'worktoy.examples.mathematics' package.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pyperclip import copy

from worktoy.examples.mathematics import BernoulliNumber

from . import MathTest

if TYPE_CHECKING:  # pragma: no cover
  from typing import Any, TypeAlias, Type, Union


class TestBernoulliNumber(MathTest):

  def test_bernoulli_number(self, ) -> None:
    """Tests the Bernoulli numbers."""
