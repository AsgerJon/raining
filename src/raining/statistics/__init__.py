"""
The 'worktoy.utilities.statistics' package provides a collection of tools
and functions for performing statistical analysis and computations. It
includes various statistical methods, distributions, and utilities
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from ._chi_square_test import chiSquareTest

__all__ = [
  'chiSquareTest',
  ]
