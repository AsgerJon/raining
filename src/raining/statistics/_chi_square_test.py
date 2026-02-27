"""
The 'chiSquareTest' function receives an array of expected frequencies and
an array of outcomes.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
  from typing import Any, TypeAlias, Type, Union

  FloatList: TypeAlias = list[float]
  IntList: TypeAlias = list[int]
  FloatTuple: TypeAlias = tuple[float, ...]
  IntTuple: TypeAlias = tuple[int, ...]
  Floats: TypeAlias = Union[FloatList, FloatTuple]
  Ints: TypeAlias = Union[IntList, IntTuple]


def chiSquareTest(expected: Floats, outcomes: Ints) -> float:
  """Performs a chi-square test on the given expected frequencies and
  outcomes.

  Args:
    expected: A list of expected frequencies for each category.
    outcomes: A list of observed frequencies for each category.
  """

  if len(expected) != len(outcomes):
    raise ValueError("Expected and outcomes must have the same length.")

  sumE = sum(expected)
  E = [e / sumE for e in expected]
  n = sum(outcomes)
  expectedNumber = [e * n for e in E]
  out = 0
  for expected, actual in zip(expectedNumber, outcomes):
    out += (expected - actual) ** 2 / expected
  return out
