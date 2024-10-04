"""RealNumber implements the descriptor to realize values for class owning
real numbers. When instantiated, it takes a default value and a variance."""
#  AGPL-3.0 license
#  Copyright (c) 2024 Asger Jon Vistisen
from __future__ import annotations

from typing import Self, TYPE_CHECKING

from random import gauss
from worktoy.base import FastObject, overload
from worktoy.desc import CoreDescriptor, AttriBox, Field
from worktoy.meta import CallMeMaybe


class RealNumber(FastObject):
  """RealNumber implements the descriptor to realize values for class owning
  real numbers. When instantiated, it takes a default value and a
  variance."""

  __sample_gen__ = None

  @staticmethod
  def _normalFactory(mean: float, stdDev: float) -> CallMeMaybe:
    """Creates a normally distributed random number generator."""

  @classmethod
  def _getFallbackSampleGen(cls, ) -> CallMeMaybe:
    """Returns a random number from a normal distribution."""

    def _getRoll(self: Self, ) -> float:
      """Default roller returns """
      return self.expVal + self.stdDev * gauss(0, 1)

    if TYPE_CHECKING:
      assert isinstance(_getRoll, CallMeMaybe)

    return _getRoll

  @overload(CallMeMaybe)
  def __init__(self, callMeMaybe: CallMeMaybe) -> None:
    self.__sample_gen__ = callMeMaybe

  if TYPE_CHECKING:
    expVal: float
    stdDev: float

  expVal = AttriBox[float](0)
  stdDev = AttriBox[float](1e-09)
  roll = Field()

  @roll.GET
  def _getRoll(self, ) -> float:
    """Returns a random number from a normal distribution."""
    return gauss(self.expVal, self.stdDev)

  def __add__(self, other: object) -> Self:
    """Adds the value of the descriptor to the given value."""
    if isinstance(other, RealNumber):
      return RealNumber(self.expVal + other.expVal,
                        (self.stdDev ** 2 + other.stdDev ** 2) ** 0.5)
    if isinstance(other, (int, float)):
      return RealNumber(self.expVal + other, self.stdDev)
    return NotImplemented

  def __sub__(self, other: object) -> Self:
    """Subtracts the value of the descriptor from the given value."""
    if isinstance(other, RealNumber):
      return RealNumber(self.expVal - other.expVal,
                        (self.stdDev ** 2 + other.stdDev ** 2) ** 0.5)
    if isinstance(other, (int, float)):
      return RealNumber(self.expVal - other, self.stdDev)
    return NotImplemented

  def __mul__(self, other: object) -> Self:
    """Multiplies the value of the descriptor by the given value."""
    if isinstance(other, RealNumber):
      newExpVal = self.expVal * other.expVal
      newStdDev = (self.stdDev / self.expVal) ** 2 + \
                  (other.stdDev / other.expVal) ** 2
      newStdDev = (newStdDev ** 0.5) * abs(newExpVal)
      return RealNumber(newExpVal, newStdDev)
    if isinstance(other, (int, float)):
      return RealNumber(self.expVal * other, self.stdDev * abs(other))
    return NotImplemented

  def __truediv__(self, other: object) -> Self:
    """Divides the value of the descriptor by the given value."""
    if isinstance(other, RealNumber):
      if other.expVal == 0:
        raise ZeroDivisionError("Division by zero.")
      newExpVal = self.expVal / other.expVal
      newStdDev = (self.stdDev / self.expVal) ** 2 + \
                  (other.stdDev / other.expVal) ** 2
      newStdDev = (newStdDev ** 0.5) * abs(newExpVal)
      return RealNumber(newExpVal, newStdDev)
    if isinstance(other, (int, float)):
      if other == 0:
        raise ZeroDivisionError("Division by zero.")
      return RealNumber(self.expVal / other, self.stdDev / abs(other))
    return NotImplemented
