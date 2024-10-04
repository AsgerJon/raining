"""AbstractDistribution provides an abstract baseclass for probability
distributions."""
#  AGPL-3.0 license
#  Copyright (c) 2024 Asger Jon Vistisen
from __future__ import annotations

from worktoy.base import FastObject


class AbstractDistribution(FastObject):
  """AbstractDistribution provides an abstract baseclass for probability
  distributions."""

  def __init__(self):
    super().__init__()

  def pdf(self, x: float) -> float:
    """pdf returns the probability density function at x."""
    raise NotImplementedError

  def cdf(self, x: float) -> float:
    """cdf returns the cumulative distribution function at x."""
    raise NotImplementedError

  def icdf(self, p: float) -> float:
    """icdf returns the inverse cumulative distribution function at p."""
    raise NotImplementedError

  def sample(self, ) -> float:
    """sample returns a random sample from the distribution."""
    raise NotImplementedError
