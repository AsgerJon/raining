"""
The 'etaFunction' returns the Dirichlet eta function at the given complex
argument.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from logging import warn
from typing import TYPE_CHECKING

from . import forwardDifference
from ...waitaminute.mathematics import ConvergenceError

if TYPE_CHECKING:  # pragma: no cover
  pass


def eta(s: complex) -> complex:
  """
    Return the Dirichlet eta function at the given complex argument.

    This function evaluates the alternating zeta-related function commonly
    used in numerical work because it behaves well near s = 1.

    Parameters
    ----------
    s : complex
      The complex input value.

    Returns
    -------
    complex
      The value of the Dirichlet eta function at ``s``.

    References
    ----------
    - NIST Digital Library of Mathematical Functions (DLMF), section on
      zeta and related functions (Dirichlet eta function).
    """
  if not s:
    return 0.5

  delta = forwardDifference(lambda k: 1 / (k + 1) ** s, )
  term = 1.
  m = 0
  out = 0.
  den = 1.

  while abs(term) > 1e-16:
    num = delta(m, 0)
    den *= 2
    term = num / den
    out += term
    m += 1
    if m > 2000:
      raise ConvergenceError(2000)
  return out
