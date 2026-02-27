"""
The 'eulerMascheroni' constant the limit of the difference between the
harmonic series and the natural logarithm for increasing positive integers.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING

from worktoy.examples.mathematics import eta, log

if TYPE_CHECKING:  # pragma: no cover
  from typing import Self

log2 = eta(1.0)
d_eta_1 = eta(1 + 1e-12 * 1j).imag / 1e-12
eulerMascheroni = (d_eta_1 / log2 + log2 / 2 + 0j).real
