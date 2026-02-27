"""
The 'worktoy.utilities.mathematics' module provides mathematical utilities
used across the 'worktoy' library.
"""
#  AGPL-3.0 license
#  Copyright (c) 2025-2026 Asger Jon Vistisen
from __future__ import annotations

from ._constants import pi, arcTan, atan2
from ._real_to_real import RealToReal
from ._trig import sin, cos, sinh, cosh, tan, tanh
from ._trig import sec, csc, cot, sech, csch, coth
from ._exp import exp, e
from ._erf import erf
from ._derivative import derivative
from ._factorial import factorial
from ._bernoulli_number import BernoulliNumber
from ._binomial_choose import binomialChoose
from ._forward_difference import forwardDifference
from ._eta_function import eta
from ._log import log
from ._log_gamma import logGamma
from ._euler_mascheroni import eulerMascheroni

__all__ = [
  'pi',
  'arcTan',
  'atan2',
  'RealToReal',
  'derivative',
  'factorial',
  'BernoulliNumber',
  'binomialChoose',
  'forwardDifference',
  'eta',
  'eulerMascheroni',
  'e',
  'log',
  'exp',
  'sin',
  'cos',
  'tan',
  'cosh',
  'sinh',
  'tanh',
  'sec',
  'csc',
  'cot',
  'sech',
  'csch',
  'coth',
  'logGamma',
  'erf',
  ]
