"""
r2rFuncs is 'dict' object providing 'RealToReal' objects for the following
functions:
  'eta',
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
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING

import worktoy.examples.mathematics as wath

if TYPE_CHECKING:  # pragma: no cover
  from typing import Any

r2rFuncs = {
  name: wath.RealToReal(getattr(wath, name))
  for name in (
    'eta',
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
    )
  }
