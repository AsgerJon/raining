"""The 'exp' function computes the exponential function."""
#  AGPL-3.0 license
#  Copyright (c) 2024 Asger Jon Vistisen
from __future__ import annotations

import sys
from math import floor

from numba import njit

eps = sys.float_info.epsilon


@njit
def exp(x: float) -> float:
  """exp returns the exponential function of x."""
  if not x:
    return 1
  if x > 32:
    return float('inf')
  if x < 0:
    return 1 / exp(-x)
  if x > 1:
    return exp(1) ** floor(x) * exp(x - floor(x))

  out = 1
  term = 1
  den = 1
  term = 0
  for i in range(1, 32):
    den *= i
    term = x ** i / den
    out += term
    if abs(term) < eps ** 0.5 * abs(out):
      break
  return out


@njit
def log(x: float) -> float:
  """Log function"""
  if not x:
    return float('-inf')
  if (x - 1) ** 2 < eps ** 0.5:
    return 0
  log2 = 0.693147180559945
  log3 = 1.0986122886681091
  if x < 0:
    return float('nan')
  if x < 1:
    return -log(1 / x)
  if x >= 3 / 2:
    return log(x * 2 / 3) - log2 + log3
  out = 0
  term = 1
  for i in range(1, 63):
    term = (x - 1) ** i / i
    out += (term if i % 2 != 0 else -term)
    if abs(term) < eps * abs(out) and i:
      break
  return out


@njit
def cosh(x: float) -> float:
  """Cosh function"""
  return (exp(x) + exp(-x)) / 2


@njit
def sinh(x: float) -> float:
  """Sinh function"""
  return (exp(x) - exp(-x)) / 2


@njit
def tanh(x: float) -> float:
  """Tanh function"""
  s, c = sinh(x), cosh(x)
  if c is float('inf'):
    if s is float('inf'):
      return 1
    if s is -float('inf'):
      return -1
    return 0
  if c:
    if s is float('inf'):
      return float('inf')
    if s is -float('inf'):
      return -float('inf')
    return s / c
  if s:
    return float('inf')
  return float('nan')


@njit
def coth(x: float) -> float:
  """Coth function"""
  s, c = sinh(x), cosh(x)
  if c is float('inf'):
    if s is float('inf'):
      return 1
    if s is -float('inf'):
      return -1
    return float('inf')
  if s is float('inf'):
    return 0
  if s and c:
    return c / s
  return float('nan')


@njit
def sech(x: float) -> float:
  """Sech function"""
  c = cosh(x)
  if c is float('inf'):
    return 0
  if c:
    return 1 / c
  return float('inf')


@njit
def csch(x: float) -> float:
  """Csch function"""
  s = sinh(x)
  if s is float('inf') or s is -float('inf'):
    return 0
  if s:
    return 1 / s
  return float('inf')


@njit
def arcsinh(x: float) -> float:
  """Arcsinh function"""
  return log(x + (x ** 2 + 1) ** 0.5)


@njit
def arccosh(x: float) -> float:
  """Arccosh function"""
  if x < 1:
    return float('nan')
  if x - 1:
    return log(x + (x ** 2 - 1) ** 0.5)
  return 0


@njit
def arctanh(x: float) -> float:
  """Arctanh function"""
  if abs(x) >= 1:
    return float('nan')
  return 0.5 * log((1 + x) / (1 - x))


@njit
def arccoth(x: float) -> float:
  """Arccoth function"""
  if x:
    return 0.5 * log((x + 1) / (x - 1))
  return float('nan')


@njit
def arcsech(x: float) -> float:
  """Arcsech function"""
  if x <= 0 or x >= 1:
    return float('nan')
  return log((1 + (1 - x ** 2) ** 0.5) / x)


@njit
def arccsch(x: float) -> float:
  """Arccsch function"""
  if x:
    return log(1 / x + (1 + 1 / x ** 2) ** 0.5)
  return float('nan')
