"""This module provides sin, cos and tan"""
#  AGPL-3.0 license
#  Copyright (c) 2024 Asger Jon Vistisen
from __future__ import annotations

import sys

from numba import njit

pi = 3.141592653589793232
eps = sys.float_info.epsilon


@njit
def _clamp(x: float) -> float:
  """Clamp x to the range [-pi, pi]"""
  while x < -pi:
    x += 2 * pi
  while x > pi:
    x -= 2 * pi
  return x


@njit
def sin(x: float) -> float:
  """Sine function"""
  x = _clamp(x)
  if x > pi / 2:
    return sin(pi - x)
  if x < -pi / 2:
    return -sin(x + pi)
  if x > pi / 4:
    return cos(pi / 2 - x)
  if x < -pi / 4:
    return -cos(pi / 2 + x)
  out = 0
  den = 1
  for i in range(63):
    if i:
      den *= i
    term = x ** i / den
    if i % 4 == 1:
      out += term
    if i % 4 == 3:
      out -= term
  return out


@njit
def cos(x: float) -> float:
  """Cos function"""
  x = _clamp(x)
  if x > pi / 2:
    return -cos(x - pi)
  if x < -pi / 2:
    return -cos(x + pi)
  if x > pi / 4:
    return sin(pi / 2 - x)
  if x < -pi / 4:
    return sin(pi / 2 + x)
  out = 0
  den = 1
  for i in range(63):
    if i:
      den *= i
    if i % 2:
      continue
    term = x ** i / den
    if i % 4:
      out -= term
    else:
      out += term
    if abs(term) < eps:
      break
  return out


@njit
def tan(x: float) -> float:
  """Tan function"""
  s, c = sin(x), cos(x)
  if c ** 2 < eps ** 0.5:
    return float('inf')
  return s / c


@njit
def csc(x: float) -> float:
  """Csc function"""
  s = sin(x)
  if s ** 2 < eps ** 0.5:
    return float('inf')
  return 1 / s


@njit
def sec(x: float) -> float:
  """Sec function"""
  c = cos(x)
  if c ** 2 < eps ** 0.5:
    return float('inf')
  return 1 / c


@njit
def cot(x: float) -> float:
  """Cot function"""
  s, c = sin(x), cos(x)
  if s ** 2 < eps ** 0.5:
    return float('inf')
  return c / s
