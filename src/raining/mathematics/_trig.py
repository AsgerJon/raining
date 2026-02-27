"""
This module provides implementations of trigonometric functions.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING

from . import pi
from ...waitaminute import TypeException
from ...waitaminute.mathematics import ConvergenceError, SingularityError

if TYPE_CHECKING:  # pragma: no cover
  from typing import TypeAlias, Union

  Number: TypeAlias = Union[int, float, complex]


def _typeGuardAngle(angle: Number) -> Number:
  if isinstance(angle, (int, float)):
    return float(angle)
  if isinstance(angle, complex):
    if abs(angle.imag) > 1e-16:
      return angle
    return angle.real
  raise TypeException('angle', angle, int, float, complex)


def _modAngle(angle: Number) -> Number:
  angle = _typeGuardAngle(angle)
  if isinstance(angle, complex):
    return angle
  return angle % (2 * pi)


def _clampAngle(angle: Number) -> Number:
  angle = _typeGuardAngle(angle)
  if isinstance(angle, complex):
    return angle
  if abs(angle - pi) < 1e-16 or abs(angle + pi) < 1e-16:
    return pi
  if angle < -pi:
    return _clampAngle(angle + 2 * pi)
  if angle > pi:
    return _clampAngle(angle - 2 * pi)
  return angle


def _cos(angle: float, **kwargs) -> float:
  n = 2
  num = 1
  term = 1.
  out = 1.
  den = 1.
  h = -1 if kwargs.get('h', False) else 1
  while abs(term) > 1e-16 or n < 4:
    num *= -angle * angle * h
    den *= max(n, 1) * max(n - 1, 1)
    term = num / den
    out += term
    n += 2
    if n > 200:
      raise ConvergenceError(200)
  return out


def _sin(angle: float, **kwargs) -> float:
  n = 3
  num = angle
  term = angle
  out = angle
  den = 1.
  h = -1 if kwargs.get('h', False) else 1
  while abs(term) > 1e-16 or n < 4:
    num *= -angle * angle * h
    den *= max(n, 1) * max(n - 1, 1)
    term = num / den
    out += term
    n += 2
    if n > 200:
      raise ConvergenceError(200)
  return out


def _prepAngle(angle: Number) -> float:
  angle = _typeGuardAngle(angle)
  if isinstance(angle, complex):
    raise NotImplementedError
  angle = _modAngle(angle)
  angle = _clampAngle(angle)
  return angle


def cos(angle: Number, **kwargs) -> Number:
  angle = _typeGuardAngle(angle)
  if isinstance(angle, complex):
    a = _cos(angle.real) * _cos(angle.imag, h=True)
    b = -_sin(angle.real) * _sin(angle.imag, h=True)
    return a + b * 1j
  angle = _modAngle(angle)
  angle = _clampAngle(angle)
  if angle < -pi / 2:
    return -cos(angle + pi, **kwargs)
  if angle > pi / 2:
    return -cos(angle - pi, **kwargs)
  if angle > pi / 4:
    return _sin(pi / 2 - angle, **kwargs)
  if angle < -pi / 4:
    return _sin(pi / 2 + angle, **kwargs)
  return _cos(angle, **kwargs)


def sin(angle: Number, **kwargs) -> Number:
  angle = _typeGuardAngle(angle)
  if isinstance(angle, complex):
    a = _sin(angle.real) * _cos(angle.imag, h=True)
    b = _cos(angle.real) * _sin(angle.imag, h=True)
    return a + b * 1j
  angle = _modAngle(angle)
  angle = _clampAngle(angle)
  if angle < -pi / 2:
    return -sin(angle + pi, **kwargs)
  if angle > pi / 2:
    return -sin(angle - pi, **kwargs)
  if angle < -pi / 4:
    return -_cos(pi / 2 + angle, **kwargs)
  if angle > pi / 4:
    return _cos(pi / 2 - angle, **kwargs)
  return _sin(angle, **kwargs)


def tan(angle: Number) -> Number:
  if abs(angle % pi - pi / 2) < 1e-16 or abs(angle % pi + pi / 2) < 1e-16:
    raise SingularityError(angle, csc)
  return sin(angle) / cos(angle)


def sec(angle: Number) -> Number:
  return 1 / cos(angle)


def csc(angle: Number) -> Number:
  if abs(angle % pi) < 1e-16 or abs(angle % pi - pi) < 1e-16:
    raise SingularityError(angle, csc)
  return 1 / sin(angle)


def cot(angle: Number) -> Number:
  return 1 / tan(angle)


def sinh(angle: Number) -> Number:
  if isinstance(angle, complex):
    a = _sin(angle.real, h=True) * cos(angle.imag)
    b = _cos(angle.real, h=True) * sin(angle.imag)
    return a + b * 1j
  return _sin(angle, h=True)


def cosh(angle: Number) -> Number:
  if isinstance(angle, complex):
    a = _cos(angle.real, h=True) * cos(angle.imag)
    b = _sin(angle.real, h=True) * sin(angle.imag)
    return a + b * 1j
  return _cos(angle, h=True)


def tanh(angle: Number) -> Number:
  return sinh(angle) / cosh(angle)


def sech(angle: Number) -> Number:
  return 1 / cosh(angle)


def csch(angle: Number) -> Number:
  if angle < 1e-16:
    raise SingularityError(angle, csch)
  return 1 / sinh(angle)


def coth(angle: Number) -> Number:
  return 1 / tanh(angle)
