"""
RealToReal subclasses 'BaseObject' providing the abstract base, not for
audiophile instruments, but for functions mapping from real numbers to
real numbers.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from random import random
from typing import TYPE_CHECKING
from types import FunctionType as Func

from ...desc import Field, FixBox
from ...mcls import BaseObject
from ...utilities import maybe, textFmt, typeCast
from ...waitaminute import MissingVariable, TypeException, VariableNotNone
from ...waitaminute.dispatch import TypeCastException
from ...waitaminute.mathematics import MathError

if TYPE_CHECKING:  # pragma: no cover
  from typing import Self, TypeAlias, Callable, Union, Optional

  R2RFunc: TypeAlias = Callable[[float], float]
  FloatSelf: TypeAlias = Union[float, Self]
  FloatField: TypeAlias = Union[float, Field]
  FloatBox: TypeAlias = Union[float, FixBox]
  MaybeFloat: TypeAlias = Optional[float]
  MaybeInt: TypeAlias = Optional[int]
  IntField: TypeAlias = Union[int, Field]
  FuncField: TypeAlias = Union[R2RFunc, Field]
  SelfFunc: TypeAlias = Union[Self, R2RFunc]
  MaybeFunc: TypeAlias = Optional[SelfFunc]
  FuncFloat: TypeAlias = Union[R2RFunc, float]
  Sample: TypeAlias = tuple[float, float]
  CacheDict: TypeAlias = Optional[dict[int, float]]
  CacheField: TypeAlias = Union[dict[int, float], Field]
  ValueNoise: TypeAlias = tuple[float, ...]
  Noise: TypeAlias = Optional[ValueNoise]
  NoiseField: TypeAlias = Union[Noise, Field]


def baseMethod(func: Func) -> Func:
  setattr(func, '__base_method__', True)
  return func


class _NoCache(ValueError):
  """
  Custom exception raised when value has no associated cached value.
  """


class RealToReal(BaseObject):
  """
  RealToReal subclasses 'BaseObject' providing the abstract base, not for
  audiophile instruments, but for functions mapping from real numbers to
  real numbers.
  """

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #  NAMESPACE  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  #  Class Variables

  #  Fallback Variables
  __fallback_count__: int = 3
  __fallback_min__: float = 1e-08
  __fallback_max__: float = 1e-06

  #  Private Variables
  __wrapped__: MaybeFunc = None
  __cached_values__: CacheDict = None
  __cached_noise__: Noise = None
  __min_step2__: MaybeFloat = None
  __sample_count__: MaybeInt = None
  __min_weight__: MaybeFloat = None
  __max_weight__: MaybeFloat = None

  #  Public Variables
  wrapped: FuncField = Field()
  cache: CacheField = Field()
  noise: NoiseField = Field()
  cacheBin: FloatBox = FixBox[float](1e-07)
  minStepSquared: FloatField = Field()
  sampleCount: IntField = Field()
  minWeight: FloatField = Field()
  maxWeight: FloatField = Field()

  #  Virtual Variables

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #  GETTERS  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  @wrapped.GET
  def _getWrapped(self, ) -> R2RFunc:
    if self.__wrapped__ is None:
      raise MissingVariable(self, '__wrapped__', Func)
    if isinstance(self.__wrapped__, Func):
      return self.__wrapped__
    raise TypeException('__wrapped__', self.__wrapped__, Func)

  def resetNoise(self, ) -> None:
    span = self.maxWeight - self.minWeight
    raw = [random() for _ in range(self.sampleCount)]
    values = [*(self.minWeight + r * span for r in raw), ]
    self.__cached_noise__ = (*sorted(values),)

  @noise.GET
  def _getNoise(self, **kwargs) -> ValueNoise:
    if self.__cached_noise__ is None:
      if kwargs.get('_recursion', False):
        raise RecursionError
      self.resetNoise()
      return self._getNoise(_recursion=True)
    return self.__cached_noise__

  @sampleCount.onSet
  @minWeight.onSet
  @maxWeight.onSet
  def _resetMinStep(self, *args, **kwargs) -> None:
    a, b, n = self.minWeight, self.maxWeight, self.sampleCount
    if n < 1:
      infoSpec = """The 'sampleCount' attribute must be a positive integer, 
      but received: '%d'!"""
      info = textFmt(infoSpec % n)
      raise ValueError(info)
    self.__min_step2__ = (a + (b - a) / (n + 1)) ** 2

  @minStepSquared.GET
  def _getMinStep(self, **kwargs) -> float:
    if self.__min_step2__ is None:
      if kwargs.get('_recursion', False):
        raise RecursionError
      self._resetMinStep()
      return self._getMinStep(_recursion=True)
    return self.__min_step2__

  @sampleCount.GET
  def _getSampleCount(self, ) -> int:
    return maybe(self.__sample_count__, self.__fallback_count__)

  @minWeight.GET
  def _getMinWeight(self, ) -> float:
    return maybe(self.__min_weight__, self.__fallback_min__)

  @maxWeight.GET
  def _getMaxWeight(self, ) -> float:
    return maybe(self.__max_weight__, self.__fallback_max__)

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #  SETTERS  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  @wrapped.SET
  def _setWrapped(self, func: R2RFunc) -> None:
    if not isinstance(func, Func):
      raise TypeException('wrapped', func, Func)
    if self.__wrapped__ is not None:
      raise VariableNotNone('__wrapped__', self.__wrapped__)
    self.__wrapped__ = func

  @sampleCount.SET
  def _setSampleCount(self, count: int) -> None:
    if isinstance(count, complex):
      if abs(count.imag) < 1e-16:
        return self._setSampleCount(int(count.real))
    if isinstance(count, float):
      if float.is_integer(count):
        return self._setSampleCount(int(count))
    if not isinstance(count, int):
      raise TypeException('sampleCount', count, int)
    if count < 1:
      infoSpec = """The 'sampleCount' attribute must be a positive 
      integer, but setter received: '%d'!"""
      info = textFmt(infoSpec % count)
      raise ValueError(info)
    self.__sample_count__ = count
    return None

  @minWeight.SET
  def _setMinWeight(self, weight: float) -> None:
    if isinstance(weight, complex):
      if abs(weight.imag) < 1e-16:
        return self._setMinWeight(float(weight.real))
    if not isinstance(weight, float):
      raise TypeException('minWeight', weight, float)
    if weight < 0:
      infoSpec = """The 'minWeight' attribute must be a non-negative 
      float, but setter received: '%f'!"""
      info = textFmt(infoSpec % weight)
      raise ValueError(info)
    self.__min_weight__ = weight
    return None

  @maxWeight.SET
  def _setMaxWeight(self, weight: float) -> None:
    if isinstance(weight, complex):
      if abs(weight.imag) < 1e-16:
        return self._setMaxWeight(float(weight.real))
    if not isinstance(weight, float):
      raise TypeException('maxWeight', weight, float)
    if weight < 0:
      infoSpec = """The 'maxWeight' attribute must be a non-negative 
      float, but setter received: '%f'!"""
      info = textFmt(infoSpec % weight)
      raise ValueError(info)
    self.__max_weight__ = weight
    return None

  @minWeight.preSet
  def _validateMinWeight(self, weight: float, *args, **kwargs) -> None:
    if weight < 0:
      infoSpec = """The 'minWeight' attribute must be a non-negative 
      float, but setter received: '%f'!"""
      info = textFmt(infoSpec % weight)
      raise ValueError(info)
    if weight >= self.maxWeight:
      infoSpec = """The 'minWeight' attribute must be less than the 
      'maxWeight' attribute, but setter received: minWeight='%f', 
      maxWeight='%f'!"""
      info = textFmt(infoSpec % (weight, self.maxWeight))
      raise ValueError(info)
    return None

  @maxWeight.preSet
  def _validateMaxWeight(self, weight: float, *args, **kwargs) -> None:
    if weight < 0:
      infoSpec = """The 'maxWeight' attribute must be a non-negative 
      float, but setter received: '%f'!"""
      info = textFmt(infoSpec % weight)
      raise ValueError(info)
    if weight <= self.minWeight:
      infoSpec = """The 'maxWeight' attribute must be greater than the 
      'minWeight' attribute, but setter received: minWeight='%f', 
      maxWeight='%f'!"""
      info = textFmt(infoSpec % (self.minWeight, weight))
      raise ValueError(info)
    return None

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #  NOTIFIERS  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #  Python API   # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  def __call__(self, item: FuncFloat) -> FloatSelf:
    if not callable(item):
      if not isinstance(item, (int, float)):
        raise TypeException('item', item, int, float)
    try:
      _ = self.wrapped
    except MissingVariable:
      if not callable(item):
        raise TypeException('item', item, Func)
      self.__wrapped__ = item
    else:
      if callable(self.wrapped):
        try:
          casted = typeCast(float, item, )
        except TypeCastException as typeCastException:
          raise TypeException('item', item, float) from typeCastException
        else:
          out = self.wrapped(casted)
          if isinstance(out, complex):
            if abs(out.imag) < 1e-16:
              out = out.real
          if out != out:
            infoSpec = """The wrapped function: '%s' returned NaN for 
            input: '%f'!"""
            info = textFmt(infoSpec % (self.wrapped.__name__, casted))
            raise MathError(info)
          return out
      raise TypeException('wrapped', self.wrapped, Func)

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #  CONSTRUCTORS   # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  def __init__(self, func: MaybeFunc = None) -> None:
    if func is not None:
      self.wrapped = func

  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  #  DOMAIN SPECIFIC  # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

  def fastDerivative(self, x: float, eps: float = 1e-6) -> float:
    """
    Returns the numerical derivative of the wrapped function at a given point
    using central difference approximation.

    Parameters:
      x (float): The point at which to compute the derivative.
      eps (float): A small value to use for the finite difference
      approximation.

    Returns:
      float: The numerical derivative of the wrapped function at point x.
    """
    return (self(x + eps) - self(x - eps)) / (2 * eps)

  def sharpDerivative(self, x: float) -> float:
    self.resetNoise()
    minStep2 = float(self.minStepSquared)
    S = [(self(x + e) - self(x - e)) / (2.0 * e) for e in self.noise]
    W = [1 / (e ** 2 + minStep2) for e in self.noise]
    return sum(s * w for s, w in zip(S, W)) / sum(W)

  D = sharpDerivative
  d = fastDerivative
