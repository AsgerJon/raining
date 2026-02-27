"""
TestRealToReal tests the 'worktoy.examples.mathematics.realToReal' class.
"""
#  AGPL-3.0 license
#  Copyright (c) 2026 Asger Jon Vistisen
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from worktoy.desc import BaseDescriptor
from worktoy.examples.mathematics import RealToReal, pi
from worktoy.mcls import BaseMeta
from worktoy.mcls import BaseSpace as BSpace
from worktoy.waitaminute import TypeException
from worktoy.waitaminute.mathematics import DomainError, SingularityError, \
  MathError
from . import MathTest, r2rFuncs

if TYPE_CHECKING:  # pragma: no cover
  from typing import TypeAlias, Any, Iterator, Union

  Bases: TypeAlias = tuple[type, ...]

MathExceptions = (
  MathError,
  ZeroDivisionError,
  OverflowError,
  )


class DescR2R(BaseDescriptor):
  """Descriptor for 'RealToReal' objects."""

  __r2r_object__: RealToReal

  def __init__(self, callMeMaybe: RealToReal) -> None:  # noqa
    self.__r2r_object__ = callMeMaybe

  def __get__(self, *_) -> RealToReal:
    return self.__r2r_object__


class R2RMeta(BaseMeta):
  """Metaclass for 'RealToReal' objects."""

  def __iter__(cls, ) -> Iterator[RealToReal]:
    yield from cls()

  @staticmethod
  def _iterFactory() -> Callable:
    def __iter__(self, ) -> Iterator[RealToReal]:
      for key in r2rFuncs.keys():
        yield getattr(self, key)

    return __iter__

  @classmethod
  def __prepare__(mcls, name: str, bases: Bases, **kw) -> BSpace:
    bSpace = super().__prepare__(name, bases, **kw)
    for key, val in r2rFuncs.items():
      bSpace[key] = DescR2R(val)
    bSpace['__iter__'] = mcls._iterFactory()
    return bSpace


class R2R(metaclass=R2RMeta):
  log: Union[Callable[[float], float], RealToReal]
  exp: Union[Callable[[float], float], RealToReal]
  sin: Union[Callable[[float], float], RealToReal]
  cos: Union[Callable[[float], float], RealToReal]
  tan: Union[Callable[[float], float], RealToReal]
  cosh: Union[Callable[[float], float], RealToReal]
  sinh: Union[Callable[[float], float], RealToReal]
  tanh: Union[Callable[[float], float], RealToReal]
  sec: Union[Callable[[float], float], RealToReal]
  csc: Union[Callable[[float], float], RealToReal]
  cot: Union[Callable[[float], float], RealToReal]
  sech: Union[Callable[[float], float], RealToReal]
  csch: Union[Callable[[float], float], RealToReal]
  coth: Union[Callable[[float], float], RealToReal]
  logGamma: Union[Callable[[float], float], RealToReal]
  erf: Union[Callable[[float], float], RealToReal]
  eta: Union[Callable[[float], float], RealToReal]


class TestRealToReal(MathTest):
  """Tests the 'worktoy.examples.mathematics.realToReal' class."""

  def setUp(self, ) -> None:
    """Sets up the test case."""
    super().setUp()

    self.samples = self.randFloats(8, 0.01, 10)
    self.contracts = (
      # Trig
      lambda x: (R2R.sin.D(x), R2R.cos(x)),
      lambda x: (R2R.cos.D(x), -R2R.sin(x)),
      lambda x: (R2R.tan.D(x), R2R.sec(x) ** 2),
      lambda x: (R2R.sec.D(x), R2R.sec(x) * R2R.tan(x)),
      lambda x: (R2R.csc.D(x), -R2R.csc(x) * R2R.cot(x)),
      lambda x: (R2R.cot.D(x), -(R2R.csc(x) ** 2)),

      # Hyperbolic
      lambda x: (R2R.cosh.D(x), R2R.sinh(x)),
      lambda x: (R2R.sinh.D(x), R2R.cosh(x)),
      lambda x: (R2R.tanh.D(x), 1 - R2R.tanh(x) ** 2),
      lambda x: (R2R.sech.D(x), -R2R.sech(x) * R2R.tanh(x)),
      lambda x: (R2R.csch.D(x), -R2R.csch(x) * R2R.coth(x)),
      lambda x: (R2R.coth.D(x), -(R2R.csch(x) ** 2)),

      # Log / Exp
      lambda x: (R2R.exp.D(x), R2R.exp(x)),
      lambda x: (R2R.log.D(x), 1 / x),

      # Erf
      lambda x: (
        R2R.erf.D(x), (2 / (pi ** 0.5)) * R2R.exp(-(x ** 2)),
        ),
      )

  def test_callable(self, ) -> None:
    """
    Tests that the 'RealToReal' objects are callable
    """
    for func in R2R:
      self.assertIsInstance(func, RealToReal)
      self.assertTrue(callable(func))

  def test_contracts(self, ) -> None:
    """
    Tests that the 'RealToReal' objects satisfy their contracts
    """
    for i, contract in enumerate(self.contracts):
      good, bad = 0, 0
      for sample in self.samples:
        try:
          left, right = contract(sample)
        except OverflowError:
          pass
        else:
          if left == left and right == right:
            self.assertAlmostEqual(left, right, )
            continue
          if isinstance(left, complex):
            if left.real == left.real and left.imag == left.imag:
              if right.real == right.real and right.imag == right.imag:
                self.assertAlmostEqual(left, right, )
          print("""Contract %d has fucking nan!""" % i)

  def test_easy(self, ) -> None:
    """Tests that the 'RealToReal' objects satisfy some easy cases."""

    @RealToReal
    def square(x_: float) -> float:
      return x_ * x_

    for x in self.linSpace(-100, 100, 8):
      self.assertAlmostEqual(square.D(x), 2 * x, )
      self.assertAlmostEqual(R2R.sin(x) ** 2 + R2R.sin(pi / 2 - x) ** 2, 1, )

  def test_chain_rule(self) -> None:
    """Tests that the 'RealToReal' objects satisfy the chain rule."""

    names = ('cos', 'sin', 'exp', 'log')
    for outerName in names:
      outerFunc = r2rFuncs[outerName]
      for innerName in names:
        innerFunc = r2rFuncs[innerName]

        @RealToReal
        def chain(x: float) -> float:
          return outerFunc(innerFunc(x))

        for x in self.linSpace(0.01, 10, 8):
          try:
            left = chain.D(x)
            right = outerFunc.D(innerFunc(x)) * innerFunc.D(x)
          except MathExceptions:
            pass
          else:
            self.assertAlmostEqual(left, right, )
