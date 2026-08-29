"""Constructores pequeños para escribir modelos sin acoplarse a un backend."""

from __future__ import annotations

from dataclasses import dataclass, field

from .ir import Expr, Index, Scalar, Tensor, Variance, function, mul
from .model import GeometrySymbols


@dataclass(frozen=True, slots=True)
class ModelBuilder:
    symbols: GeometrySymbols = field(default_factory=GeometrySymbols)

    def up(self, name: str) -> Index:
        return Index(name, Variance.UP, self.symbols.index_space)

    def down(self, name: str) -> Index:
        return Index(name, Variance.DOWN, self.symbols.index_space)

    @property
    def phi(self) -> Scalar:
        return Scalar(self.symbols.scalar)

    def scalar(self, name: str) -> Scalar:
        return Scalar(name)

    def metric(self, first: str, second: str) -> Tensor:
        return Tensor(self.symbols.metric, (self.up(first), self.up(second)))

    def riemann(self, a: str, b: str, c: str, d: str) -> Tensor:
        return Tensor(
            self.symbols.curvature,
            (self.down(a), self.down(b), self.down(c), self.down(d)),
        )

    def scalar_gradient(self, index: str) -> Tensor:
        return Tensor(self.symbols.scalar_gradient, (self.down(index),))

    def function(self, name: str, *arguments: Expr) -> Expr:
        """Construye una función escalar declarable sin sintaxis de backend."""

        return function(name, *arguments)

    def ricci_scalar(self) -> Expr:
        """Invariante escalar R=g^ac g^bd R_abcd con la convención activa."""

        return mul(
            self.metric("a", "c"),
            self.metric("b", "d"),
            self.riemann("a", "b", "c", "d"),
        )

    def kinetic_scalar(self) -> Expr:
        """Invariante X=g^ab u_a u_b, con u_a tratado como argumento independiente."""

        return mul(
            self.metric("a", "b"),
            self.scalar_gradient("a"),
            self.scalar_gradient("b"),
        )
