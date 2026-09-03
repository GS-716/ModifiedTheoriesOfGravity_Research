"""Registro extensible de azúcar sintáctico que se expande a la IR existente.

No contiene reglas variacionales, geometría coordenada ni familias de teorías.
Los constructores son código Python de confianza; nunca proceden de la fuente.
"""

from __future__ import annotations

from dataclasses import dataclass
import keyword
import re
from typing import Callable

from .builders import ModelBuilder
from .errors import SourceCompilationError
from .ir import Expr


TENSOR_SOURCE_CONSTRUCTORS = frozenset({"contract", "Riemann", "metric", "gradient"})
_ALIAS_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class InvariantSpec:
    alias: str
    constructor: Callable[[ModelBuilder], Expr]
    description: str
    version: str = "1"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.alias, str)
            or not _ALIAS_RE.fullmatch(self.alias)
            or keyword.iskeyword(self.alias)
            or self.alias in TENSOR_SOURCE_CONSTRUCTORS
        ):
            raise SourceCompilationError(f"Alias de invariante inválido o reservado: {self.alias!r}.")
        if not callable(self.constructor):
            raise SourceCompilationError("Un invariante requiere un constructor de IR.")
        if not isinstance(self.version, str) or not self.version:
            raise SourceCompilationError("La versión del invariante debe ser una cadena no vacía.")

    def expand(self, builder: ModelBuilder) -> Expr:
        expression = self.constructor(builder)
        if not isinstance(expression, Expr) or not expression.is_scalar:
            raise SourceCompilationError(f"El invariante {self.alias!r} debe producir una IR escalar.")
        return expression


@dataclass(frozen=True, slots=True)
class InvariantRegistry:
    """Registro inmutable; las extensiones no alteran otras corridas."""

    entries: tuple[InvariantSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        if any(not isinstance(item, InvariantSpec) for item in self.entries):
            raise SourceCompilationError("El registro solo admite InvariantSpec.")
        if len(set(self.aliases)) != len(self.entries):
            raise SourceCompilationError("El registro contiene alias duplicados.")

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(item.alias for item in self.entries)

    def get(self, alias: str) -> InvariantSpec:
        for entry in self.entries:
            if entry.alias == alias:
                return entry
        raise SourceCompilationError(f"Invariante no registrado: {alias!r}.")

    def with_invariant(self, entry: InvariantSpec) -> "InvariantRegistry":
        """Devuelve un registro ampliado y rechaza reemplazos implícitos."""
        return InvariantRegistry(self.entries + (entry,))

    def expand(self, alias: str, builder: ModelBuilder) -> Expr:
        return self.get(alias).expand(builder)


DEFAULT_INVARIANTS = InvariantRegistry((
    InvariantSpec("R", ModelBuilder.ricci_scalar, "Escalar de Ricci"),
    InvariantSpec("X", ModelBuilder.kinetic_scalar, "g^ab u_a u_b; sin factor -1/2"),
    InvariantSpec("RicciUU", ModelBuilder.ricci_uu, "R_ab u^a u^b"),
    InvariantSpec("RicciSq", ModelBuilder.ricci_squared, "R_ab R^ab"),
    InvariantSpec("RiemannSq", ModelBuilder.riemann_squared, "R_abcd R^abcd"),
    InvariantSpec("phi", lambda builder: builder.phi, "Campo escalar"),
))
