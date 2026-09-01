"""Read-only presentation of existing IR; never input to a calculation/backend.

Scalar polynomial operations use SymPy with tensor contraction blocks and unsafe
powers as opaque atoms. This is not a tensor simplifier: mathematical tensor
contractions belong to the canonical backend, not this presentation layer.
Only hygienic renaming of dummy indices is performed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import TYPE_CHECKING

import sympy as sp

from .indices import canonicalize_dummy_indices
from .ir import (
    Add, CovariantDerivative, Expr, Function, FunctionDerivative, Mul, Number,
    Power, Scalar, Tensor, Variation, add, mul, walk,
)
from .model import ModelSpec

if TYPE_CHECKING:
    from .exporting import RunPackage


@dataclass(frozen=True, slots=True)
class DisplayPolicy:
    factor: bool = True
    collect: bool = True
    together: bool = True
    canonicalize_indices: bool = True
    aggressive: bool = False
    enabled: bool = True
    max_nodes: int = 4000

    def __post_init__(self) -> None:
        for key in ("factor", "collect", "together", "canonicalize_indices", "aggressive", "enabled"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"DisplayPolicy.{key} debe ser booleano.")
        if isinstance(self.max_nodes, bool) or not isinstance(self.max_nodes, int) or self.max_nodes < 1:
            raise ValueError("max_nodes debe ser un entero positivo.")


def _key(expr: Expr) -> str:
    return json.dumps(expr.to_data(), sort_keys=True, separators=(",", ":"))


def _cost(expr: Expr) -> tuple[int, int]:
    return sum(1 for _ in walk(expr)), len(_key(expr))


def _signature(expr: Expr) -> tuple:
    return tuple(sorted((i.space, i.name, i.variance.value) for i in expr.free_indices))


def _domain_guards(expr: Expr, nonzero: dict[str, str]) -> set[Expr]:
    guards = set()
    for node in walk(expr):
        if isinstance(node, Power):
            if isinstance(node.exponent, Number) and node.exponent.denominator == 1:
                if node.exponent.numerator >= 0:
                    continue
                if isinstance(node.base, Number) and node.base.numerator:
                    continue
                if _assumption_name(node.base) not in nonzero:
                    guards.add(node.base)
            else:
                guards.add(node)
    return guards


@dataclass(frozen=True, slots=True)
class DisplayExpression:
    canonical: Expr
    presentation: Expr
    status: str
    operations: tuple[str, ...] = ()
    assumptions_used: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_data(self) -> dict:
        return {
            "canonical_sha256": hashlib.sha256(_key(self.canonical).encode()).hexdigest(),
            "expression": self.presentation.to_data(),
            "status": self.status,
            "operations": list(self.operations),
            "assumptions_used": list(self.assumptions_used),
            "notes": list(self.notes),
        }


def _assumption_name(expr: Expr) -> str | None:
    if isinstance(expr, Scalar):
        return expr.name
    if isinstance(expr, Function) and all(isinstance(a, Scalar) for a in expr.arguments):
        return expr.name + "(" + ",".join(a.name for a in expr.arguments) + ")"
    return None


class _ScalarAlgebra:
    """Reversible atomization. No text parsing, function evaluation or index math."""

    def __init__(self, assumptions: tuple[str, ...]):
        self.forward: dict[Expr, sp.Symbol] = {}
        self.reverse: dict[sp.Symbol, Expr] = {}
        self.protected: set[sp.Symbol] = set()
        self.used: set[str] = set()
        self.nonzero: dict[str, str] = {}
        for original in assumptions:
            normalized = re.sub(r"\s+", "", original)
            match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*(?:\([A-Za-z][A-Za-z0-9_,]*\))?)(!=|>|<)0", normalized)
            if match:
                self.nonzero[match[1]] = original

    def atom(self, expr: Expr) -> sp.Symbol:
        if expr not in self.forward:
            # Sequential symbols are local and cannot collide with user names.
            symbol = sp.Symbol(f"displayatom{len(self.forward):06d}")
            self.forward[expr] = symbol
            self.reverse[symbol] = expr
        return self.forward[expr]

    @staticmethod
    def tensorial(expr: Expr) -> bool:
        return any(isinstance(n, (Tensor, CovariantDerivative, Variation)) for n in walk(expr))

    def encode(self, expr: Expr) -> sp.Expr:
        if isinstance(expr, Number):
            return sp.Rational(expr.numerator, expr.denominator)
        if isinstance(expr, Add):
            return sp.Add(*(self.encode(t) for t in expr.terms))
        if isinstance(expr, Mul):
            # Never split a contraction into independently commuting atoms.
            scalar, tensor = [], []

            def extract(factor: Expr) -> Expr | None:
                if not self.tensorial(factor):
                    scalar.append(factor)
                    return None
                if isinstance(factor, Mul):
                    # Extract only index-free scalar coefficients. Keep each
                    # nested contraction as a block with its original scope.
                    remaining = [value for child in factor.factors
                                 if (value := extract(child)) is not None]
                    return mul(*remaining) if remaining else None
                return factor

            for factor in expr.factors:
                remaining = extract(factor)
                if remaining is not None:
                    tensor.append(remaining)
            encoded = [self.encode(f) for f in scalar]
            if tensor:
                encoded.append(self.atom(mul(*tensor)))
            return sp.Mul(*encoded)
        if isinstance(expr, Power):
            if isinstance(expr.exponent, Number) and expr.exponent.denominator == 1:
                n = expr.exponent.numerator
                if 0 < n <= 100 and not self.tensorial(expr.base):
                    return self.encode(expr.base) ** n
                name = _assumption_name(expr.base)
                if n < 0 and name in self.nonzero:
                    self.used.add(self.nonzero[name])
                    return self.encode(expr.base) ** n
                if n < 0 and isinstance(expr.base, Number) and expr.base.numerator:
                    return self.encode(expr.base) ** n
            symbol = self.atom(expr)
            self.protected.add(symbol)
            return symbol
        # Functions, formal derivatives and other leaves remain uninterpreted.
        return self.atom(expr)

    def decode(self, expr: sp.Expr) -> Expr:
        if expr in self.reverse:
            return self.reverse[expr]
        if expr.is_Rational:
            return Number(int(expr.p), int(expr.q))
        if isinstance(expr, sp.Add):
            return add(*(self.decode(t) for t in expr.args))
        if isinstance(expr, sp.Mul):
            return mul(*(self.decode(f) for f in expr.args))
        if isinstance(expr, sp.Pow):
            base, exponent = self.decode(expr.base), self.decode(expr.exp)
            # Positive integer powers of reciprocal atoms retain the same poles.
            if (isinstance(base, Power) and isinstance(base.exponent, Number)
                    and base.exponent.denominator == 1 and base.exponent.numerator < 0
                    and isinstance(exponent, Number) and exponent.denominator == 1
                    and exponent.numerator > 0):
                return Power(base.base, Number(base.exponent.numerator * exponent.numerator))
            return Power(base, exponent)
        raise ValueError(f"Forma escalar de presentación no soportada: {type(expr).__name__}")


class PresentationBuilder:
    """One export-local cache; no mutation or global state shared with the run."""

    def __init__(self, model: ModelSpec, policy: DisplayPolicy | None = None):
        self.model = model
        self.policy = policy or DisplayPolicy()
        self.assumptions = tuple(model.assumptions) + tuple(
            f"{p.name}!=0" for p in model.parameters
            if any(a in ("nonzero", "positive", "negative") for a in p.assumptions)
        )
        self.cache: dict[tuple[Expr, tuple[str, ...]], DisplayExpression] = {}

    def expression(self, canonical: Expr, *, assumptions: tuple[str, ...] = ()) -> DisplayExpression:
        cache_key = (canonical, tuple(assumptions))
        if cache_key not in self.cache:
            self.cache[cache_key] = self._expression(canonical, tuple(assumptions))
        return self.cache[cache_key]

    def _expression(self, original: Expr, assumptions: tuple[str, ...]) -> DisplayExpression:
        policy = self.policy
        if not policy.enabled:
            return DisplayExpression(original, original, "disabled")
        if _cost(original)[0] > policy.max_nodes:
            return DisplayExpression(original, original, "unchanged", notes=("Límite de nodos de presentación excedido.",))
        current = original
        operations: list[str] = []
        notes: list[str] = []
        index_assumptions: set[str] = set()
        try:
            domain = _ScalarAlgebra(self.assumptions + assumptions).nonzero
            guards = _domain_guards(original, domain)
            if policy.canonicalize_indices and _ScalarAlgebra.tensorial(current):
                # Presentation may rename dummies but must not eliminate an
                # identity or perform independent metric contractions.
                candidate = canonicalize_dummy_indices(current)
                if (candidate != current and _cost(candidate) <= _cost(current)
                        and guards.issubset(_domain_guards(candidate, domain))):
                    if candidate != Number(0) and _signature(candidate) != _signature(original):
                        raise ValueError("La canonización cambió la firma de índices libres.")
                    removed = _domain_guards(current, {}) - _domain_guards(candidate, {})
                    index_assumptions.update(domain[_assumption_name(n)] for n in removed
                                             if _assumption_name(n) in domain)
                    current = candidate
                    operations.append("rename_dummy_indices")
            algebra = _ScalarAlgebra(self.assumptions + assumptions)
            encoded = algebra.encode(current)
            initial = encoded
            attempted = ["combine_like_terms", "deterministic_order", "normalize_signs_and_numbers"]
            choices = [(encoded, tuple(attempted))]
            if policy.together:
                encoded = sp.together(encoded)
                attempted.append("together_safe_fractions")
                choices.append((encoded, tuple(attempted)))
            if policy.collect:
                parameters = [algebra.forward[Scalar(p.name)] for p in self.model.parameters
                              if Scalar(p.name) in algebra.forward]
                if parameters:
                    encoded = sp.collect(encoded, parameters)
                    attempted.append("collect_parameters")
                    choices.append((encoded, tuple(attempted)))
            if policy.factor:
                encoded = sp.factor_terms(encoded)
                attempted.append("factor_common_scalar_terms")
                choices.append((encoded, tuple(attempted)))
                if policy.aggressive:
                    # More expensive polynomial search, NOT weaker domain guards.
                    choices.append((sp.factor(encoded), tuple(attempted + ["factor_polynomial"])))
            selected: tuple[str, ...] = ()
            best = current
            for candidate, candidate_ops in choices:
                if not algebra.protected.issubset(candidate.free_symbols):
                    notes.append("Se evitó eliminar una potencia protegida sin hipótesis explícitas.")
                    continue
                if sp.cancel(candidate - initial) != 0:
                    continue
                decoded = algebra.decode(candidate)
                if not guards.issubset(_domain_guards(decoded, domain)):
                    continue
                if decoded != Number(0) and _signature(decoded) != _signature(original):
                    continue
                if _cost(decoded) < _cost(best):
                    best, selected = decoded, candidate_ops
            if algebra.protected:
                notes.append("Potencias no justificadas conservadas como bloques; sin cancelación de sus factores.")
            operations.extend(selected)
            used = tuple(sorted(index_assumptions | (algebra.used if selected else set())))
            return DisplayExpression(
                original, best, "simplified" if best != original else "unchanged",
                tuple(operations), used, tuple(dict.fromkeys(notes)),
            )
        except Exception as error:
            # Presentation must never cause a successful calculation to disappear.
            return DisplayExpression(original, original, "fallback", notes=(f"Presentación no evaluada: {type(error).__name__}: {error}",))


@dataclass(frozen=True, slots=True)
class ReportPresentation:
    run_id: str
    policy: DisplayPolicy
    expressions: tuple[tuple[str, DisplayExpression], ...]

    def record(self, key: str) -> DisplayExpression:
        return next(value for name, value in self.expressions if name == key)

    def to_data(self) -> dict:
        return {
            "schema_version": "1.0", "purpose": "presentation_only", "run_id": self.run_id,
            "policy": asdict(self.policy),
            "expressions": {key: value.to_data() for key, value in self.expressions},
        }


def build_presentation(
    package: RunPackage, policy: DisplayPolicy | None = None, *,
    projected_assumptions: tuple[str, ...] = (),
) -> ReportPresentation:
    """Build both views from stored quantities, including every sparse component.

    Pass the actual ansatz assumptions when exporting a stored package; a name
    alone is never used to guess geometric conditions.
    """
    builder = PresentationBuilder(package.model, policy)
    entries: list[tuple[str, DisplayExpression]] = []
    if package.abstract is not None:
        for key, expr in package.abstract.expression_items():
            entries.append((f"abstract.{key}", builder.expression(expr)))
        if package.derived is not None:
            entries.append(("abstract.curvature_derivative_metric_term", builder.expression(package.derived.curvature_derivative_metric_term)))
        if package.projected is not None:
            for item in package.projected.quantities:
                if item.components is None:
                    entries.append((f"projected.{item.key}.abstract_fallback", builder.expression(getattr(package.abstract, item.key))))
                elif not item.components.free_indices:
                    entries.append((f"projected.{item.key}.scalar", builder.expression(item.components.scalar, assumptions=projected_assumptions)))
                elif not item.components.values:
                    entries.append((f"projected.{item.key}.zero", builder.expression(Number(0))))
                else:
                    for position, expr in item.components.values:
                        suffix = ",".join(map(str, position))
                        entries.append((f"projected.{item.key}[{suffix}]", builder.expression(expr, assumptions=projected_assumptions)))
    return ReportPresentation(package.run_id, builder.policy, tuple(entries))
