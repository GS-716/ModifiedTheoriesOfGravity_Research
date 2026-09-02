"""Read-only presentation of existing IR; never input to a calculation/backend.

Scalar polynomial operations use SymPy with tensor contraction blocks and unsafe
powers as opaque atoms. This is not a tensor simplifier: mathematical tensor
contractions belong to the canonical backend, not this presentation layer.
Only hygienic renaming of dummy indices is performed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import product
import json
import re
from typing import TYPE_CHECKING

import sympy as sp

from .indices import canonicalize_dummy_indices
from .ir import (
    Add, CovariantDerivative, Expr, Function, FunctionDerivative, Index, Mul,
    Number, Power, Scalar, Tensor, Variance, Variation, add, mul, walk,
)
from .model import ModelSpec

if TYPE_CHECKING:
    from .components import ComponentEvaluation, SympyComponentBackend
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


@dataclass(frozen=True, slots=True)
class CompactProjection:
    status: str
    reason: str
    free_indices: tuple[Index, ...] = ()
    dimension: int | None = None
    components: tuple[tuple[tuple[int, ...], DisplayExpression], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "free_indices", tuple(self.free_indices))
        object.__setattr__(
            self,
            "components",
            tuple((tuple(position), value) for position, value in self.components),
        )
        if self.status not in {"completed", "symbolic", "unavailable"}:
            raise ValueError(f"Estado de proyección compacta inválido: {self.status!r}.")
        if self.status == "completed" and self.dimension is None:
            raise ValueError("Una proyección compacta completada requiere dimensión.")
        if self.status != "completed" and self.components:
            raise ValueError("Una proyección compacta no completada no contiene componentes.")

    def to_data(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "free_indices": [item.to_data() for item in self.free_indices],
            "dimension": self.dimension,
            "components": [
                {
                    "position": list(position),
                    **value.to_data(),
                    "expanded_expression": value.canonical.to_data(),
                }
                for position, value in self.components
            ],
        }


@dataclass(frozen=True, slots=True)
class CompactBlock:
    key: str
    label_latex: str
    expression: DisplayExpression
    source_keys: tuple[str, ...]
    projection: CompactProjection

    def to_data(self) -> dict:
        return {
            "key": self.key,
            "label_latex": self.label_latex,
            "source_keys": list(self.source_keys),
            "compact": self.expression.to_data(),
            "expanded_expression": self.expression.canonical.to_data(),
            "projection": self.projection.to_data(),
        }


@dataclass(frozen=True, slots=True)
class CompactDecomposition:
    key: str
    label_latex: str
    formula_latex: str
    expression: DisplayExpression
    blocks: tuple[CompactBlock, ...]
    reconstruction_status: str
    reconstruction_reason: str
    projection: CompactProjection
    projection_reconstruction_status: str
    projection_reconstruction_reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if self.reconstruction_status not in {"verified", "symbolic"}:
            raise ValueError("Estado de reconstrucción compacta inválido.")
        if self.projection_reconstruction_status not in {
            "verified", "symbolic", "unavailable"
        }:
            raise ValueError("Estado de reconstrucción proyectada inválido.")

    def to_data(self) -> dict:
        return {
            "key": self.key,
            "label_latex": self.label_latex,
            "formula_latex": self.formula_latex,
            "compact": self.expression.to_data(),
            "expanded_expression": self.expression.canonical.to_data(),
            "blocks": [item.to_data() for item in self.blocks],
            "reconstruction": {
                "status": self.reconstruction_status,
                "reason": self.reconstruction_reason,
            },
            "projection": self.projection.to_data(),
            "projection_reconstruction": {
                "status": self.projection_reconstruction_status,
                "reason": self.projection_reconstruction_reason,
            },
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


def _compact_projection(
    builder: PresentationBuilder,
    evaluation: ComponentEvaluation | None,
    *,
    assumptions: tuple[str, ...],
    reason: str,
    unavailable_status: str = "symbolic",
) -> CompactProjection:
    if evaluation is None:
        return CompactProjection(unavailable_status, reason)
    values = evaluation.values
    if not evaluation.free_indices:
        values = (((), evaluation.scalar),)
    return CompactProjection(
        "completed",
        reason,
        evaluation.free_indices,
        evaluation.dimension,
        tuple(
            (
                position,
                builder.expression(expression, assumptions=assumptions),
            )
            for position, expression in values
        ),
    )


def _evaluate_compact_block(
    builder: PresentationBuilder,
    expression: Expr,
    component_backend: SympyComponentBackend | None,
    *,
    assumptions: tuple[str, ...],
    ansatz_name: str | None,
    expected_free_indices: tuple[Index, ...] | None = None,
) -> CompactProjection:
    if component_backend is None:
        return CompactProjection(
            "symbolic",
            (
                "La expresión abstracta se conserva: la geometría completa del ansatz "
                "no forma parte de RunPackage y no se proporcionó un backend de componentes."
            ),
        )
    try:
        evaluation = component_backend.evaluate(expression)
        if (
            expected_free_indices
            and not evaluation.free_indices
            and expression == Number(0)
        ):
            from .components import ComponentEvaluation
            evaluation = ComponentEvaluation(
                expected_free_indices,
                evaluation.dimension,
                (),
            )
    except Exception as error:
        return CompactProjection(
            "unavailable",
            f"El backend no pudo proyectar este bloque: {type(error).__name__}: {error}",
        )
    return _compact_projection(
        builder,
        evaluation,
        assumptions=assumptions,
        reason=f"Proyectada mediante el ansatz {ansatz_name or 'sin nombre'}.",
    )


def _build_compact_decompositions(
    package: RunPackage,
    builder: PresentationBuilder,
    *,
    projected_assumptions: tuple[str, ...],
    component_backend: SympyComponentBackend | None,
) -> tuple[CompactDecomposition, ...]:
    """Construye una vista adicional desde objetos ya calculados; no muta la corrida."""

    from .backends.structural import StructuralTensorBackend
    from .components import ComponentEvaluation
    from .euler import curvature_algebraic_metric_term
    from .variational import VariationalContext

    context = VariationalContext.from_model(package.model)
    backend = StructuralTensorBackend.from_model(package.model)
    space = package.model.symbols.index_space
    a, b = (Index(name, Variance.DOWN, space) for name in ("a", "b"))

    metric_momentum = package.momenta.metric
    curvature_algebraic = curvature_algebraic_metric_term(
        package.momenta.curvature,
        context,
    )
    volume = mul(
        Number(-1, 2),
        Tensor(package.model.symbols.metric, (a, b)),
        package.lagrangian,
    )
    derivative = package.euler.curvature_derivative_metric_term
    metric_reconstruction = add(
        metric_momentum,
        curvature_algebraic,
        volume,
        derivative,
    )
    metric_residual = backend.simplify(
        add(metric_reconstruction, mul(-1, package.euler.metric_euler))
    )

    scalar_force = package.momenta.scalar
    scalar_divergence = backend.simplify(
        add(package.euler.scalar_euler, mul(-1, scalar_force))
    )
    scalar_reconstruction = add(scalar_force, scalar_divergence)
    scalar_residual = backend.simplify(
        add(scalar_reconstruction, mul(-1, package.euler.scalar_euler))
    )

    def existing_projection(key: str) -> ComponentEvaluation | None:
        if package.projected is None:
            return None
        return package.projected.quantity(key).components

    ansatz_name = None if package.projected is None else package.projected.ansatz_name
    metric_target_projection = _compact_projection(
        builder,
        existing_projection("metric_euler"),
        assumptions=projected_assumptions,
        reason="Reutilizada desde la proyección existente de E_ab.",
    )
    scalar_target_projection = _compact_projection(
        builder,
        existing_projection("scalar_euler"),
        assumptions=projected_assumptions,
        reason="Reutilizada desde la proyección existente de E_phi.",
    )
    curvature_target_projection = _compact_projection(
        builder,
        existing_projection("curvature_momentum"),
        assumptions=projected_assumptions,
        reason="Reutilizada desde la proyección existente de P^{abcd}.",
    )

    metric_projection = existing_projection("metric_momentum")
    scalar_force_projection = existing_projection("scalar_derivative")
    scalar_euler_projection = existing_projection("scalar_euler")
    scalar_divergence_projection: ComponentEvaluation | None = None
    if (
        scalar_force_projection is not None
        and scalar_euler_projection is not None
        and not scalar_force_projection.free_indices
        and not scalar_euler_projection.free_indices
        and scalar_force_projection.dimension == scalar_euler_projection.dimension
    ):
        value = backend.simplify(
            add(scalar_euler_projection.scalar, mul(-1, scalar_force_projection.scalar))
        )
        scalar_divergence_projection = ComponentEvaluation(
            (),
            scalar_euler_projection.dimension,
            () if value == Number(0) else (((), value),),
        )

    metric_blocks = (
        CompactBlock(
            "metric_momentum",
            r"M_{ab}",
            builder.expression(metric_momentum),
            ("metric_momentum",),
            _compact_projection(
                builder,
                metric_projection,
                assumptions=projected_assumptions,
                reason="Reutilizada desde la proyección existente de M_ab.",
            ),
        ),
        CompactBlock(
            "curvature_algebraic",
            r"-\mathcal{R}_{(ab)}[P]",
            builder.expression(curvature_algebraic),
            ("curvature_momentum", "riemann_tensor"),
            _evaluate_compact_block(
                builder,
                curvature_algebraic,
                component_backend,
                assumptions=projected_assumptions,
                ansatz_name=ansatz_name,
                expected_free_indices=(a, b),
            ),
        ),
        CompactBlock(
            "volume_term",
            r"-\frac{1}{2}g_{ab}L",
            builder.expression(volume),
            ("lagrangian",),
            _evaluate_compact_block(
                builder,
                volume,
                component_backend,
                assumptions=projected_assumptions,
                ansatz_name=ansatz_name,
                expected_free_indices=(a, b),
            ),
        ),
        CompactBlock(
            "curvature_derivative",
            r"E_{ab}^{(\nabla\nabla P)}",
            builder.expression(derivative),
            ("curvature_momentum", "nabla_nabla_P"),
            _evaluate_compact_block(
                builder,
                derivative,
                component_backend,
                assumptions=projected_assumptions,
                ansatz_name=ansatz_name,
                expected_free_indices=(a, b),
            ),
        ),
    )
    scalar_blocks = (
        CompactBlock(
            "scalar_force",
            r"F_{\phi}",
            builder.expression(scalar_force),
            ("scalar_derivative",),
            _compact_projection(
                builder,
                scalar_force_projection,
                assumptions=projected_assumptions,
                reason="Reutilizada desde la proyección existente de F_phi.",
            ),
        ),
        CompactBlock(
            "scalar_current_divergence",
            r"-\nabla_a J^a",
            builder.expression(scalar_divergence),
            ("scalar_gradient_momentum", "scalar_euler"),
            _compact_projection(
                builder,
                scalar_divergence_projection,
                assumptions=projected_assumptions,
                reason=(
                    "Reconstruida de las proyecciones existentes E_phi y F_phi."
                    if scalar_divergence_projection is not None
                    else "No se dispuso simultáneamente de las proyecciones escalares E_phi y F_phi."
                ),
            ),
        ),
    )
    curvature_blocks = (
        CompactBlock(
            "curvature_momentum",
            r"P^{abcd}",
            builder.expression(package.momenta.curvature),
            ("curvature_momentum",),
            curvature_target_projection,
        ),
    )

    def verify_projected_reconstruction(
        target: CompactProjection,
        blocks: tuple[CompactBlock, ...],
    ) -> tuple[str, str]:
        if target.status != "completed":
            return (
                target.status,
                "La cantidad objetivo no dispone de una proyección completa.",
            )
        incomplete = tuple(
            block.key for block in blocks if block.projection.status != "completed"
        )
        if incomplete:
            return (
                "symbolic",
                "No se compararon componentes porque faltan proyecciones de: "
                + ", ".join(incomplete)
                + ".",
            )
        target_indices = target.free_indices
        if any(
            block.projection.dimension != target.dimension
            or set(block.projection.free_indices) != set(target_indices)
            for block in blocks
        ):
            return (
                "unavailable",
                "Las proyecciones no comparten dimensión y firma de índices libres.",
            )

        def component_at(
            projection: CompactProjection,
            target_position: tuple[int, ...],
        ) -> Expr:
            coordinates = dict(zip(target_indices, target_position, strict=True))
            local_position = tuple(
                coordinates[index] for index in projection.free_indices
            )
            return dict(projection.components).get(local_position, DisplayExpression(
                Number(0), Number(0), "unchanged"
            )).canonical

        dimension = target.dimension
        assert dimension is not None
        unresolved_reason = ""
        for position in product(range(dimension), repeat=len(target_indices)):
            residual = backend.simplify(
                add(
                    *(component_at(block.projection, position) for block in blocks),
                    mul(-1, component_at(target, position)),
                )
            )
            if residual != Number(0):
                try:
                    from .components import ir_scalar_to_sympy
                    if sp.simplify(ir_scalar_to_sympy(residual)) == 0:
                        continue
                except Exception as error:
                    unresolved_reason = f" ({type(error).__name__}: {error})"
                return (
                    "symbolic",
                    "El álgebra escalar segura no redujo a cero todos los residuales "
                    f"por componente{unresolved_reason}.",
                )
        return (
            "verified",
            "La suma de los bloques coincide componente a componente con la proyección objetivo.",
        )

    metric_projection_status, metric_projection_reason = (
        verify_projected_reconstruction(metric_target_projection, metric_blocks)
    )
    scalar_projection_status, scalar_projection_reason = (
        verify_projected_reconstruction(scalar_target_projection, scalar_blocks)
    )
    curvature_projection_status, curvature_projection_reason = (
        verify_projected_reconstruction(curvature_target_projection, curvature_blocks)
    )

    return (
        CompactDecomposition(
            "metric_euler",
            r"E_{ab}",
            (
                r"E_{ab}=M_{ab}-\mathcal{R}_{(ab)}[P]"
                r"-\frac{1}{2}g_{ab}L-2\nabla_m\nabla_nP^{mn}{}_{ab}"
            ),
            builder.expression(package.euler.metric_euler),
            metric_blocks,
            "verified" if metric_residual == Number(0) else "symbolic",
            (
                "Los cuatro bloques reconstruyen exactamente E_ab en la IR canónica."
                if metric_residual == Number(0)
                else "El backend conservador no redujo a cero el residual de reconstrucción."
            ),
            metric_target_projection,
            metric_projection_status,
            metric_projection_reason,
        ),
        CompactDecomposition(
            "scalar_euler",
            r"E_{\phi}",
            r"E_{\phi}=F_{\phi}-\nabla_aJ^a",
            builder.expression(package.euler.scalar_euler),
            scalar_blocks,
            "verified" if scalar_residual == Number(0) else "symbolic",
            (
                "Los dos bloques reconstruyen exactamente E_phi en la IR canónica."
                if scalar_residual == Number(0)
                else "El backend conservador no redujo a cero el residual de reconstrucción."
            ),
            scalar_target_projection,
            scalar_projection_status,
            scalar_projection_reason,
        ),
        CompactDecomposition(
            "curvature_momentum",
            r"P^{abcd}",
            r"P^{abcd}=\partial L/\partial R_{abcd}",
            builder.expression(package.momenta.curvature),
            curvature_blocks,
            "verified",
            "La forma compacta y la expandida referencian el mismo objeto P^{abcd}.",
            curvature_target_projection,
            curvature_projection_status,
            curvature_projection_reason,
        ),
    )


@dataclass(frozen=True, slots=True)
class ReportPresentation:
    run_id: str
    policy: DisplayPolicy
    expressions: tuple[tuple[str, DisplayExpression], ...]
    compact_decompositions: tuple[CompactDecomposition, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "expressions", tuple(self.expressions))
        object.__setattr__(
            self,
            "compact_decompositions",
            tuple(self.compact_decompositions),
        )

    def record(self, key: str) -> DisplayExpression:
        return next(value for name, value in self.expressions if name == key)

    def to_data(self) -> dict:
        return {
            "schema_version": "1.0", "purpose": "presentation_only", "run_id": self.run_id,
            "policy": asdict(self.policy),
            "expressions": {key: value.to_data() for key, value in self.expressions},
            "compact_decompositions": [
                item.to_data() for item in self.compact_decompositions
            ],
        }


def build_presentation(
    package: RunPackage, policy: DisplayPolicy | None = None, *,
    projected_assumptions: tuple[str, ...] = (),
    component_backend: SympyComponentBackend | None = None,
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
    decompositions = (
        ()
        if package.abstract is None
        else _build_compact_decompositions(
            package,
            builder,
            projected_assumptions=projected_assumptions,
            component_backend=component_backend,
        )
    )
    return ReportPresentation(
        package.run_id,
        builder.policy,
        tuple(entries),
        decompositions,
    )
