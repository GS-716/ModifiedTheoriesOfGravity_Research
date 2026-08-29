"""Corrientes de Noether y potenciales de carga de Iyer-Wald."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .differential import DifferentialContext, covariant_derivative, divergence
from .errors import TensorAlgebraError
from .euler import EulerLagrangeResult
from .indices import index_key, rename_free_indices
from .ir import Expr, Index, Number, Scalar, Tensor, Variance, add, expr_from_data, infer_free_indices, mul
from .transform import antisymmetrize, symmetrize
from .variational import LagrangianMomenta, VariationalContext


@dataclass(frozen=True, slots=True)
class DiffeomorphismVariation:
    """Variaciones de los campos inducidas por un generador xi^a."""

    inverse_metric: Expr
    scalar: Expr

    def to_data(self) -> dict[str, Any]:
        return {
            "inverse_metric": self.inverse_metric.to_data(),
            "scalar": self.scalar.to_data(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "DiffeomorphismVariation":
        return cls(
            inverse_metric=expr_from_data(data["inverse_metric"]),
            scalar=expr_from_data(data["scalar"]),
        )


@dataclass(frozen=True, slots=True)
class NoetherWaldResult:
    """Resultado fuera de capa para difeomorfismos y carga de Wald."""

    diffeomorphism: DiffeomorphismVariation
    boundary_metric: Expr
    boundary_scalar: Expr
    boundary_total: Expr
    noether_current: Expr
    constraint_current: Expr
    charge_potential: Expr
    charge_divergence: Expr
    decomposition_residual: Expr
    noether_identity: Expr

    def to_data(self) -> dict[str, Any]:
        return {
            "diffeomorphism_variation": self.diffeomorphism.to_data(),
            "diffeomorphism_boundary_metric": self.boundary_metric.to_data(),
            "diffeomorphism_boundary_scalar": self.boundary_scalar.to_data(),
            "diffeomorphism_boundary_total": self.boundary_total.to_data(),
            "noether_current": self.noether_current.to_data(),
            "constraint_current": self.constraint_current.to_data(),
            "charge_potential": self.charge_potential.to_data(),
            "charge_divergence": self.charge_divergence.to_data(),
            "decomposition_residual": self.decomposition_residual.to_data(),
            "noether_identity": self.noether_identity.to_data(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "NoetherWaldResult":
        return cls(
            diffeomorphism=DiffeomorphismVariation.from_data(
                data["diffeomorphism_variation"]
            ),
            boundary_metric=expr_from_data(data["diffeomorphism_boundary_metric"]),
            boundary_scalar=expr_from_data(data["diffeomorphism_boundary_scalar"]),
            boundary_total=expr_from_data(data["diffeomorphism_boundary_total"]),
            noether_current=expr_from_data(data["noether_current"]),
            constraint_current=expr_from_data(data["constraint_current"]),
            charge_potential=expr_from_data(data["charge_potential"]),
            charge_divergence=expr_from_data(data["charge_divergence"]),
            decomposition_residual=expr_from_data(data["decomposition_residual"]),
            noether_identity=expr_from_data(data["noether_identity"]),
        )


def _is_zero(expr: Expr) -> bool:
    return isinstance(expr, Number) and expr.value == 0


def _fresh_name(expressions: tuple[Expr, ...], indices: tuple[Index, ...], prefix: str) -> str:
    occupied = {(item.space, item.name) for expr in expressions for item in infer_free_indices(expr)}
    occupied.update((item.space, item.name) for item in indices)
    space = indices[0].space if indices else "M"
    counter = 0
    while (space, f"{prefix}{counter}") in occupied:
        counter += 1
    return f"{prefix}{counter}"


def _relabel(
    expr: Expr,
    source_names: tuple[str, ...],
    target_indices: tuple[Index, ...],
) -> Expr:
    if _is_zero(expr):
        return expr
    if len(source_names) != len(target_indices):
        raise TensorAlgebraError("La relabelización de Noether recibió rangos incompatibles.")
    free = infer_free_indices(expr)
    by_key = {(item.space, item.name): item for item in free}
    if len(free) != len(source_names):
        raise TensorAlgebraError("El objeto de Noether no conserva el rango esperado.")
    mapping: dict[tuple[str, str], str] = {}
    for source, target in zip(source_names, target_indices, strict=True):
        key = (target.space, source)
        actual = by_key.get(key)
        if actual is None or actual.variance is not target.variance:
            raise TensorAlgebraError(
                f"El índice {source} no coincide con el contrato de Noether."
            )
        mapping[index_key(actual)] = target.name
    return rename_free_indices(expr, mapping)


def diffeomorphism_inverse_metric_variation(
    first: Index,
    second: Index,
    variational_context: VariationalContext | None = None,
    differential_context: DifferentialContext | None = None,
) -> Expr:
    """Construye delta_xi g^{ab}=-2 nabla^{(a}xi^{b)}."""

    variational_context = variational_context or VariationalContext()
    differential_context = differential_context or DifferentialContext()
    if first.variance is not Variance.UP or second.variance is not Variance.UP:
        raise TensorAlgebraError("La variación difeomorfa de g^{ab} requiere índices superiores.")
    if first.space != second.space:
        raise TensorAlgebraError("Los índices métricos deben compartir espacio.")
    dummy_name = _fresh_name((), (first, second), "xg")
    dummy_up = Index(dummy_name, Variance.UP, first.space)
    dummy_down = dummy_up.flipped()
    first_term = mul(
        Tensor(variational_context.metric_name, (first, dummy_up)),
        covariant_derivative(
            Tensor(differential_context.lie_vector_name, (second,)),
            dummy_down,
            differential_context,
        ),
    )
    second_term = mul(
        Tensor(variational_context.metric_name, (second, dummy_up)),
        covariant_derivative(
            Tensor(differential_context.lie_vector_name, (first,)),
            dummy_down,
            differential_context,
        ),
    )
    return symmetrize(mul(-1, add(first_term, second_term)), (first, second))


def diffeomorphism_scalar_variation(
    variational_context: VariationalContext | None = None,
    differential_context: DifferentialContext | None = None,
) -> Expr:
    """Construye delta_xi phi=xi^a nabla_a phi."""

    variational_context = variational_context or VariationalContext()
    differential_context = differential_context or DifferentialContext()
    index = Index("xphi", Variance.UP, variational_context.index_space)
    return mul(
        Tensor(differential_context.lie_vector_name, (index,)),
        Tensor(variational_context.scalar_gradient_name, (index.flipped(),)),
    )


def diffeomorphism_variation(
    variational_context: VariationalContext | None = None,
    differential_context: DifferentialContext | None = None,
) -> DiffeomorphismVariation:
    variational_context = variational_context or VariationalContext()
    space = variational_context.index_space
    first, second = (Index(name, Variance.UP, space) for name in ("a", "b"))
    return DiffeomorphismVariation(
        inverse_metric=diffeomorphism_inverse_metric_variation(
            first,
            second,
            variational_context,
            differential_context,
        ),
        scalar=diffeomorphism_scalar_variation(
            variational_context,
            differential_context,
        ),
    )


def diffeomorphism_boundary_potential(
    momenta: LagrangianMomenta,
    variational_context: VariationalContext | None = None,
    differential_context: DifferentialContext | None = None,
) -> tuple[Expr, Expr, Expr]:
    """Evalúa Theta^a en las variaciones generadas por xi^a."""

    variational_context = variational_context or VariationalContext()
    differential_context = differential_context or DifferentialContext()
    space = variational_context.index_space
    a, b, c, d = (Index(name, Variance.UP, space) for name in ("a", "b", "c", "d"))
    m, n = (Index(name, Variance.DOWN, space) for name in ("m", "n"))

    metric_part: Expr = Number(0)
    if not _is_zero(momenta.curvature):
        momentum = _relabel(momenta.curvature, ("a", "b", "c", "d"), (a, b, c, d))
        delta_metric = diffeomorphism_inverse_metric_variation(
            m.flipped(),
            n.flipped(),
            variational_context,
            differential_context,
        )
        metric_factors = (
            Tensor(variational_context.metric_name, (b.flipped(), m)),
            Tensor(variational_context.metric_name, (c.flipped(), n)),
        )
        derivative_delta = covariant_derivative(delta_metric, d.flipped(), differential_context)
        first = mul(-2, momentum, *metric_factors, derivative_delta)
        momentum_derivative = covariant_derivative(momentum, d.flipped(), differential_context)
        second = (
            Number(0)
            if _is_zero(momentum_derivative)
            else mul(2, momentum_derivative, *metric_factors, delta_metric)
        )
        metric_part = add(first, second)

    scalar_part: Expr = Number(0)
    if not _is_zero(momenta.scalar_gradient):
        current = _relabel(momenta.scalar_gradient, ("a",), (a,))
        scalar_part = mul(
            current,
            diffeomorphism_scalar_variation(variational_context, differential_context),
        )
    return metric_part, scalar_part, add(metric_part, scalar_part)


def noether_charge_potential(
    curvature_momentum: Expr,
    variational_context: VariationalContext | None = None,
    differential_context: DifferentialContext | None = None,
) -> Expr:
    """Construye Q_xi^{ab}=-2P^{abcd}nabla_c xi_d+4xi_d nabla_cP^{abcd}."""

    if _is_zero(curvature_momentum):
        return Number(0)
    variational_context = variational_context or VariationalContext()
    differential_context = differential_context or DifferentialContext()
    space = variational_context.index_space
    a, b, c, d = (Index(name, Variance.UP, space) for name in ("a", "b", "c", "d"))
    momentum = _relabel(curvature_momentum, ("a", "b", "c", "d"), (a, b, c, d))
    xi_down = Tensor(differential_context.lie_vector_name, (d.flipped(),))
    first = mul(
        -2,
        momentum,
        covariant_derivative(xi_down, c.flipped(), differential_context),
    )
    momentum_derivative = covariant_derivative(momentum, c.flipped(), differential_context)
    second = (
        Number(0)
        if _is_zero(momentum_derivative)
        else mul(4, xi_down, momentum_derivative)
    )
    return antisymmetrize(add(first, second), (a, b))


def noether_constraint_current(
    metric_euler: Expr,
    variational_context: VariationalContext | None = None,
    differential_context: DifferentialContext | None = None,
) -> Expr:
    """Construye C_xi^a=2 E^a_b xi^b."""

    if _is_zero(metric_euler):
        return Number(0)
    variational_context = variational_context or VariationalContext()
    differential_context = differential_context or DifferentialContext()
    space = variational_context.index_space
    a = Index("a", Variance.UP, space)
    b, c = (Index(name, Variance.DOWN, space) for name in ("b", "c"))
    euler = _relabel(metric_euler, ("a", "b"), (c, b))
    return mul(
        2,
        Tensor(variational_context.metric_name, (a, c.flipped())),
        euler,
        Tensor(differential_context.lie_vector_name, (b.flipped(),)),
    )


def noether_identity_residual(
    metric_euler: Expr,
    scalar_euler: Expr,
    variational_context: VariationalContext | None = None,
    differential_context: DifferentialContext | None = None,
) -> Expr:
    """Construye I_b=2 nabla^a E_ab+E_phi nabla_b phi."""

    variational_context = variational_context or VariationalContext()
    differential_context = differential_context or DifferentialContext()
    space = variational_context.index_space
    b = Index("b", Variance.DOWN, space)
    terms: list[Expr] = []
    if not _is_zero(metric_euler):
        a = Index("a", Variance.DOWN, space)
        euler = _relabel(metric_euler, ("a", "b"), (a, b))
        terms.append(mul(2, divergence(euler, a, differential_context)))
    if not _is_zero(scalar_euler):
        terms.append(
            mul(
                scalar_euler,
                Tensor(variational_context.scalar_gradient_name, (b,)),
            )
        )
    return add(*terms)


def derive_noether_wald(
    lagrangian: Expr,
    momenta: LagrangianMomenta,
    euler: EulerLagrangeResult,
    variational_context: VariationalContext | None = None,
    differential_context: DifferentialContext | None = None,
) -> NoetherWaldResult:
    """Construye corriente, restricción, carga e identidad difeomorfa."""

    if not lagrangian.is_scalar:
        raise TensorAlgebraError("La construcción de Noether requiere un lagrangiano escalar.")
    variational_context = variational_context or VariationalContext()
    differential_context = differential_context or DifferentialContext()
    space = variational_context.index_space
    diffeomorphism = diffeomorphism_variation(variational_context, differential_context)
    boundary_metric, boundary_scalar, boundary_total = diffeomorphism_boundary_potential(
        momenta,
        variational_context,
        differential_context,
    )
    a = Index("a", Variance.UP, space)
    current = add(
        boundary_total,
        mul(-1, Tensor(differential_context.lie_vector_name, (a,)), lagrangian),
    )
    constraint = noether_constraint_current(
        euler.metric_euler,
        variational_context,
        differential_context,
    )
    charge = noether_charge_potential(
        momenta.curvature,
        variational_context,
        differential_context,
    )
    charge_divergence = (
        Number(0)
        if _is_zero(charge)
        else divergence(charge, Index("b", Variance.UP, space), differential_context)
    )
    decomposition = add(current, mul(-1, constraint), mul(-1, charge_divergence))
    identity = noether_identity_residual(
        euler.metric_euler,
        euler.scalar_euler,
        variational_context,
        differential_context,
    )
    return NoetherWaldResult(
        diffeomorphism=diffeomorphism,
        boundary_metric=boundary_metric,
        boundary_scalar=boundary_scalar,
        boundary_total=boundary_total,
        noether_current=current,
        constraint_current=constraint,
        charge_potential=charge,
        charge_divergence=charge_divergence,
        decomposition_residual=decomposition,
        noether_identity=identity,
    )
