"""Integración por partes y objetos de Euler-Lagrange covariantes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .differential import DifferentialContext, covariant_derivative, divergence
from .errors import TensorAlgebraError
from .indices import index_key, rename_free_indices
from .ir import (
    CovariantDerivative,
    Expr,
    Index,
    Number,
    Scalar,
    Tensor,
    Variance,
    Variation,
    VolumeElement,
    add,
    expr_from_data,
    infer_free_indices,
    mul,
)
from .transform import symmetrize
from .variational import LagrangianMomenta, VariationalContext


@dataclass(frozen=True, slots=True)
class EulerLagrangeResult:
    metric_euler: Expr
    scalar_euler: Expr
    boundary_metric: Expr
    boundary_scalar: Expr
    boundary_total: Expr
    full_variation: Expr
    density_variation: Expr

    def to_data(self) -> dict[str, Any]:
        return {
            "metric_euler": self.metric_euler.to_data(),
            "scalar_euler": self.scalar_euler.to_data(),
            "boundary_potential_metric": self.boundary_metric.to_data(),
            "boundary_potential_scalar": self.boundary_scalar.to_data(),
            "boundary_potential_total": self.boundary_total.to_data(),
            "full_variation": self.full_variation.to_data(),
            "density_variation": self.density_variation.to_data(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "EulerLagrangeResult":
        return cls(
            metric_euler=expr_from_data(data["metric_euler"]),
            scalar_euler=expr_from_data(data["scalar_euler"]),
            boundary_metric=expr_from_data(data["boundary_potential_metric"]),
            boundary_scalar=expr_from_data(data["boundary_potential_scalar"]),
            boundary_total=expr_from_data(data["boundary_potential_total"]),
            full_variation=expr_from_data(data["full_variation"]),
            density_variation=expr_from_data(data["density_variation"]),
        )


def _is_zero(expr: Expr) -> bool:
    return isinstance(expr, Number) and expr.value == 0


def _relabel_momentum(
    expr: Expr,
    source_names: tuple[str, ...],
    target_indices: tuple[Index, ...],
) -> Expr:
    if _is_zero(expr):
        return expr
    if len(source_names) != len(target_indices):
        raise TensorAlgebraError("La relabelización tensorial recibió rangos incompatibles.")
    free = infer_free_indices(expr)
    space = target_indices[0].space
    by_name = {item.name: item for item in free if item.space == space}
    if set(by_name) != set(source_names) or len(free) != len(source_names):
        raise TensorAlgebraError(
            "El momento no conserva los índices canónicos esperados: "
            + ", ".join(source_names)
        )
    for source, target in zip(source_names, target_indices, strict=True):
        if by_name[source].variance is not target.variance:
            raise TensorAlgebraError("La varianza del momento no coincide con su contrato.")
    return rename_free_indices(
        expr,
        {
            index_key(by_name[source]): target.name
            for source, target in zip(source_names, target_indices, strict=True)
        },
    )


def curvature_algebraic_metric_term(
    curvature_momentum: Expr,
    context: VariationalContext | None = None,
) -> Expr:
    """Construye -P_(a^{cde} R_b)cde para la variable g^{ab}."""

    if _is_zero(curvature_momentum):
        return Number(0)
    context = context or VariationalContext()
    space = context.index_space
    a, b = (Index(name, Variance.DOWN, space) for name in ("a", "b"))
    p, c, d, e = (Index(name, Variance.UP, space) for name in ("p", "c", "d", "e"))
    momentum = _relabel_momentum(
        curvature_momentum,
        ("a", "b", "c", "d"),
        (p, c, d, e),
    )
    raw = mul(
        -1,
        Tensor(context.metric_name, (a, p.flipped())),
        momentum,
        Tensor(context.curvature_name, (b, c.flipped(), d.flipped(), e.flipped())),
    )
    return symmetrize(raw, (a, b))


def curvature_derivative_metric_term(
    curvature_momentum: Expr,
    variational_context: VariationalContext | None = None,
    differential_context: DifferentialContext | None = None,
) -> Expr:
    """Construye -2 nabla^c nabla^d P_acdb."""

    if _is_zero(curvature_momentum):
        return Number(0)
    variational_context = variational_context or VariationalContext()
    differential_context = differential_context or DifferentialContext()
    space = variational_context.index_space
    a, b = (Index(name, Variance.DOWN, space) for name in ("a", "b"))
    p, q, r, s = (Index(name, Variance.UP, space) for name in ("p", "q", "r", "s"))
    momentum = _relabel_momentum(
        curvature_momentum,
        ("a", "b", "c", "d"),
        (p, q, r, s),
    )
    inner = covariant_derivative(momentum, r.flipped(), differential_context)
    if _is_zero(inner):
        return Number(0)
    outer = covariant_derivative(inner, q.flipped(), differential_context)
    if _is_zero(outer):
        return Number(0)
    raw = mul(
        -2,
        Tensor(variational_context.metric_name, (a, p.flipped())),
        Tensor(variational_context.metric_name, (b, s.flipped())),
        outer,
    )
    return symmetrize(raw, (a, b))


def metric_euler_expression(
    lagrangian: Expr,
    momenta: LagrangianMomenta,
    variational_context: VariationalContext | None = None,
    differential_context: DifferentialContext | None = None,
) -> Expr:
    """Construye E_ab para la convención de variación respecto a g^{ab}."""

    if not lagrangian.is_scalar:
        raise TensorAlgebraError("E_ab requiere un lagrangiano escalar.")
    variational_context = variational_context or VariationalContext()
    space = variational_context.index_space
    a, b = (Index(name, Variance.DOWN, space) for name in ("a", "b"))
    metric_momentum = _relabel_momentum(momenta.metric, ("a", "b"), (a, b))
    volume_term = mul(
        Number(-1, 2),
        Tensor(variational_context.metric_name, (a, b)),
        lagrangian,
    )
    result = add(
        metric_momentum,
        curvature_algebraic_metric_term(momenta.curvature, variational_context),
        curvature_derivative_metric_term(
            momenta.curvature,
            variational_context,
            differential_context,
        ),
        volume_term,
    )
    if _is_zero(result):
        return result
    return symmetrize(result, (a, b))


def scalar_euler_expression(
    momenta: LagrangianMomenta,
    variational_context: VariationalContext | None = None,
    differential_context: DifferentialContext | None = None,
) -> Expr:
    """Construye E_phi=F_phi-nabla_a J^a."""

    variational_context = variational_context or VariationalContext()
    differential_context = differential_context or DifferentialContext()
    if _is_zero(momenta.scalar_gradient):
        return momenta.scalar
    a = Index("a", Variance.UP, variational_context.index_space)
    current = _relabel_momentum(momenta.scalar_gradient, ("a",), (a,))
    return add(momenta.scalar, mul(-1, divergence(current, a, differential_context)))


def metric_boundary_potential(
    curvature_momentum: Expr,
    variational_context: VariationalContext | None = None,
    differential_context: DifferentialContext | None = None,
) -> Expr:
    """Potencial Theta_g^a expresado en términos de delta g^{mn}."""

    if _is_zero(curvature_momentum):
        return Number(0)
    variational_context = variational_context or VariationalContext()
    differential_context = differential_context or DifferentialContext()
    space = variational_context.index_space
    a, b, c, d = (Index(name, Variance.UP, space) for name in ("a", "b", "c", "d"))
    m, n = (Index(name, Variance.DOWN, space) for name in ("m", "n"))
    momentum = _relabel_momentum(
        curvature_momentum,
        ("a", "b", "c", "d"),
        (a, b, c, d),
    )
    metric_factors = (
        Tensor(variational_context.metric_name, (b.flipped(), m)),
        Tensor(variational_context.metric_name, (c.flipped(), n)),
    )
    inverse_metric_variation = Variation(
        Tensor(variational_context.metric_name, (m.flipped(), n.flipped()))
    )
    derivative_variation = CovariantDerivative(d.flipped(), inverse_metric_variation)
    first = mul(-2, momentum, *metric_factors, derivative_variation)

    momentum_divergence = covariant_derivative(
        momentum,
        d.flipped(),
        differential_context,
    )
    second = (
        Number(0)
        if _is_zero(momentum_divergence)
        else mul(2, momentum_divergence, *metric_factors, inverse_metric_variation)
    )
    return add(first, second)


def scalar_boundary_potential(
    scalar_gradient_momentum: Expr,
    variational_context: VariationalContext | None = None,
) -> Expr:
    """Construye Theta_phi^a=J^a delta phi."""

    if _is_zero(scalar_gradient_momentum):
        return Number(0)
    variational_context = variational_context or VariationalContext()
    a = Index("a", Variance.UP, variational_context.index_space)
    current = _relabel_momentum(scalar_gradient_momentum, ("a",), (a,))
    return mul(current, Variation(Scalar(variational_context.scalar_name)))


def scalar_integration_by_parts_residual(
    momenta: LagrangianMomenta,
    variational_context: VariationalContext | None = None,
    differential_context: DifferentialContext | None = None,
) -> Expr:
    """Residual entre la regla de cadena escalar y su forma integrada."""

    variational_context = variational_context or VariationalContext()
    differential_context = differential_context or DifferentialContext()
    delta_phi = Variation(Scalar(variational_context.scalar_name))
    raw_terms: list[Expr] = [mul(momenta.scalar, delta_phi)]
    if not _is_zero(momenta.scalar_gradient):
        a = Index("a", Variance.UP, variational_context.index_space)
        current = _relabel_momentum(momenta.scalar_gradient, ("a",), (a,))
        raw_terms.append(
            mul(current, CovariantDerivative(a.flipped(), delta_phi))
        )
    raw = add(*raw_terms)
    euler_term = mul(
        scalar_euler_expression(momenta, variational_context, differential_context),
        delta_phi,
    )
    theta = scalar_boundary_potential(momenta.scalar_gradient, variational_context)
    boundary = Number(0)
    if not _is_zero(theta):
        boundary = divergence(
            theta,
            Index("a", Variance.UP, variational_context.index_space),
            differential_context,
        )
    return add(raw, mul(-1, add(euler_term, boundary)))


def derive_euler_lagrange(
    lagrangian: Expr,
    momenta: LagrangianMomenta,
    variational_context: VariationalContext | None = None,
    differential_context: DifferentialContext | None = None,
) -> EulerLagrangeResult:
    """Separa bulk y frontera para una acción sqrt(-g)L."""

    variational_context = variational_context or VariationalContext()
    differential_context = differential_context or DifferentialContext()
    metric_euler = metric_euler_expression(
        lagrangian,
        momenta,
        variational_context,
        differential_context,
    )
    scalar_euler = scalar_euler_expression(
        momenta,
        variational_context,
        differential_context,
    )
    boundary_metric = metric_boundary_potential(
        momenta.curvature,
        variational_context,
        differential_context,
    )
    boundary_scalar = scalar_boundary_potential(
        momenta.scalar_gradient,
        variational_context,
    )
    boundary_total = add(boundary_metric, boundary_scalar)

    bulk_terms: list[Expr] = []
    if not _is_zero(metric_euler):
        a, b = (
            Index(name, Variance.DOWN, variational_context.index_space)
            for name in ("a", "b")
        )
        bulk_terms.append(
            mul(
                metric_euler,
                Variation(
                    Tensor(
                        variational_context.metric_name,
                        (a.flipped(), b.flipped()),
                    )
                ),
            )
        )
    if not _is_zero(scalar_euler):
        bulk_terms.append(
            mul(
                scalar_euler,
                Variation(Scalar(variational_context.scalar_name)),
            )
        )
    if not _is_zero(boundary_total):
        bulk_terms.append(
            divergence(
                boundary_total,
                Index("a", Variance.UP, variational_context.index_space),
                differential_context,
            )
        )
    full_variation = add(*bulk_terms)
    density_variation = mul(VolumeElement(variational_context.metric_name), full_variation)
    return EulerLagrangeResult(
        metric_euler=metric_euler,
        scalar_euler=scalar_euler,
        boundary_metric=boundary_metric,
        boundary_scalar=boundary_scalar,
        boundary_total=boundary_total,
        full_variation=full_variation,
        density_variation=density_variation,
    )
