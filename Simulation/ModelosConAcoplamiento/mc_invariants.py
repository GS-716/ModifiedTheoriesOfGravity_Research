"""Base configurable de invariantes para la familia EQT tridimensional.

Un modelo es una suma finita de las dos torres del articulo de EQT. El usuario
elige los ordenes y, si lo desea, expresiones arbitrarias para sus acoplamientos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import sympy as sp


@dataclass
class EQTModelSpec:
    """Especificacion declarativa de un lagrangiano EQT finito.

    ``alpha[n]`` multiplica -ell^(2(n-1)) X^n para n>=1.
    ``beta[m]`` multiplica ell^(2(m+1)) X^m B_m para m>=0, donde
    B_m=(3+2m) R_ab u^a u^b-X R.
    """

    name: str = "EQT configurable"
    alpha: dict[int, sp.Expr] = field(default_factory=dict)
    beta: dict[int, sp.Expr] = field(default_factory=dict)
    include_einstein: bool = True
    include_ads: bool = True

    def validate(self) -> "EQTModelSpec":
        if not self.include_einstein:
            raise NotImplementedError("La reduccion radial actual requiere el termino de Einstein")
        if any(not isinstance(order, int) or order < 1 for order in self.alpha):
            raise ValueError("Los ordenes alpha deben ser enteros n>=1")
        if any(not isinstance(order, int) or order < 0 for order in self.beta):
            raise ValueError("Los ordenes beta deben ser enteros m>=0")
        return self

    @property
    def alpha_orders(self) -> tuple[int, ...]:
        return tuple(sorted(self.alpha))

    @property
    def beta_orders(self) -> tuple[int, ...]:
        return tuple(sorted(self.beta))


def symbolic_eqt_spec(
    alpha_orders: tuple[int, ...] = (1, 2),
    beta_orders: tuple[int, ...] = (0, 1),
    name: str = "EQT general finito",
) -> EQTModelSpec:
    """Crea una especificacion cuyos acoplamientos son simbolos alpha_n,beta_m."""
    alpha = {order: sp.symbols(f"alpha_{order}", real=True) for order in alpha_orders}
    beta = {order: sp.symbols(f"beta_{order}", real=True) for order in beta_orders}
    return EQTModelSpec(name=name, alpha=alpha, beta=beta).validate()


def alpha_density(order: int, coupling: sp.Expr, ell: sp.Expr, X: sp.Expr) -> sp.Expr:
    """Invariante -alpha_n ell^(2(n-1)) X^n."""
    return -coupling * ell ** (2 * (order - 1)) * X**order


def alpha_current(
    order: int,
    coupling: sp.Expr,
    ell: sp.Expr,
    X: sp.Expr,
    u_up: sp.Matrix,
) -> sp.Matrix:
    """Momento J^a de un invariante de la torre alpha."""
    factor = -2 * order * coupling * ell ** (2 * (order - 1)) * X ** (order - 1)
    return (factor * u_up).applyfunc(sp.simplify)


def beta_density(
    order: int,
    coupling: sp.Expr,
    ell: sp.Expr,
    X: sp.Expr,
    Y: sp.Expr,
    R: sp.Expr,
) -> sp.Expr:
    """Invariante beta_m completo, con Y=R_ab u^a u^b."""
    return sp.simplify(
        coupling * ell ** (2 * (order + 1)) * X**order
        * ((3 + 2 * order) * Y - X * R)
    )


def beta_ricci_coefficient(
    order: int,
    coupling: sp.Expr,
    ell: sp.Expr,
    X: sp.Expr,
    u_up: sp.Matrix,
    metric_inverse: sp.Matrix,
) -> sp.Matrix:
    """Coeficiente C_m^{ab} tal que L_beta_m=C_m^{ab} R_ab."""
    prefactor = coupling * ell ** (2 * (order + 1)) * X**order
    return (prefactor * (
        (3 + 2 * order) * u_up * u_up.T - X * metric_inverse
    )).applyfunc(sp.simplify)


def beta_current(
    order: int,
    coupling: sp.Expr,
    ell: sp.Expr,
    X: sp.Expr,
    Y: sp.Expr,
    R: sp.Expr,
    ricci_upup: sp.Matrix,
    u_cov: sp.Matrix,
    u_up: sp.Matrix,
) -> sp.Matrix:
    """Momento J^a obtenido derivando el invariante beta_m respecto de u_a."""
    scale = 2 * coupling * ell ** (2 * (order + 1))
    result = (3 + 2 * order) * X**order * ricci_upup * u_cov
    result -= (order + 1) * X**order * R * u_up
    if order > 0:
        result += order * (3 + 2 * order) * X ** (order - 1) * Y * u_up
    return (scale * result).applyfunc(sp.simplify)


def eqt_density(
    spec: EQTModelSpec,
    ell: sp.Expr,
    X: sp.Expr,
    Y: sp.Expr,
    R: sp.Expr,
) -> sp.Expr:
    """Compone el lagrangiano de una especificacion valida."""
    spec.validate()
    value = R + (2 / ell**2 if spec.include_ads else 0)
    value += sum(alpha_density(n, coefficient, ell, X) for n, coefficient in spec.alpha.items())
    value += sum(beta_density(m, coefficient, ell, X, Y, R) for m, coefficient in spec.beta.items())
    return sp.simplify(value)


def analytic_branch(
    spec: EQTModelSpec,
    r: sp.Expr,
    ell: sp.Expr,
    p: sp.Expr,
    lam: sp.Expr,
    r0: sp.Expr,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Devuelve N(r), H(r) y f=N/H para una suma EQT finita."""
    numerator = r**2 / ell**2 - lam
    for order, coefficient in spec.alpha.items():
        if order == 1:
            numerator -= coefficient * p**2 * sp.log(r / r0)
        else:
            numerator += (
                coefficient * ell ** (2 * (order - 1)) * p ** (2 * order)
                / (2 * (order - 1) * r ** (2 * (order - 1)))
            )
    denominator = sp.S.One + sum(
        coefficient * (2 * order + 1) * (p * ell / r) ** (2 * (order + 1))
        for order, coefficient in spec.beta.items()
    )
    return sp.factor(numerator), sp.factor(denominator), sp.factor(numerator / denominator)
