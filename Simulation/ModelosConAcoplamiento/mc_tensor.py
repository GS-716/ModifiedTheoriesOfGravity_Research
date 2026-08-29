"""Operaciones tensoriales comunes para teorias lineales en la curvatura.

El modulo trabaja sobre ``CoordinateGeometry`` y no conoce ningun caso EQT
particular. Separa la mecanica de indices de la eleccion del lagrangiano.
"""

from __future__ import annotations

from itertools import product

import sympy as sp

from mc_core import latex_expr


def curvature_momentum(geo, coefficient: sp.Matrix) -> dict[tuple[int, int, int, int], sp.Expr]:
    """Construye P^{abcd} cuando el sector de curvatura es C^{ab} R_ab."""
    gi, n = geo.g_inv, geo.n
    return {
        (a, b, c, d): sp.simplify(sp.Rational(1, 4) * (
            coefficient[a, c] * gi[b, d]
            - coefficient[a, d] * gi[b, c]
            - coefficient[b, c] * gi[a, d]
            + coefficient[b, d] * gi[a, c]
        ))
        for a, b, c, d in product(range(n), repeat=4)
    }


def independent_rank4(momentum, n: int) -> dict[tuple[int, int, int, int], sp.Expr]:
    """Elige representantes no nulos usando las simetrias de Riemann."""
    result = {}
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    for index, (a, b) in enumerate(pairs):
        for c, d in pairs[index:]:
            value = sp.simplify(momentum[a, b, c, d])
            if value != 0:
                result[(a, b, c, d)] = value
    return result


def lower_rank4(momentum, geo) -> dict[tuple[int, int, int, int], sp.Expr]:
    """Baja los cuatro indices de un tensor contravariante de rango cuatro."""
    n, g = geo.n, geo.g
    lowered = {}
    for a, b, c, d in product(range(n), repeat=4):
        value = sp.S.Zero
        for i, j, k, l in product(range(n), repeat=4):
            value += g[a, i] * g[b, j] * g[c, k] * g[d, l] * momentum[i, j, k, l]
        lowered[a, b, c, d] = sp.simplify(value)
    return lowered


def generalized_ricci(momentum, geo) -> sp.Matrix:
    """Calcula Rcal_ab=P_a{}^{cde} R_bcde por contraccion directa."""
    n, g, riem = geo.n, geo.g, geo.Riemann_down
    return sp.Matrix(n, n, lambda a, b: sp.simplify(sum(
        g[a, i] * momentum[i, c, d, e] * riem[b][c][d][e]
        for i, c, d, e in product(range(n), repeat=4)
    )))


def double_divergence(momentum, geo) -> sp.Matrix:
    """Calcula nabla^m nabla^n P_(a|mn|b) con indices covariantes."""
    n, gi, gamma, coordinates = geo.n, geo.g_inv, geo.Gamma, geo.x
    lowered = lower_rank4(momentum, geo)
    tensor = {
        (a, m, q, b): sp.simplify(sp.Rational(1, 2) * (
            lowered[a, m, q, b] + lowered[b, m, q, a]
        ))
        for a, m, q, b in product(range(n), repeat=4)
    }

    def covariant_rank4(derivative, a, m, q, b):
        indices = (a, m, q, b)
        value = sp.diff(tensor[indices], coordinates[derivative])
        for position in range(4):
            for shifted_index in range(n):
                shifted = list(indices)
                shifted[position] = shifted_index
                value -= (
                    gamma[shifted_index][derivative][indices[position]]
                    * tensor[tuple(shifted)]
                )
        return sp.simplify(value)

    first = {}
    for a, m, b in product(range(n), repeat=3):
        first[a, m, b] = sp.simplify(sum(
            gi[q, derivative] * covariant_rank4(derivative, a, m, q, b)
            for q, derivative in product(range(n), repeat=2)
        ))

    def covariant_rank3(derivative, a, m, b):
        indices = (a, m, b)
        value = sp.diff(first[indices], coordinates[derivative])
        for position in range(3):
            for shifted_index in range(n):
                shifted = list(indices)
                shifted[position] = shifted_index
                value -= (
                    gamma[shifted_index][derivative][indices[position]]
                    * first[tuple(shifted)]
                )
        return sp.simplify(value)

    return sp.Matrix(n, n, lambda a, b: sp.simplify(sum(
        gi[m, derivative] * covariant_rank3(derivative, a, m, b)
        for m, derivative in product(range(n), repeat=2)
    )))


def vector_divergence(vector: sp.Matrix, geo) -> sp.Expr:
    """Divergencia de un vector contravariante."""
    return sp.simplify(sum(
        sp.diff(vector[a], geo.x[a])
        + sum(geo.Gamma[a][a][b] * vector[b] for b in range(geo.n))
        for a in range(geo.n)
    ))


def symmetric_current_gradient(current_lower: sp.Matrix, gradient: sp.Matrix) -> sp.Matrix:
    """Devuelve (1/2) J_(a u_b), incluyendo el peso de simetrizacion."""
    n = gradient.rows
    return sp.Matrix(n, n, lambda a, b: sp.simplify(
        sp.Rational(1, 4) * (
            current_lower[a] * gradient[b] + current_lower[b] * gradient[a]
        )
    ))


def momentum_latex(momentum, names, label: str) -> str:
    """Formatea representantes independientes de un tensor tipo Riemann."""
    terms = [
        rf"{label}^{{{{{names[a]}}}{{{names[b]}}}{{{names[c]}}}{{{names[d]}}}}}={latex_expr(value)}"
        for (a, b, c, d), value in momentum.items()
    ]
    return r"\begin{aligned}" + r",\\[2pt]".join(r"&" + term for term in terms) + r"\end{aligned}"


def diagonal_tensor_latex(tensor: sp.Matrix, names, label: str) -> str:
    """Presenta un tensor diagonal por componentes para conservar legibilidad."""
    terms = [
        rf"{label}_{{{names[index]}{names[index]}}}={latex_expr(tensor[index, index])}"
        for index in range(tensor.rows)
    ]
    return r"\begin{aligned}" + r",\\[4pt]".join(r"&" + term for term in terms) + r"\end{aligned}"
