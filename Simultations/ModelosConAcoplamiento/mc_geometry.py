"""Geometria coordenada para una metrica diagonal de dimension arbitraria pequena."""

from __future__ import annotations

from itertools import product

import sympy as sp


class CoordinateGeometry:
    """Calcula Gamma, Riemann, Ricci, R y Einstein desde una matriz metrica."""

    def __init__(self, coordinates: tuple[sp.Symbol, ...], metric: sp.Matrix):
        self.x = coordinates
        self.g = sp.Matrix(metric)
        self.n = len(coordinates)
        self.g_inv = sp.simplify(self.g.inv())
        self.Gamma = self._christoffel()
        self.Riemann_up = self._riemann_up()
        self.Riemann_down = self._riemann_down()
        self.Ricci = self._ricci()
        self.Rscalar = sp.simplify(sum(
            self.g_inv[a, b] * self.Ricci[a, b]
            for a, b in product(range(self.n), repeat=2)
        ))
        self.Einstein = self._einstein()

    def _christoffel(self):
        n, x, g, gi = self.n, self.x, self.g, self.g_inv
        return [[[sp.simplify(sp.Rational(1, 2) * sum(
            gi[rho, sigma] * (
                sp.diff(g[sigma, nu], x[mu])
                + sp.diff(g[sigma, mu], x[nu])
                - sp.diff(g[mu, nu], x[sigma])
            ) for sigma in range(n)
        )) for nu in range(n)] for mu in range(n)] for rho in range(n)]

    def _riemann_up(self):
        n, x, G = self.n, self.x, self.Gamma
        # Convencion: R^rho_{ sigma mu nu} = d_mu Gamma^rho_{nu sigma} - ...
        return [[[[sp.simplify(
            sp.diff(G[rho][nu][sigma], x[mu])
            - sp.diff(G[rho][mu][sigma], x[nu])
            + sum(
                G[rho][mu][lam] * G[lam][nu][sigma]
                - G[rho][nu][lam] * G[lam][mu][sigma]
                for lam in range(n)
            )
        ) for nu in range(n)] for mu in range(n)] for sigma in range(n)] for rho in range(n)]

    def _riemann_down(self):
        n = self.n
        return [[[[sp.simplify(sum(
            self.g[a, rho] * self.Riemann_up[rho][b][c][d]
            for rho in range(n)
        )) for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]

    def _ricci(self):
        n = self.n
        return sp.Matrix(n, n, lambda a, b: sp.simplify(sum(
            self.Riemann_up[rho][a][rho][b] for rho in range(n)
        )))

    def _einstein(self):
        return sp.Matrix(self.n, self.n, lambda a, b: sp.simplify(
            self.Ricci[a, b] - sp.Rational(1, 2) * self.g[a, b] * self.Rscalar
        ))

    def divergence_cov2(self, tensor: sp.Matrix) -> sp.Matrix:
        """Devuelve nabla^a T_ab para un tensor covariante de rango dos."""
        n, x, gi, G = self.n, self.x, self.g_inv, self.Gamma
        result = []
        for b in range(n):
            value = sp.S.Zero
            for a, c in product(range(n), repeat=2):
                cov = sp.diff(tensor[a, b], x[c])
                cov -= sum(G[d][c][a] * tensor[d, b] for d in range(n))
                cov -= sum(G[d][c][b] * tensor[a, d] for d in range(n))
                value += gi[a, c] * cov
            result.append(sp.simplify(value))
        return sp.Matrix(result)

    def scalar_gradient_cov(self, scalar: sp.Expr) -> sp.Matrix:
        """Gradiente covariante de un escalar (coincide con la derivada parcial)."""
        return sp.Matrix([sp.diff(scalar, coordinate) for coordinate in self.x])

    def scalar_laplacian(self, scalar: sp.Expr) -> sp.Expr:
        """Calcula Box(scalar)=g^{ab} nabla_a nabla_b scalar."""
        gradient = self.scalar_gradient_cov(scalar)
        result = sp.S.Zero
        for a in range(self.n):
            for b in range(self.n):
                second = sp.diff(gradient[b], self.x[a])
                second -= sum(
                    self.Gamma[c][a][b] * gradient[c]
                    for c in range(self.n)
                )
                result += self.g_inv[a, b] * second
        return sp.simplify(result)

    def nonzero_christoffel(self):
        return {
            (rho, mu, nu): value
            for rho, mu, nu in product(range(self.n), repeat=3)
            if (value := sp.simplify(self.Gamma[rho][mu][nu])) != 0
        }

    def independent_riemann(self):
        """Representantes R_abcd con a<b, c<d y (ab)<=(cd)."""
        result = {}
        pairs = [(a, b) for a in range(self.n) for b in range(a + 1, self.n)]
        for i, (a, b) in enumerate(pairs):
            for c, d in pairs[i:]:
                value = sp.simplify(self.Riemann_down[a][b][c][d])
                if value != 0:
                    result[(a, b, c, d)] = value
        return result

    def independent_einstein_hilbert_momentum(self):
        """Representantes de P^{abcd}=1/2(g^{ac}g^{bd}-g^{ad}g^{bc}).

        Se aprovechan las simetrias de Riemann: cada par esta ordenado y solo
        se conserva una de las componentes relacionadas por intercambio de
        pares. Asi no se imprimen las 81 componentes redundantes en 3D.
        """
        result = {}
        pairs = [(a, b) for a in range(self.n) for b in range(a + 1, self.n)]
        for i, (a, b) in enumerate(pairs):
            for c, d in pairs[i:]:
                value = sp.simplify(sp.Rational(1, 2) * (
                    self.g_inv[a, c] * self.g_inv[b, d]
                    - self.g_inv[a, d] * self.g_inv[b, c]
                ))
                if value != 0:
                    result[(a, b, c, d)] = value
        return result

    def kretschmann(self) -> sp.Expr:
        n, gi, rd = self.n, self.g_inv, self.Riemann_down
        value = sp.S.Zero
        for a, b, c, d, e, f, h, i in product(range(n), repeat=8):
            value += (
                gi[a, e] * gi[b, f] * gi[c, h] * gi[d, i]
                * rd[a][b][c][d] * rd[e][f][h][i]
            )
        return sp.simplify(value)
