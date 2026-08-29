"""Caso-2 de Draft4: acoplamiento derivativo del escalar con la curvatura.

El modulo mantiene los mismos dos niveles de los Casos 0 y 1:
1. momentos y ecuaciones tensoriales sin elegir metrica ni perfil escalar;
2. evaluacion coordinada sobre g[f], phi=p*varphi y, finalmente, f=f_(2).
"""

from __future__ import annotations

from itertools import product

import sympy as sp

from mc_core import CouplingContext, latex_expr


def _curvature_momentum(geo, coefficient: sp.Matrix) -> dict[tuple[int, int, int, int], sp.Expr]:
    """Construye P^{abcd} para un lagrangiano C^{ab}R_ab."""
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


def _independent_rank4(momentum, n: int) -> dict[tuple[int, int, int, int], sp.Expr]:
    """Representantes no nulos usando las simetrias de Riemann."""
    result = {}
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    for i, (a, b) in enumerate(pairs):
        for c, d in pairs[i:]:
            value = sp.simplify(momentum[a, b, c, d])
            if value != 0:
                result[(a, b, c, d)] = value
    return result


def _lower_rank4(momentum, geo) -> dict[tuple[int, int, int, int], sp.Expr]:
    """Baja los cuatro indices de P usando la metrica del ansatz."""
    n, g = geo.n, geo.g
    lowered = {}
    for a, b, c, d in product(range(n), repeat=4):
        value = sp.S.Zero
        for i, j, k, l in product(range(n), repeat=4):
            value += g[a, i] * g[b, j] * g[c, k] * g[d, l] * momentum[i, j, k, l]
        lowered[a, b, c, d] = sp.simplify(value)
    return lowered


def _generalized_ricci(momentum, geo) -> sp.Matrix:
    """Calcula Rcal_ab=P_a{}^{cde} R_bcde por contraccion directa."""
    n, g, riem = geo.n, geo.g, geo.Riemann_down
    return sp.Matrix(n, n, lambda a, b: sp.simplify(sum(
        g[a, i] * momentum[i, c, d, e] * riem[b][c][d][e]
        for i, c, d, e in product(range(n), repeat=4)
    )))


def _double_divergence(momentum, geo) -> sp.Matrix:
    """Calcula nabla^m nabla^n P_(a|mn|b) con todos los indices bajos."""
    n, gi, gamma, x = geo.n, geo.g_inv, geo.Gamma, geo.x
    lowered = _lower_rank4(momentum, geo)
    tensor = {
        (a, m, q, b): sp.simplify(sp.Rational(1, 2) * (
            lowered[a, m, q, b] + lowered[b, m, q, a]
        ))
        for a, m, q, b in product(range(n), repeat=4)
    }

    def cov4(derivative, a, m, q, b):
        indices = (a, m, q, b)
        value = sp.diff(tensor[indices], x[derivative])
        for position in range(4):
            for s in range(n):
                shifted = list(indices)
                shifted[position] = s
                value -= gamma[s][derivative][indices[position]] * tensor[tuple(shifted)]
        return sp.simplify(value)

    first = {}
    for a, m, b in product(range(n), repeat=3):
        first[a, m, b] = sp.simplify(sum(
            gi[q, derivative] * cov4(derivative, a, m, q, b)
            for q, derivative in product(range(n), repeat=2)
        ))

    def cov3(derivative, a, m, b):
        indices = (a, m, b)
        value = sp.diff(first[indices], x[derivative])
        for position in range(3):
            for s in range(n):
                shifted = list(indices)
                shifted[position] = s
                value -= gamma[s][derivative][indices[position]] * first[tuple(shifted)]
        return sp.simplify(value)

    return sp.Matrix(n, n, lambda a, b: sp.simplify(sum(
        gi[m, derivative] * cov3(derivative, a, m, b)
        for m, derivative in product(range(n), repeat=2)
    )))


def _vector_divergence(vector: sp.Matrix, geo) -> sp.Expr:
    """Divergencia de un vector contravariante."""
    return sp.simplify(sum(
        sp.diff(vector[a], geo.x[a])
        + sum(geo.Gamma[a][a][b] * vector[b] for b in range(geo.n))
        for a in range(geo.n)
    ))


def _momentum_latex(momentum, names, label: str) -> str:
    terms = [
        rf"{label}^{{{{{names[a]}}}{{{names[b]}}}{{{names[c]}}}{{{names[d]}}}}}={latex_expr(value)}"
        for (a, b, c, d), value in momentum.items()
    ]
    return r"\begin{aligned}" + r",\\[2pt]".join(r"&" + term for term in terms) + r"\end{aligned}"


def _diagonal_tensor_latex(tensor: sp.Matrix, names, label: str) -> str:
    """Presenta un tensor diagonal por componentes para mantenerlo legible en PDF."""
    terms = [
        rf"{label}_{{{names[index]}{names[index]}}}={latex_expr(tensor[index, index])}"
        for index in range(tensor.rows)
    ]
    return r"\begin{aligned}" + r",\\[4pt]".join(r"&" + term for term in terms) + r"\end{aligned}"


def build_case2(ctx: CouplingContext) -> CouplingContext:
    """Registra el Caso-2 a nivel tensorial, antes de cualquier ansatz."""
    group = "Casos I: formulacion tensorial sin ansatz::Caso-2"
    ctx.add(
        "case2_truncation", "Truncamiento del Draft4",
        r"\alpha_{n\ge2}=\beta_{m\ge1}=\alpha_1",
        r"0,\qquad \beta_0\ne0",
        group,
    )
    ctx.add(
        "case2_lagrangian", "Lagrangiano antes de imponer el ansatz",
        r"L_2[g,\phi]",
        r"R+\frac{2}{\ell^2}+\ell^2\beta_0\left(3R_{ab}u^au^b-RX\right),\quad X=u^au_a",
        group,
    )
    ctx.add(
        "case2_action", "Accion bulk",
        r"I_2",
        r"\frac{1}{16\pi G}\int_Md^3x\sqrt{-g}\,L_2",
        group,
    )
    ctx.add(
        "case2_C", "Tensor coeficiente de Ricci",
        r"C^{ab}",
        r"g^{ab}+\ell^2\beta_0H^{ab},\qquad H^{ab}=3u^au^b-Xg^{ab}",
        group,
        r"El sector de curvatura puede escribirse como \(C^{ab}R_{ab}\).",
    )
    ctx.add(
        "case2_P", "Momento de curvatura",
        r"P_2^{abcd}",
        r"\frac14\left(C^{ac}g^{bd}-C^{ad}g^{bc}-C^{bc}g^{ad}+C^{bd}g^{ac}\right)",
        group,
    )
    ctx.add(
        "case2_M", "Momento metrico",
        r"M^{(2)}_{ab}",
        r"2R_{ab}+\ell^2\beta_0\left\{3\left[R_{(a|c|b)d}u^cu^d+2R_{c(a}u_{b)}u^c\right]-2XR_{ab}-Ru_au_b\right\}",
        group,
    )
    ctx.add(
        "case2_J", "Momento de gradiente escalar",
        r"J_2^a",
        r"2\ell^2\beta_0\left(3R^{ab}-Rg^{ab}\right)u_b",
        group,
    )
    ctx.add(
        "case2_F", "Momento escalar explicito",
        r"F_\phi^{(2)}",
        r"0",
        group,
        "El Caso-2 conserva la simetria de desplazamiento del escalar.",
    )
    ctx.add(
        "case2_Rcal", "Ricci generalizado",
        r"\mathcal R^{(2)}_{ab}",
        r"P^{(2)}_a{}^{cde}R_{bcde}",
        group,
    )
    ctx.add(
        "case2_divP", "Divergencia del momento de curvatura",
        r"\nabla_aP_2^{abcd}",
        r"\frac{\ell^2\beta_0}{4}\nabla_a\left(H^{ac}g^{bd}-H^{ad}g^{bc}-H^{bc}g^{ad}+H^{bd}g^{ac}\right)",
        group,
    )
    ctx.add(
        "case2_Eab", "Ecuacion metrica sin ansatz",
        r"E^{(2)}_{ab}",
        r"\mathcal R^{(2)}_{(ab)}-\frac12g_{ab}L_2-2\nabla^m\nabla^nP^{(2)}_{(a|mn|b)}+\frac12J^{(2)}_{(a}u_{b)}",
        group,
    )
    ctx.add(
        "case2_Ephi", "Ecuacion escalar sin ansatz",
        r"E^{(2)}_\phi",
        r"-\nabla_aJ_2^a=-2\ell^2\beta_0\nabla_a\left[(3R^{ab}-Rg^{ab})u_b\right]",
        group,
    )
    ctx.add(
        "case2_bianchi", "Bianchi-Noether off-shell del Caso-2",
        r"2\nabla^aE^{(2)}_{ab}+E^{(2)}_\phi u_b",
        r"0",
        group,
        check=sp.S.Zero,
    )
    ctx.add(
        "case2_boundary_scalar", "Termino de frontera escalar",
        r"n_aJ_2^a\,\delta\phi",
        r"2\ell^2\beta_0n_a(3R^{ab}-Rg^{ab})u_b\,\delta\phi",
        group,
        r"Se anula genericamente con Dirichlet; sobre el ansatz diagonal se anula automaticamente.",
    )
    ctx.add(
        "case2_boundary_metric", "Residuo metrico bajo Dirichlet",
        r"\Theta_{(g)}\big|_{\delta h=0}",
        r"-2\delta K+2\ell^2\beta_0X\,\delta K-3\ell^2\beta_0u^au^b\delta K_{ab}",
        group,
    )
    ctx.add(
        "case2_boundary_action", "Accion mejorada para Dirichlet",
        r"I^{(2)}_{\mathrm{tot}}",
        r"I_2+\frac{1}{16\pi G}\int_{\partial M}d^2x\sqrt{|h|}\left(2K-2\ell^2\beta_0XK+3\ell^2\beta_0K_{ij}D^i\phi D^j\phi\right)",
        group,
    )
    return ctx


def evaluate_case2_ansatz(ctx: CouplingContext) -> CouplingContext:
    """Evalua el Caso-2 sobre g[f], phi=p*varphi y la solucion de Draft4."""
    group = "Casos II: sustitucion completa del ansatz::Caso-2"
    geo = ctx.objects["geometry"]
    tau, r, varphi = ctx.objects["coordinates"]
    ell, f, lam = ctx.objects["ell"], ctx.objects["f"], ctx.objects["lambda"]
    p = ctx.objects["p"]
    beta0 = sp.symbols("beta_0", real=True, nonzero=True)
    coupling = ell**2 * beta0
    phi = p * varphi
    names = (r"\tau", "r", r"\varphi")

    u_cov = geo.scalar_gradient_cov(phi)
    u_up = (geo.g_inv * u_cov).applyfunc(sp.simplify)
    X = sp.simplify((u_cov.T * u_up)[0])
    ricci_upup = (geo.g_inv * geo.Ricci * geo.g_inv).applyfunc(sp.simplify)
    scalar_ricci_coupling = sp.simplify((u_up.T * geo.Ricci * u_up)[0])
    H = (3 * u_up * u_up.T - X * geo.g_inv).applyfunc(sp.simplify)
    C = (geo.g_inv + coupling * H).applyfunc(sp.simplify)
    P_full = _curvature_momentum(geo, C)
    P_independent = _independent_rank4(P_full, geo.n)
    Rcal = _generalized_ricci(P_full, geo)
    Rcal_sym = sp.Rational(1, 2) * (Rcal + Rcal.T)
    J = (2 * coupling * (3 * ricci_upup * u_cov - geo.Rscalar * u_up)).applyfunc(sp.simplify)
    J_lower = (geo.g * J).applyfunc(sp.simplify)
    Fphi = sp.S.Zero

    # Momento metrico calculado directamente a R_abcd y u_a fijos.
    M = sp.zeros(geo.n)
    for a, b in product(range(geo.n), repeat=2):
        riemann_piece = sp.Rational(1, 2) * sum(
            (geo.Riemann_down[a][c][b][d] + geo.Riemann_down[b][c][a][d])
            * u_up[c] * u_up[d]
            for c, d in product(range(geo.n), repeat=2)
        )
        ricci_piece = sum(
            (geo.Ricci[c, a] * u_cov[b] + geo.Ricci[c, b] * u_cov[a]) * u_up[c]
            for c in range(geo.n)
        )
        M[a, b] = sp.simplify(
            2 * geo.Ricci[a, b]
            + coupling * (
                3 * (riemann_piece + ricci_piece)
                - 2 * X * geo.Ricci[a, b]
                - geo.Rscalar * u_cov[a] * u_cov[b]
            )
        )

    moment_identity = (M - 2 * Rcal_sym - sp.Matrix(geo.n, geo.n, lambda a, b:
        sp.Rational(1, 4) * (J_lower[a] * u_cov[b] + J_lower[b] * u_cov[a])
    )).applyfunc(sp.simplify)

    double_divergence = _double_divergence(P_full, geo)
    lagrangian = sp.simplify(
        geo.Rscalar + 2 / ell**2
        + coupling * (3 * scalar_ricci_coupling - geo.Rscalar * X)
    )
    Ephi = sp.simplify(-_vector_divergence(J, geo))
    E = sp.Matrix(geo.n, geo.n, lambda a, b: sp.simplify(
        Rcal_sym[a, b]
        - sp.Rational(1, 2) * geo.g[a, b] * lagrangian
        - 2 * double_divergence[a, b]
        + sp.Rational(1, 4) * (J_lower[a] * u_cov[b] + J_lower[b] * u_cov[a])
    ))

    ctx.put("beta0", beta0)
    ctx.put("phi_case2", phi)
    ctx.put("P_case2_ansatz", P_independent)
    ctx.put("M_case2_ansatz", M)
    ctx.put("J_case2_ansatz", J)
    ctx.put("F_case2_ansatz", Fphi)
    ctx.put("Rcal_case2_ansatz", Rcal)
    ctx.put("doubledivP_case2_ansatz", double_divergence)
    ctx.put("Ephi_case2_ansatz", Ephi)
    ctx.put("E_case2_ansatz", E)

    ctx.add(
        "case2_ansatz_data", "Sustitucion simultanea de metrica y escalar",
        r"(ds^2,\phi)",
        r"\left(-f(r)d\tau^2+\frac{dr^2}{f(r)}+r^2d\varphi^2,\;p\varphi\right)",
        group,
    )
    ctx.add(
        "case2_ansatz_gradient", "Gradiente escalar y contraccion cinetica",
        r"(u_a,u^a,X)",
        rf"\left({latex_expr(u_cov)},\,{latex_expr(u_up)},\,{latex_expr(X)}\right)",
        group,
    )
    ctx.add(
        "case2_ansatz_C", "Coeficiente de Ricci sobre el ansatz",
        r"C^{ab}\big|_{g[f],\phi=p\varphi}", latex_expr(C), group,
    )
    ctx.add(
        "case2_ansatz_P", "Momento de curvatura sobre el ansatz",
        r"\{P_2^{abcd}\}_{\mathrm{indep}}\big|_{g[f],\phi=p\varphi}",
        _momentum_latex(P_independent, names, "P_2"), group,
    )
    ctx.add(
        "case2_ansatz_M", "Momento metrico sobre el ansatz",
        r"\{M^{(2)}_{aa}\}_{\rm diag}\big|_{g[f],\phi=p\varphi}",
        _diagonal_tensor_latex(M, names, r"M^{(2)}"), group,
    )
    ctx.add(
        "case2_ansatz_J", "Momento de gradiente escalar sobre el ansatz",
        r"J_2^a\big|_{g[f],\phi=p\varphi}", latex_expr(J), group,
    )
    ctx.add(
        "case2_ansatz_F", "Momento escalar explicito sobre el ansatz",
        r"F_\phi^{(2)}\big|_{g[f],\phi=p\varphi}", latex_expr(Fphi), group,
    )
    ctx.add(
        "case2_ansatz_Rcal", "Ricci generalizado sobre el ansatz",
        r"\{\mathcal R^{(2)}_{aa}\}_{\rm diag}\big|_{g[f],\phi=p\varphi}",
        _diagonal_tensor_latex(Rcal, names, r"\mathcal R^{(2)}"), group,
    )
    ctx.add(
        "case2_moment_identity", "Chequeo de la identidad algebraica de momentos",
        r"M^{(2)}_{ab}-2\mathcal R^{(2)}_{(ab)}-\frac12J^{(2)}_{(a}u_{b)}",
        latex_expr(moment_identity), group,
        check=sum((entry**2 for entry in moment_identity), sp.S.Zero),
    )
    ctx.add(
        "case2_ansatz_doubledivP", "Doble divergencia del momento de curvatura",
        r"\nabla^m\nabla^nP^{(2)}_{(a|mn|b)}\big|_{g[f],\phi=p\varphi}",
        latex_expr(double_divergence), group,
    )
    ctx.add(
        "case2_ansatz_Ephi", "Ecuacion escalar sobre el ansatz",
        r"E^{(2)}_\phi\big|_{g[f],\phi=p\varphi}", latex_expr(Ephi), group,
        check=Ephi,
    )
    ctx.add(
        "case2_ansatz_E", "Tensor metrico antes de resolver f(r)",
        r"\{E^{(2)}_{aa}\}_{\rm diag}\big|_{g[f],\phi=p\varphi}",
        _diagonal_tensor_latex(E, names, r"E^{(2)}"), group,
    )

    Hradial = sp.simplify(1 + beta0 * p**2 * ell**2 / r**2)
    f_solution = sp.factor((r**2 / ell**2 - lam) / Hradial)
    ctx.put("H_case2", Hradial)
    ctx.put("f_case2_solution", f_solution)
    ctx.add(
        "case2_radial_equations", "Ecuaciones radiales en la variable N=Hf",
        r"N(r)=H(r)f(r),\qquad E^{(2)}_{ab}=0",
        r"N'(r)=\frac{2r}{\ell^2},\qquad N''(r)=\frac{2}{\ell^2}",
        group,
    )
    ctx.add(
        "case2_f_solution", "Solucion racional del Draft4",
        r"f_{(2)}(r)", latex_expr(f_solution), group,
        check=sp.simplify(Hradial * f_solution - (r**2 / ell**2 - lam)),
    )

    substitutions = {
        f: f_solution,
        sp.diff(f, r): sp.diff(f_solution, r),
        sp.diff(f, r, 2): sp.diff(f_solution, r, 2),
        sp.diff(f, r, 3): sp.diff(f_solution, r, 3),
        sp.diff(f, r, 4): sp.diff(f_solution, r, 4),
    }
    P_final = {key: sp.factor(value.subs(substitutions)) for key, value in P_independent.items()}
    M_final = M.applyfunc(lambda value: sp.factor(value.subs(substitutions)))
    J_final = J.applyfunc(lambda value: sp.factor(value.subs(substitutions)))
    Rcal_final = Rcal.applyfunc(lambda value: sp.factor(value.subs(substitutions)))
    E_final = E.applyfunc(lambda value: sp.factor(value.subs(substitutions)))
    metric_final = geo.g.applyfunc(lambda value: sp.factor(value.subs(substitutions)))
    ricci_final = geo.Ricci.applyfunc(lambda value: sp.factor(value.subs(substitutions)))
    R_final = sp.factor(geo.Rscalar.subs(substitutions))
    ricci_squared = sp.factor(sum(
        geo.g_inv[a, c] * geo.g_inv[b, d] * geo.Ricci[a, b] * geo.Ricci[c, d]
        for a, b, c, d in product(range(geo.n), repeat=4)
    ).subs(substitutions))
    kretschmann = sp.factor(4 * ricci_squared - R_final**2)
    bianchi = geo.divergence_cov2(E).applyfunc(sp.simplify)
    noether = (2 * bianchi + Ephi * u_cov).applyfunc(sp.simplify)

    ctx.put("P_case2_final", P_final)
    ctx.put("M_case2_final", M_final)
    ctx.put("J_case2_final", J_final)
    ctx.put("F_case2_final", Fphi)
    ctx.put("Rcal_case2_final", Rcal_final)
    ctx.put("E_case2_solution", E_final)
    ctx.put("metric_case2_solution", metric_final)
    ctx.put("Ricci_case2_solution", ricci_final)
    ctx.put("R_case2_solution", R_final)
    ctx.put("Ricci2_case2_solution", ricci_squared)
    ctx.put("K_case2_solution", kretschmann)
    ctx.put("bianchi_case2_ansatz", bianchi)
    ctx.put("noether_case2_ansatz", noether)

    ctx.add(
        "case2_final_P", "Momento de curvatura final",
        r"\{P_2^{abcd}\}_{\mathrm{indep}}\big|_{f=f_{(2)},\phi=p\varphi}",
        _momentum_latex(P_final, names, "P_2"), group,
        r"Aqui y en los cuatro bloques siguientes ya se uso la solucion racional explicita \(f_{(2)}(r)\).",
    )
    ctx.add(
        "case2_final_M", "Momento metrico final",
        r"\{M^{(2)}_{aa}\}_{\rm diag}\big|_{f=f_{(2)},\phi=p\varphi}",
        _diagonal_tensor_latex(M_final, names, r"M^{(2)}"), group,
    )
    ctx.add(
        "case2_final_J", "Momento de gradiente escalar final",
        r"J_2^a\big|_{f=f_{(2)},\phi=p\varphi}", latex_expr(J_final), group,
    )
    ctx.add(
        "case2_final_F", "Momento escalar explicito final",
        r"F_\phi^{(2)}\big|_{f=f_{(2)},\phi=p\varphi}", latex_expr(Fphi), group,
    )
    ctx.add(
        "case2_final_Rcal", "Ricci generalizado final",
        r"\{\mathcal R^{(2)}_{aa}\}_{\rm diag}\big|_{f=f_{(2)},\phi=p\varphi}",
        _diagonal_tensor_latex(Rcal_final, names, r"\mathcal R^{(2)}"), group,
    )
    ctx.add(
        "case2_bianchi_ansatz", "Bianchi metrico antes de resolver f(r)",
        r"\nabla^aE^{(2)}_{ab}\big|_{g[f],\phi=p\varphi}", latex_expr(bianchi), group,
        check=sum((entry**2 for entry in bianchi), sp.S.Zero),
    )
    ctx.add(
        "case2_noether_ansatz", "Identidad Bianchi-Noether completa",
        r"2\nabla^aE^{(2)}_{ab}+E^{(2)}_\phi u_b", latex_expr(noether), group,
        check=sum((entry**2 for entry in noether), sp.S.Zero),
    )
    ctx.add(
        "case2_field_solution", "Verificacion final de la ecuacion metrica",
        r"E^{(2)}_{ab}\big|_{f=f_{(2)},\phi=p\varphi}", latex_expr(E_final), group,
        check=sum((entry**2 for entry in E_final), sp.S.Zero),
    )
    ctx.add(
        "case2_scalar_solution", "Verificacion final de la ecuacion escalar",
        r"E^{(2)}_\phi\big|_{f=f_{(2)},\phi=p\varphi}", latex_expr(Ephi), group,
        check=Ephi,
    )
    ctx.add(
        "case2_R_solution", "Escalar de Ricci de la solucion racional",
        r"R\big|_{f=f_{(2)}}", latex_expr(R_final), group,
    )
    ctx.add(
        "case2_Ricci_solution", "Tensor de Ricci de la solucion racional",
        r"\{R_{aa}\}_{\rm diag}\big|_{f=f_{(2)}}",
        _diagonal_tensor_latex(ricci_final, names, r"R"), group,
    )
    ctx.add(
        "case2_Ricci2_solution", "Invariante cuadratico de Ricci",
        r"R_{ab}R^{ab}\big|_{f=f_{(2)}}", latex_expr(ricci_squared), group,
    )
    ctx.add(
        "case2_K_solution", "Invariante de Kretschmann en tres dimensiones",
        r"R_{abcd}R^{abcd}\big|_{f=f_{(2)}}", latex_expr(kretschmann), group,
        r"Se usa la identidad tridimensional \(R_{abcd}R^{abcd}=4R_{ab}R^{ab}-R^2\).",
    )
    ctx.add(
        "case2_boundary_ansatz", "Flujo escalar normal al borde radial",
        r"n_aJ_2^a\big|_{\phi=p\varphi,\,r=\mathrm{cte}}", r"0", group,
        "La corriente es puramente tangencial sobre la metrica diagonal.",
        check=sp.S.Zero,
    )
    return ctx
