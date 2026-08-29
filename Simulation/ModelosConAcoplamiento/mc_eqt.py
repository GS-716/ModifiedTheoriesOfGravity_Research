"""Pipeline general para sumas finitas de invariantes EQT.

La implementacion reutiliza la geometria y el contexto existentes. No sustituye
los Casos 0--2: los contiene como limites y los usa como regresiones del motor.
"""

from __future__ import annotations

from itertools import product

import sympy as sp

from mc_core import CouplingContext, latex_expr
from mc_invariants import (
    EQTModelSpec,
    alpha_current,
    analytic_branch,
    beta_current,
    beta_ricci_coefficient,
    eqt_density,
)
from mc_tensor import (
    curvature_momentum,
    diagonal_tensor_latex,
    double_divergence,
    generalized_ricci,
    independent_rank4,
    momentum_latex,
    symmetric_current_gradient,
    vector_divergence,
)


def build_eqt_general(ctx: CouplingContext, spec: EQTModelSpec) -> CouplingContext:
    """Registra la base de invariantes y las reglas variacionales seleccionadas."""
    spec.validate()
    group = "Generalizacion EQT I: lagrangiano e invariantes"
    ell = sp.symbols("ell", positive=True)
    X, Y, R = sp.symbols("X Y R")
    abstract_density = eqt_density(spec, ell, X, Y, R)

    ctx.put("eqt_spec", spec)
    ctx.put("eqt_abstract_density", abstract_density)
    ctx.add(
        "eqt_supported_basis", "Base de invariantes soportados",
        r"(X,Y,\mathcal A_n,\mathcal B_m)",
        r"\left(u^au_a,\,R_{ab}u^au^b,\,-\ell^{2(n-1)}X^n,\,"
        r"\ell^{2(m+1)}X^m[(3+2m)Y-XR]\right)",
        group,
        "La especificacion elige cualquier subconjunto finito de las dos torres EQT.",
    )
    ctx.add(
        "eqt_selected_model", "Lagrangiano configurado",
        r"L_{\mathrm{EQT}}", latex_expr(abstract_density), group,
        rf"Ordenes activos: alpha={spec.alpha_orders}, beta={spec.beta_orders}.",
    )
    ctx.add(
        "eqt_linearity", "Regla de composicion",
        r"(P,M,J,F_\phi)[L_1+L_2]",
        r"(P,M,J,F_\phi)[L_1]+(P,M,J,F_\phi)[L_2]",
        group,
        "La linealidad permite sumar contribuciones invariantes antes de evaluar el ansatz.",
    )

    for order, coefficient in sorted(spec.alpha.items()):
        ctx.add(
            f"eqt_alpha_{order}_rule", f"Regla variacional del invariante alpha-{order}",
            rf"\mathcal A_{{{order}}}",
            rf"-{latex_expr(coefficient)}\ell^{{{2*(order-1)}}}X^{{{order}}},\quad "
            rf"P^{{abcd}}=0,\quad F_\phi=0,\quad "
            rf"J^a=-2({order}){latex_expr(coefficient)}\ell^{{{2*(order-1)}}}"
            rf"X^{{{order-1}}}u^a",
            group,
            r"El momento metrico se reconstruye con la identidad de difeomorfismos.",
        )

    for order, coefficient in sorted(spec.beta.items()):
        ctx.add(
            f"eqt_beta_{order}_rule", f"Regla variacional del invariante beta-{order}",
            rf"\mathcal B_{{{order}}}",
            rf"{latex_expr(coefficient)}\ell^{{{2*(order+1)}}}X^{{{order}}}"
            rf"\left[({3+2*order})R_{{ab}}u^au^b-XR\right]",
            group,
            r"Su momento de curvatura se genera desde el coeficiente de $R_{ab}$; "
            r"$J^a$ se obtiene derivando tanto $X^m$ como las contracciones con $u^a$.",
        )

    ctx.add(
        "eqt_moment_reconstruction", "Reconstruccion covariante del momento metrico",
        r"M_{ab}", r"2\mathcal R_{(ab)}+\frac12J_{(a}u_{b)}", group,
        "Se aplica despues de componer P y J de todos los invariantes.",
    )
    return ctx


def evaluate_eqt_general_ansatz(ctx: CouplingContext, spec: EQTModelSpec) -> CouplingContext:
    """Compone momentos, reduce radialmente y evalua la rama EQT analitica."""
    spec.validate()
    group_ansatz = "Generalizacion EQT II: motor sobre el ansatz"
    group_final = "Generalizacion EQT III: rama y verificacion"

    geo = ctx.objects["geometry"]
    tau, r, varphi = ctx.objects["coordinates"]
    ell, f, lam = ctx.objects["ell"], ctx.objects["f"], ctx.objects["lambda"]
    p = ctx.objects["p"]
    r0 = ctx.objects["r0"]
    names = (r"\tau", "r", r"\varphi")

    phi = p * varphi
    u_cov = geo.scalar_gradient_cov(phi)
    u_up = (geo.g_inv * u_cov).applyfunc(sp.simplify)
    X = sp.simplify((u_cov.T * u_up)[0])
    ricci_upup = (geo.g_inv * geo.Ricci * geo.g_inv).applyfunc(sp.simplify)
    Y = sp.simplify((u_up.T * geo.Ricci * u_up)[0])

    coefficient = geo.g_inv.copy()
    for order, coupling in spec.beta.items():
        coefficient += beta_ricci_coefficient(
            order, coupling, ell, X, u_up, geo.g_inv
        )
    coefficient = coefficient.applyfunc(sp.simplify)

    P_full = curvature_momentum(geo, coefficient)
    P_independent = independent_rank4(P_full, geo.n)
    Rcal = generalized_ricci(P_full, geo)
    Rcal_sym = ((Rcal + Rcal.T) / 2).applyfunc(sp.simplify)

    J = sp.zeros(geo.n, 1)
    for order, coupling in spec.alpha.items():
        J += alpha_current(order, coupling, ell, X, u_up)
    for order, coupling in spec.beta.items():
        J += beta_current(
            order, coupling, ell, X, Y, geo.Rscalar,
            ricci_upup, u_cov, u_up,
        )
    J = J.applyfunc(sp.simplify)
    J_lower = (geo.g * J).applyfunc(sp.simplify)
    Ju = symmetric_current_gradient(J_lower, u_cov)
    M = (2 * Rcal_sym + Ju).applyfunc(sp.simplify)

    lagrangian = eqt_density(spec, ell, X, Y, geo.Rscalar)
    double_div_P = double_divergence(P_full, geo)
    Ephi = sp.simplify(-vector_divergence(J, geo))
    E = sp.Matrix(geo.n, geo.n, lambda a, b: sp.factor(
        Rcal_sym[a, b]
        - sp.Rational(1, 2) * geo.g[a, b] * lagrangian
        - 2 * double_div_P[a, b]
        + Ju[a, b]
    ))

    ctx.put("eqt_phi", phi)
    ctx.put("eqt_X", X)
    ctx.put("eqt_Y", Y)
    ctx.put("eqt_C", coefficient)
    ctx.put("eqt_L_ansatz", lagrangian)
    ctx.put("eqt_P_ansatz", P_independent)
    ctx.put("eqt_M_ansatz", M)
    ctx.put("eqt_J_ansatz", J)
    ctx.put("eqt_F_ansatz", sp.S.Zero)
    ctx.put("eqt_Rcal_ansatz", Rcal)
    ctx.put("eqt_doubledivP_ansatz", double_div_P)
    ctx.put("eqt_Ephi_ansatz", Ephi)
    ctx.put("eqt_E_ansatz", E)

    ctx.add(
        "eqt_ansatz_data", "Ansatz comun para el modelo configurado",
        r"(ds^2,\phi)",
        r"\left(-f(r)d\tau^2+\frac{dr^2}{f(r)}+r^2d\varphi^2,\,p\varphi\right)",
        group_ansatz,
    )
    ctx.add(
        "eqt_ansatz_invariants", "Invariantes elementales sobre el ansatz",
        r"(X,Y)", rf"\left({latex_expr(X)},\,{latex_expr(Y)}\right)", group_ansatz,
    )
    ctx.add(
        "eqt_ansatz_lagrangian", "Lagrangiano compuesto sobre el ansatz",
        r"L_{\mathrm{EQT}}[f]", latex_expr(lagrangian), group_ansatz,
    )
    ctx.add(
        "eqt_ansatz_C", "Coeficiente total del tensor de Ricci",
        r"C^{ab}_{\mathrm{tot}}", latex_expr(coefficient), group_ansatz,
    )
    ctx.add(
        "eqt_ansatz_P", "Momento de curvatura total",
        r"\{P^{abcd}_{\mathrm{EQT}}\}_{\mathrm{indep}}",
        momentum_latex(P_independent, names, r"P_{\mathrm{EQT}}"), group_ansatz,
    )
    ctx.add(
        "eqt_ansatz_M", "Momento metrico total",
        r"\{M^{\mathrm{EQT}}_{aa}\}_{\mathrm{diag}}",
        diagonal_tensor_latex(M, names, r"M^{\mathrm{EQT}}"), group_ansatz,
    )
    ctx.add(
        "eqt_ansatz_J", "Momento de gradiente escalar total",
        r"J^a_{\mathrm{EQT}}", latex_expr(J), group_ansatz,
    )
    ctx.add(
        "eqt_ansatz_F", "Momento escalar explicito total",
        r"F^{\mathrm{EQT}}_\phi", r"0", group_ansatz,
    )
    ctx.add(
        "eqt_ansatz_Rcal", "Ricci generalizado total",
        r"\{\mathcal R^{\mathrm{EQT}}_{aa}\}_{\mathrm{diag}}",
        diagonal_tensor_latex(Rcal, names, r"\mathcal R^{\mathrm{EQT}}"), group_ansatz,
    )
    ctx.add(
        "eqt_ansatz_doubledivP", "Doble divergencia del momento total",
        r"\nabla^m\nabla^nP^{\mathrm{EQT}}_{(a|mn|b)}",
        diagonal_tensor_latex(double_div_P, names, r"D^{\mathrm{EQT}}"), group_ansatz,
    )
    ctx.add(
        "eqt_ansatz_Ephi", "Ecuacion escalar sobre el ansatz",
        r"E^{\mathrm{EQT}}_\phi", latex_expr(Ephi), group_ansatz, check=Ephi,
    )
    ctx.add(
        "eqt_ansatz_E", "Ecuacion metrica antes de resolver f(r)",
        r"\{E^{\mathrm{EQT}}_{aa}\}_{\mathrm{diag}}",
        diagonal_tensor_latex(E, names, r"E^{\mathrm{EQT}}"), group_ansatz,
    )

    numerator, denominator, f_solution = analytic_branch(spec, r, ell, p, lam, r0)
    radial_residual = sp.factor(sp.diff(denominator * f - numerator, r))
    radial_factor = sp.factor(sp.cancel(E[1, 1] / radial_residual))
    radial_check = sp.factor(E[1, 1] - radial_factor * radial_residual)

    ctx.put("eqt_N", numerator)
    ctx.put("eqt_H", denominator)
    ctx.put("eqt_f_solution", f_solution)
    ctx.put("eqt_radial_residual", radial_residual)
    ctx.put("eqt_radial_factor", radial_factor)
    ctx.add(
        "eqt_radial_reduction", "Reduccion a una ecuacion radial integrable",
        r"E^{\mathrm{EQT}}_{rr}",
        rf"\left({latex_expr(radial_factor)}\right)\frac{{d}}{{dr}}[H(r)f(r)-N(r)]",
        group_final, check=radial_check,
    )
    ctx.add(
        "eqt_N_branch", "Numerador de la rama analitica",
        r"N(r)", latex_expr(numerator), group_final,
    )
    ctx.add(
        "eqt_H_branch", "Denominador de la rama analitica",
        r"H(r)", latex_expr(denominator), group_final,
    )
    ctx.add(
        "eqt_f_branch", "Solucion EQT configurada",
        r"f_{\mathrm{EQT}}(r)", latex_expr(f_solution), group_final,
        check=sp.factor(denominator * f_solution - numerator),
    )

    substitutions = {f: f_solution}
    substitutions.update({
        sp.diff(f, r, order): sp.diff(f_solution, r, order)
        for order in range(1, 5)
    })
    P_final = {key: sp.factor(value.subs(substitutions)) for key, value in P_independent.items()}
    M_final = M.applyfunc(lambda value: sp.factor(value.subs(substitutions)))
    J_final = J.applyfunc(lambda value: sp.factor(value.subs(substitutions)))
    Rcal_final = Rcal.applyfunc(lambda value: sp.factor(value.subs(substitutions)))
    E_final = E.applyfunc(lambda value: sp.factor(value.subs(substitutions)))
    R_final = sp.factor(geo.Rscalar.subs(substitutions))
    ricci_final = geo.Ricci.applyfunc(lambda value: sp.factor(value.subs(substitutions)))
    ricci_squared = sp.factor(sum(
        geo.g_inv[a, c] * geo.g_inv[b, d] * geo.Ricci[a, b] * geo.Ricci[c, d]
        for a, b, c, d in product(range(geo.n), repeat=4)
    ).subs(substitutions))
    kretschmann = sp.factor(4 * ricci_squared - R_final**2)
    bianchi = geo.divergence_cov2(E).applyfunc(sp.simplify)
    noether = (2 * bianchi + Ephi * u_cov).applyfunc(sp.simplify)

    ctx.put("eqt_P_final", P_final)
    ctx.put("eqt_M_final", M_final)
    ctx.put("eqt_J_final", J_final)
    ctx.put("eqt_F_final", sp.S.Zero)
    ctx.put("eqt_Rcal_final", Rcal_final)
    ctx.put("eqt_E_solution", E_final)
    ctx.put("eqt_R_solution", R_final)
    ctx.put("eqt_Ricci_solution", ricci_final)
    ctx.put("eqt_Ricci2_solution", ricci_squared)
    ctx.put("eqt_K_solution", kretschmann)
    ctx.put("eqt_bianchi_ansatz", bianchi)
    ctx.put("eqt_noether_ansatz", noether)

    ctx.add(
        "eqt_final_P", "Momento de curvatura final",
        r"\{P^{abcd}_{\mathrm{EQT}}\}_{\mathrm{indep}}\big|_{f=f_{\mathrm{EQT}}}",
        momentum_latex(P_final, names, r"P_{\mathrm{EQT}}"), group_final,
        "Desde este bloque se ha sustituido la rama completa y todas sus derivadas.",
    )
    ctx.add(
        "eqt_final_M", "Momento metrico final",
        r"\{M^{\mathrm{EQT}}_{aa}\}_{\mathrm{diag}}\big|_{f=f_{\mathrm{EQT}}}",
        diagonal_tensor_latex(M_final, names, r"M^{\mathrm{EQT}}"), group_final,
    )
    ctx.add(
        "eqt_final_J", "Momento de gradiente escalar final",
        r"J^a_{\mathrm{EQT}}\big|_{f=f_{\mathrm{EQT}}}", latex_expr(J_final), group_final,
    )
    ctx.add(
        "eqt_final_F", "Momento escalar explicito final",
        r"F^{\mathrm{EQT}}_\phi\big|_{f=f_{\mathrm{EQT}}}", r"0", group_final,
    )
    ctx.add(
        "eqt_final_Rcal", "Ricci generalizado final",
        r"\{\mathcal R^{\mathrm{EQT}}_{aa}\}_{\mathrm{diag}}\big|_{f=f_{\mathrm{EQT}}}",
        diagonal_tensor_latex(Rcal_final, names, r"\mathcal R^{\mathrm{EQT}}"), group_final,
    )
    ctx.add(
        "eqt_final_field", "Verificacion final de la ecuacion metrica",
        r"E^{\mathrm{EQT}}_{ab}\big|_{f=f_{\mathrm{EQT}}}", latex_expr(E_final), group_final,
        check=sum((entry**2 for entry in E_final), sp.S.Zero),
    )
    ctx.add(
        "eqt_final_scalar", "Verificacion final de la ecuacion escalar",
        r"E^{\mathrm{EQT}}_\phi\big|_{f=f_{\mathrm{EQT}}}", latex_expr(Ephi), group_final,
        check=Ephi,
    )
    ctx.add(
        "eqt_final_noether", "Identidad Bianchi-Noether del modelo compuesto",
        r"2\nabla^aE^{\mathrm{EQT}}_{ab}+E^{\mathrm{EQT}}_\phi u_b",
        latex_expr(noether), group_final,
        check=sum((entry**2 for entry in noether), sp.S.Zero),
    )
    ctx.add(
        "eqt_final_R", "Escalar de Ricci de la rama seleccionada",
        r"R\big|_{f=f_{\mathrm{EQT}}}", latex_expr(R_final), group_final,
    )
    ctx.add(
        "eqt_final_Ricci", "Tensor de Ricci de la rama seleccionada",
        r"\{R_{aa}\}_{\mathrm{diag}}\big|_{f=f_{\mathrm{EQT}}}",
        diagonal_tensor_latex(ricci_final, names, "R"), group_final,
    )
    ctx.add(
        "eqt_final_Ricci2", "Invariante cuadratico de Ricci",
        r"R_{ab}R^{ab}\big|_{f=f_{\mathrm{EQT}}}", latex_expr(ricci_squared), group_final,
    )
    ctx.add(
        "eqt_final_K", "Invariante de Kretschmann",
        r"R_{abcd}R^{abcd}\big|_{f=f_{\mathrm{EQT}}}", latex_expr(kretschmann), group_final,
        r"En tres dimensiones se usa (R_{abcd}R^{abcd}=4R_{ab}R^{ab}-R^2).",
    )
    return ctx
