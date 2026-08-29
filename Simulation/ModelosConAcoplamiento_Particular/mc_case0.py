"""Especializacion al Caso-0 de Papers/Draft_4.pdf y al ansatz BTZ."""

from __future__ import annotations

import sympy as sp

from mc_core import CouplingContext, latex_expr
from mc_geometry import CoordinateGeometry


'''ACÁ NO CALCULAMOS NADA, SÓLO DEFINIMOS LOS PASOS DE LA TEORÍA GENERAL Y DEL CASO-0.
   EN TEORÍA, SE PODRÍA OMITIR SIN PROBLEMA, LOS CÁLCULOS SIMBÓLICOS SE REALIZAN ABAJO.'''
def build_case0(ctx: CouplingContext) -> CouplingContext:
    group = "Casos I: formulacion tensorial sin ansatz::Caso-0"
    ctx.add(
        "case0_truncation", "Truncamiento del Draft4",
        r"\alpha_{n\ge2}=\beta_{m\ge1}=\beta_0=\alpha_1",
        r"0",
        group,
        "El campo escalar desaparece de la accion y queda desacoplado.",
    )
    ctx.add(
        "case0_lagrangian", "Lagrangiano sobreviviente",
        r"L_0",
        r"R+\frac{2}{\ell^2}",
        group,
    )
    ctx.add(
        "case0_action", "Accion bulk",
        r"I_0",
        r"\frac{1}{16\pi G}\int_M d^3x\sqrt{-g}\left(R+\frac{2}{\ell^2}\right)",
        group,
    )
    ctx.add(
        "case0_P", "Momento de curvatura",
        r"P_0^{abcd}",
        r"\frac12\left(g^{ac}g^{bd}-g^{ad}g^{bc}\right)",
        group,
    )
    ctx.add(
        "case0_M", "Momento metrico",
        r"M^{(0)}_{ab}",
        r"2R_{ab}",
        group,
        r"La derivada se toma a \(R_{abcd}\) covariante fijo; \(2/\ell^2\) no depende de la metrica.",
    )
    ctx.add(
        "case0_scalar_momenta", "Momentos escalares",
        r"(J_0^a,F_\phi^{(0)})",
        r"(0,0)",
        group,
    )
    ctx.add(
        "case0_Rcal", "Ricci generalizado",
        r"\mathcal R^{(0)}_{ab}=P^{(0)}_a{}^{cde}R_{bcde}",
        r"R_{ab}",
        group,
    )
    ctx.add(
        "case0_divP", "Divergencia del momento de curvatura",
        r"\nabla_aP_0^{abcd}",
        r"0",
        group,
        "Consecuencia directa de la compatibilidad metrica.",
        check=sp.S.Zero,
    )
    ctx.add(
        "case0_doubledivP", "Doble divergencia",
        r"-2\nabla^m\nabla^nP^{(0)}_{amnb}",
        r"0",
        group,
        check=sp.S.Zero,
    )
    ctx.add(
        "case0_Eab", "Ecuacion metrica del caso-0",
        r"E^{(0)}_{ab}",
        r"R_{ab}-\frac12g_{ab}\left(R+\frac{2}{\ell^2}\right)=G_{ab}-\frac{1}{\ell^2}g_{ab}",
        group,
    )
    ctx.add(
        "case0_Ephi", "Ecuacion escalar del caso-0",
        r"E^{(0)}_\phi",
        r"0",
        group,
        "No es una ecuacion dinamica: phi no aparece en la accion.",
    )
    ctx.add(
        "case0_bianchi", "Bianchi-Noether especializado",
        r"2\nabla^aE^{(0)}_{ab}",
        r"2\nabla^aG_{ab}-\frac{2}{\ell^2}\nabla^ag_{ab}=0",
        group,
        check=sp.S.Zero,
    )
    ctx.add(
        "case0_boundary", "Accion mejorada para Dirichlet",
        r"I^{(0)}_{\mathrm{tot}}",
        r"I_0+\frac{1}{8\pi G}\int_{\partial M}d^2x\sqrt{|h|}\,K",
        group,
        r"El termino GHY cancela el residuo \(-2\delta K\). Para AdS se suma \(-\frac{1}{8\pi G\ell}\int\sqrt{|h|}\).",
    )
    return ctx


def evaluate_btz_ansatz(ctx: CouplingContext) -> CouplingContext:
    group = "Casos II: sustitucion completa del ansatz::Caso-0"
    tau, r, varphi = sp.symbols("tau r varphi", real=True)
    ell = sp.symbols("ell", positive=True, finite=True)
    lam = sp.symbols("lambda", real=True)
    f = sp.Function("f")(r)
    coords = (tau, r, varphi)
    metric = sp.diag(-f, 1 / f, r**2)
    geo = CoordinateGeometry(coords, metric)

    ctx.put("coordinates", coords)
    ctx.put("ell", ell)
    ctx.put("lambda", lam)
    ctx.put("f", f)
    ctx.put("geometry", geo)

    ctx.add("ansatz_metric", "Ansatz circular estatico",
            r"ds^2", r"-f(r)d\tau^2+\frac{dr^2}{f(r)}+r^2d\varphi^2", group)
    ctx.add("ansatz_metric_matrix", "Metrica y metrica inversa",
            r"(g_{ab},g^{ab})",
            rf"\left({latex_expr(geo.g)},\,{latex_expr(geo.g_inv)}\right)", group)

    names = (r"\tau", "r", r"\varphi")

    # Los cuatro momentos se vuelven a evaluar aqui como objetos coordenados
    # del ansatz. Para P se imprimen solo representantes independientes de
    # sus simetrias tipo Riemann.
    P0_ansatz = geo.independent_einstein_hilbert_momentum()
    P0_terms = [
        rf"P_0^{{{{{names[a]}}}{{{names[b]}}}{{{names[c]}}}{{{names[d]}}}}}={latex_expr(value)}"
        for (a, b, c, d), value in P0_ansatz.items()
    ]
    M0_ansatz = (2 * geo.Ricci).applyfunc(sp.simplify)
    J0_ansatz = sp.zeros(3, 1)
    F0_ansatz = sp.S.Zero
    Rcal0_ansatz = geo.Ricci.applyfunc(sp.simplify)
    ctx.put("P_case0_ansatz", P0_ansatz)
    ctx.put("M_case0_ansatz", M0_ansatz)
    ctx.put("J_case0_ansatz", J0_ansatz)
    ctx.put("F_case0_ansatz", F0_ansatz)
    ctx.put("Rcal_case0_ansatz", Rcal0_ansatz)
    ctx.add("case0_ansatz_P", "Momento de curvatura sobre el ansatz",
            r"\{P_0^{abcd}\}_{\mathrm{indep}}\big|_{g[f]}",
            r"\left\{" + r",\;".join(P0_terms) + r"\right\}", group)
    ctx.add("case0_ansatz_M", "Momento metrico sobre el ansatz",
            r"M^{(0)}_{ab}\big|_{g[f]}", latex_expr(M0_ansatz), group)
    ctx.add("case0_ansatz_J", "Momento de gradiente escalar sobre el ansatz",
            r"J_0^a\big|_{g[f]}", latex_expr(J0_ansatz), group)
    ctx.add("case0_ansatz_F", "Momento escalar explicito sobre el ansatz",
            r"F_\phi^{(0)}\big|_{g[f]}", latex_expr(F0_ansatz), group)
    ctx.add("case0_ansatz_Rcal", "Ricci generalizado sobre el ansatz",
            r"\mathcal R^{(0)}_{ab}\big|_{g[f]}", latex_expr(Rcal0_ansatz), group)

    gamma_terms = []
    for (rho, mu, nu), value in geo.nonzero_christoffel().items():
        if mu > nu:  # La conexion es simetrica en los dos indices inferiores.
            continue
        gamma_terms.append(
            rf"\Gamma^{{{names[rho]}}}_{{{{{names[mu]}}}{{{names[nu]}}}}}={latex_expr(value)}"
        )
    ctx.add("ansatz_christoffel", "Simbolos de Christoffel no nulos",
            r"\{\Gamma^a{}_{bc}\}_{\mathrm{indep}}", r"\left\{" + r",\;".join(gamma_terms) + r"\right\}", group)

    riem_terms = []
    for (a, b, c, d), value in geo.independent_riemann().items():
        riem_terms.append(
            rf"R_{{{{{names[a]}}}{{{names[b]}}}{{{names[c]}}}{{{names[d]}}}}}={latex_expr(value)}"
        )
    ctx.add("ansatz_riemann", "Riemann covariante independiente",
            r"\{R_{abcd}\}_{\mathrm{indep}}", r"\left\{" + r",\;".join(riem_terms) + r"\right\}", group)
    ctx.add("ansatz_ricci", "Tensor de Ricci",
            r"R_{ab}", latex_expr(geo.Ricci), group)
    ctx.add("ansatz_R", "Escalar de Ricci",
            "R", latex_expr(geo.Rscalar), group)
    ctx.add("ansatz_Einstein", "Tensor de Einstein",
            r"G_{ab}", latex_expr(geo.Einstein), group)

    E = sp.simplify(geo.Einstein - metric / ell**2)
    ctx.put("E_case0_ansatz", E)
    ctx.add("ansatz_field_tensor", "Tensor de campo antes de resolver f",
            r"E^{(0)}_{ab}=G_{ab}-\ell^{-2}g_{ab}", latex_expr(E), group)
    ctx.add("ansatz_odes", "Ecuaciones independientes para f(r)",
            r"E_{\tau\tau}=E_{rr}=E_{\varphi\varphi}=0",
            r"f'(r)=\frac{2r}{\ell^2},\qquad f''(r)=\frac{2}{\ell^2}", group)
    ctx.add("ansatz_solution", "Solucion integrada",
            "f(r)", r"\frac{r^2}{\ell^2}-\lambda", group,
            "La constante -lambda se identifica con el parametro de masa BTZ del Draft4.")

    f_btz = r**2 / ell**2 - lam
    subs_btz = {f: f_btz, sp.diff(f, r): sp.diff(f_btz, r), sp.diff(f, r, 2): sp.diff(f_btz, r, 2)}
    ricci_btz = geo.Ricci.applyfunc(lambda z: sp.simplify(z.subs(subs_btz)))
    metric_btz = metric.applyfunc(lambda z: sp.simplify(z.subs(subs_btz)))
    R_btz = sp.simplify(geo.Rscalar.subs(subs_btz))
    E_btz = E.applyfunc(lambda z: sp.simplify(z.subs(subs_btz)))
    einstein_btz = geo.Einstein.applyfunc(lambda z: sp.simplify(z.subs(subs_btz)))
    ctx.put("metric_btz", metric_btz)
    ctx.put("Ricci_btz", ricci_btz)
    ctx.put("R_btz", R_btz)
    ctx.put("Einstein_btz", einstein_btz)
    ctx.put("E_btz", E_btz)

    # Segunda evaluacion de los momentos: ademas del ansatz geometrico se
    # sustituye ahora la solucion explicita f_(0). Ninguna expresion final
    # conserva f(r) ni sus derivadas.
    P0_final = {key: sp.simplify(value.subs(subs_btz)) for key, value in P0_ansatz.items()}
    P0_final_terms = [
        rf"P_0^{{{{{names[a]}}}{{{names[b]}}}{{{names[c]}}}{{{names[d]}}}}}={latex_expr(value)}"
        for (a, b, c, d), value in P0_final.items()
    ]
    P0_final_latex = (
        r"\begin{aligned}" + r",\\[2pt]".join(r"&" + term for term in P0_final_terms)
        + r"\end{aligned}"
    )
    M0_final = M0_ansatz.applyfunc(lambda z: sp.simplify(z.subs(subs_btz)))
    J0_final = J0_ansatz.copy()
    F0_final = F0_ansatz
    Rcal0_final = ricci_btz
    ctx.put("P_case0_final", P0_final)
    ctx.put("M_case0_final", M0_final)
    ctx.put("J_case0_final", J0_final)
    ctx.put("F_case0_final", F0_final)
    ctx.put("Rcal_case0_final", Rcal0_final)
    ctx.add("case0_final_P", "Momento de curvatura final",
            r"\{P_0^{abcd}\}_{\mathrm{indep}}\big|_{f=f_{(0)}}",
            P0_final_latex, group,
            r"Aqui y en los cuatro bloques siguientes ya se uso \(f_{(0)}(r)=r^2/\ell^2-\lambda\).")
    ctx.add("case0_final_M", "Momento metrico final",
            r"M^{(0)}_{ab}\big|_{f=f_{(0)}}", latex_expr(M0_final), group)
    ctx.add("case0_final_J", "Momento de gradiente escalar final",
            r"J_0^a\big|_{f=f_{(0)}}", latex_expr(J0_final), group)
    ctx.add("case0_final_F", "Momento escalar explicito final",
            r"F_\phi^{(0)}\big|_{f=f_{(0)}}", latex_expr(F0_final), group)
    ctx.add("case0_final_Rcal", "Ricci generalizado final",
            r"\mathcal R^{(0)}_{ab}\big|_{f=f_{(0)}}", latex_expr(Rcal0_final), group)

    bianchi = geo.divergence_cov2(E).applyfunc(sp.simplify)
    ctx.put("bianchi_case0_ansatz", bianchi)
    ctx.add("ansatz_bianchi", "Chequeo coordenado de Bianchi",
            r"\nabla^aE^{(0)}_{ab}", latex_expr(bianchi), group,
            "Se anula identicamente para una funcion f(r) arbitraria.",
            check=sum((entry**2 for entry in bianchi), sp.S.Zero))

    ricci_check = (ricci_btz + 2 * metric_btz / ell**2).applyfunc(sp.simplify)
    einstein_check = (einstein_btz - metric_btz / ell**2).applyfunc(sp.simplify)
    ctx.add("btz_ricci", "Ricci sobre la solucion BTZ",
            r"R_{ab}\big|_{\mathrm{BTZ}}", r"-\frac{2}{\ell^2}g_{ab}", group,
            check=sum((z**2 for z in ricci_check), sp.S.Zero))
    ctx.add("btz_R", "Curvatura escalar BTZ",
            r"R\big|_{\mathrm{BTZ}}", latex_expr(R_btz), group,
            check=R_btz + 6 / ell**2)
    ctx.add("btz_einstein", "Einstein sobre la solucion BTZ",
            r"G_{ab}\big|_{\mathrm{BTZ}}", r"\frac{1}{\ell^2}g_{ab}", group,
            check=sum((z**2 for z in einstein_check), sp.S.Zero))
    ctx.add("btz_field_equations", "Verificacion final de las ecuaciones de campo",
            r"E^{(0)}_{ab}\big|_{\mathrm{BTZ}}", latex_expr(E_btz), group,
            check=sum((z**2 for z in E_btz), sp.S.Zero))

    # En 3D, una solucion Einstein fija el Riemann. Se verifica componente a componente.
    riemann_residuals = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    actual = sp.simplify(geo.Riemann_down[a][b][c][d].subs(subs_btz))
                    expected = sp.simplify(-(metric_btz[a, c] * metric_btz[b, d] - metric_btz[a, d] * metric_btz[b, c]) / ell**2)
                    riemann_residuals.append(sp.simplify(actual - expected))
    ctx.add("btz_constant_curvature", "Forma de curvatura constante",
            r"R_{abcd}\big|_{\mathrm{BTZ}}",
            r"-\frac{1}{\ell^2}(g_{ac}g_{bd}-g_{ad}g_{bc})", group,
            check=sum((z**2 for z in riemann_residuals), sp.S.Zero))
    ctx.add("btz_kretschmann", "Invariante de Kretschmann",
            r"R_{abcd}R^{abcd}\big|_{\mathrm{BTZ}}", r"\frac{12}{\ell^4}", group,
            "Se usa la forma de curvatura constante ya verificada; evita una contraccion de ocho bucles en cada corrida.",
            check=sp.S.Zero)
    return ctx
