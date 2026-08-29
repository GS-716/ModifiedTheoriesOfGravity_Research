"""Caso-1 de Draft4: Einstein-AdS mas un escalar cinetico minimo.

El modulo conserva dos niveles separados:
1. identidades tensoriales para g_ab y phi completamente arbitrarios;
2. evaluacion posterior sobre ds^2=-f dt^2+dr^2/f+r^2 dvarphi^2 y phi=p varphi.
"""

from __future__ import annotations

from itertools import product

import sympy as sp

from mc_core import CouplingContext, latex_expr


def build_case1(ctx: CouplingContext) -> CouplingContext:
    """Construye el Caso-1 sin escoger coordenadas, f(r) ni perfil escalar."""
    group = "Casos I: formulacion tensorial sin ansatz::Caso-1"
    ctx.add(
        "case1_truncation", "Truncamiento del Draft4",
        r"\alpha_{n\ge2}=\beta_{m\ge1}=\beta_0",
        r"0,\qquad \alpha_1\ne0",
        group,
    )
    ctx.add(
        "case1_lagrangian", "Lagrangiano antes de imponer el ansatz",
        r"L_1[g,\phi]",
        r"R+\frac{2}{\ell^2}-\alpha_1X,\qquad X\equiv u^au_a,\quad u_a\equiv\nabla_a\phi",
        group,
    )
    ctx.add(
        "case1_action", "Accion bulk",
        r"I_1",
        r"\frac{1}{16\pi G}\int_Md^3x\sqrt{-g}\left(R+\frac{2}{\ell^2}-\alpha_1X\right)",
        group,
    )
    ctx.add(
        "case1_P", "Momento de curvatura",
        r"P_1^{abcd}=\frac{\partial L_1}{\partial R_{abcd}}",
        r"\frac12\left(g^{ac}g^{bd}-g^{ad}g^{bc}\right)",
        group,
    )
    ctx.add(
        "case1_M", "Momento metrico",
        r"M^{(1)}_{ab}=\frac{\partial L_1}{\partial g^{ab}}",
        r"2R_{ab}-\alpha_1u_au_b",
        group,
        r"La derivada se toma manteniendo fijos \(R_{abcd}\), \(u_a\) y \(\phi\).",
    )
    ctx.add(
        "case1_J", "Momento escalar",
        r"J_1^a=\frac{\partial L_1}{\partial u_a}",
        r"-2\alpha_1u^a",
        group,
    )
    ctx.add(
        "case1_F", "Derivada escalar explicita",
        r"F_\phi^{(1)}=\frac{\partial L_1}{\partial\phi}",
        r"0",
        group,
        "El Caso-1 conserva la simetria de desplazamiento del campo escalar.",
    )
    ctx.add(
        "case1_Rcal", "Ricci generalizado",
        r"\mathcal R^{(1)}_{ab}=P^{(1)}_a{}^{cde}R_{bcde}",
        r"R_{ab}",
        group,
    )
    ctx.add(
        "case1_divP", "Divergencia del momento de curvatura",
        r"\nabla_aP_1^{abcd}",
        r"0",
        group,
        check=sp.S.Zero,
    )
    ctx.add(
        "case1_doubledivP", "Doble divergencia",
        r"-2\nabla^m\nabla^nP^{(1)}_{amnb}",
        r"0",
        group,
        check=sp.S.Zero,
    )
    ctx.add(
        "case1_stress", "Tensor cinetico del escalar",
        r"T^{(\phi)}_{ab}",
        r"u_au_b-\frac12g_{ab}X",
        group,
    )
    ctx.add(
        "case1_Eab", "Ecuacion metrica sin ansatz",
        r"E^{(1)}_{ab}",
        r"G_{ab}-\frac{1}{\ell^2}g_{ab}-\alpha_1\left(u_au_b-\frac12g_{ab}X\right)",
        group,
    )
    ctx.add(
        "case1_Ephi", "Ecuacion escalar sin ansatz",
        r"E^{(1)}_\phi=F_\phi^{(1)}-\nabla_aJ_1^a",
        r"2\alpha_1\Box\phi",
        group,
    )
    ctx.add(
        "case1_bianchi", "Bianchi-Noether off-shell del Caso-1",
        r"2\nabla^aE^{(1)}_{ab}+E^{(1)}_\phi u_b",
        r"0",
        group,
        check=sp.S.Zero,
    )
    ctx.add(
        "case1_boundary_scalar", "Termino de frontera escalar",
        r"n_a\Theta^a_{(\phi)}",
        r"-2\alpha_1(n^au_a)\,\delta\phi",
        group,
        r"Se anula genericamente con Dirichlet \(\delta\phi|_{\partial M}=0\).",
    )
    ctx.add(
        "case1_boundary_action", "Accion mejorada para Dirichlet",
        r"I^{(1)}_{\mathrm{tot}}",
        r"I_1+\frac{1}{8\pi G}\int_{\partial M}d^2x\sqrt{|h|}\,K",
        group,
        "No se requiere un contratermino escalar para el principio variacional; la renormalizacion on-shell de la rama lenta es un problema adicional.",
    )
    return ctx


def evaluate_case1_ansatz(ctx: CouplingContext) -> CouplingContext:
    """Sustituye phi=p*varphi, resuelve f y verifica todas las ecuaciones."""
    group = "Casos II: sustitucion completa del ansatz::Caso-1"
    geo = ctx.objects["geometry"]
    tau, r, varphi = ctx.objects["coordinates"]
    ell, f = ctx.objects["ell"], ctx.objects["f"]

    alpha1 = sp.symbols("alpha_1", real=True, nonzero=True)
    p = sp.symbols("p", real=True)
    r0 = sp.symbols("r_0", positive=True)
    lam = ctx.objects["lambda"]
    phi = p * varphi

    u_cov = geo.scalar_gradient_cov(phi)
    u_up = (geo.g_inv * u_cov).applyfunc(sp.simplify)
    X = sp.simplify((u_cov.T * geo.g_inv * u_cov)[0])
    stress = (u_cov * u_cov.T - sp.Rational(1, 2) * geo.g * X).applyfunc(sp.simplify)
    box_phi = geo.scalar_laplacian(phi)
    Ephi = sp.simplify(2 * alpha1 * box_phi)
    E = (geo.Einstein - geo.g / ell**2 - alpha1 * stress).applyfunc(sp.simplify)

    ctx.put("alpha1", alpha1)
    ctx.put("p", p)
    ctx.put("r0", r0)
    ctx.put("phi_case1", phi)
    ctx.put("u_case1_cov", u_cov)
    ctx.put("u_case1_up", u_up)
    ctx.put("X_case1_ansatz", X)
    ctx.put("T_case1_ansatz", stress)
    ctx.put("box_phi_case1_ansatz", box_phi)
    ctx.put("Ephi_case1_ansatz", Ephi)
    ctx.put("E_case1_ansatz", E)

    ctx.add(
        "case1_ansatz_data", "Sustitucion simultanea de metrica y escalar",
        r"(ds^2,\phi)",
        r"\left(-f(r)d\tau^2+\frac{dr^2}{f(r)}+r^2d\varphi^2,\;p\varphi\right)",
        group,
    )
    ctx.add(
        "case1_ansatz_gradient", "Gradiente escalar y contraccion cinetica",
        r"(u_a,u^a,X)",
        rf"\left({latex_expr(u_cov)},\,{latex_expr(u_up)},\,{latex_expr(X)}\right)",
        group,
    )

    # Se muestran por separado todos los momentos una vez sustituidos tanto
    # la metrica g[f] como el perfil angular phi=p*varphi.
    names = (r"\tau", "r", r"\varphi")
    P1_ansatz = geo.independent_einstein_hilbert_momentum()
    P1_terms = [
        rf"P_1^{{{{{names[a]}}}{{{names[b]}}}{{{names[c]}}}{{{names[d]}}}}}={latex_expr(value)}"
        for (a, b, c, d), value in P1_ansatz.items()
    ]
    M1_ansatz = (2 * geo.Ricci - alpha1 * u_cov * u_cov.T).applyfunc(sp.simplify)
    J1_ansatz = (-2 * alpha1 * u_up).applyfunc(sp.simplify)
    F1_ansatz = sp.S.Zero
    Rcal1_ansatz = geo.Ricci.applyfunc(sp.simplify)
    ctx.put("P_case1_ansatz", P1_ansatz)
    ctx.put("M_case1_ansatz", M1_ansatz)
    ctx.put("J_case1_ansatz", J1_ansatz)
    ctx.put("F_case1_ansatz", F1_ansatz)
    ctx.put("Rcal_case1_ansatz", Rcal1_ansatz)
    ctx.add(
        "case1_ansatz_P", "Momento de curvatura sobre el ansatz",
        r"\{P_1^{abcd}\}_{\mathrm{indep}}\big|_{g[f],\phi=p\varphi}",
        r"\left\{" + r",\;".join(P1_terms) + r"\right\}", group,
    )
    ctx.add(
        "case1_ansatz_M", "Momento metrico sobre el ansatz",
        r"M^{(1)}_{ab}\big|_{g[f],\phi=p\varphi}", latex_expr(M1_ansatz), group,
    )
    ctx.add(
        "case1_ansatz_J", "Momento de gradiente escalar sobre el ansatz",
        r"J_1^a\big|_{g[f],\phi=p\varphi}", latex_expr(J1_ansatz), group,
    )
    ctx.add(
        "case1_ansatz_F", "Momento escalar explicito sobre el ansatz",
        r"F_\phi^{(1)}\big|_{g[f],\phi=p\varphi}", latex_expr(F1_ansatz), group,
    )
    ctx.add(
        "case1_ansatz_Rcal", "Ricci generalizado sobre el ansatz",
        r"\mathcal R^{(1)}_{ab}\big|_{g[f],\phi=p\varphi}", latex_expr(Rcal1_ansatz), group,
    )

    ctx.add(
        "case1_ansatz_stress", "Tensor escalar evaluado",
        r"T^{(\phi)}_{ab}\big|_{\mathrm{ansatz}}",
        latex_expr(stress),
        group,
    )
    ctx.add(
        "case1_ansatz_box", "Ecuacion de Klein-Gordon sobre el ansatz",
        r"\Box\phi\big|_{\phi=p\varphi}",
        latex_expr(box_phi),
        group,
        "El perfil angular satisface la ecuacion escalar para cualquier f(r).",
        check=box_phi,
    )
    ctx.add(
        "case1_ansatz_E", "Tensor metrico antes de resolver f(r)",
        r"E^{(1)}_{ab}\big|_{g[f],\phi=p\varphi}",
        latex_expr(E),
        group,
    )

    first_order_rhs = sp.simplify(2 * r / ell**2 - alpha1 * p**2 / r)
    integrated_raw = sp.integrate(first_order_rhs, r)
    expected_integrated = r**2 / ell**2 - alpha1 * p**2 * sp.log(r)
    ctx.put("case1_fprime_rhs", first_order_rhs)
    ctx.put("case1_integrated_raw", integrated_raw)
    ctx.add(
        "case1_ansatz_odes", "Ecuaciones independientes para f(r)",
        r"E_{\tau\tau}=E_{rr}=E_{\varphi\varphi}=0",
        r"f'(r)=\frac{2r}{\ell^2}-\frac{\alpha_1p^2}{r},\qquad f''(r)=\frac{2}{\ell^2}+\frac{\alpha_1p^2}{r^2}",
        group,
    )
    ctx.add(
        "case1_integrate_f", "Integracion simbolica de la ecuacion radial",
        r"\int dr\left(\frac{2r}{\ell^2}-\frac{\alpha_1p^2}{r}\right)",
        latex_expr(integrated_raw),
        group,
        check=sp.simplify(integrated_raw - expected_integrated),
    )

    f_solution = r**2 / ell**2 - lam - alpha1 * p**2 * sp.log(r / r0)
    ctx.put("f_case1_solution", f_solution)
    ctx.add(
        "case1_f_solution", "Rama logaritmica del Draft4",
        r"f_{(1)}(r)",
        latex_expr(f_solution),
        group,
        check=sp.simplify(sp.diff(f_solution, r) - first_order_rhs),
    )

    substitutions = {
        f: f_solution,
        sp.diff(f, r): sp.diff(f_solution, r),
        sp.diff(f, r, 2): sp.diff(f_solution, r, 2),
    }
    metric_solution = geo.g.applyfunc(lambda z: sp.simplify(z.subs(substitutions)))
    ricci_solution = geo.Ricci.applyfunc(lambda z: sp.simplify(z.subs(substitutions)))
    R_solution = sp.simplify(geo.Rscalar.subs(substitutions))
    einstein_solution = geo.Einstein.applyfunc(lambda z: sp.simplify(z.subs(substitutions)))
    E_solution = E.applyfunc(lambda z: sp.simplify(z.subs(substitutions)))
    bianchi = geo.divergence_cov2(E).applyfunc(sp.simplify)
    noether = (2 * bianchi + Ephi * u_cov).applyfunc(sp.simplify)

    ricci_squared = sp.simplify(sum(
        geo.g_inv[a, c] * geo.g_inv[b, d] * geo.Ricci[a, b] * geo.Ricci[c, d]
        for a, b, c, d in product(range(3), repeat=4)
    ).subs(substitutions))
    kretschmann = sp.factor(4 * ricci_squared - R_solution**2)

    ctx.put("metric_case1_solution", metric_solution)
    ctx.put("Ricci_case1_solution", ricci_solution)
    ctx.put("R_case1_solution", R_solution)
    ctx.put("Einstein_case1_solution", einstein_solution)
    ctx.put("E_case1_solution", E_solution)
    ctx.put("bianchi_case1_ansatz", bianchi)
    ctx.put("noether_case1_ansatz", noether)
    ctx.put("Ricci2_case1_solution", ricci_squared)
    ctx.put("K_case1_solution", kretschmann)

    # Momentos completamente on-shell respecto del ansatz: aqui tambien se
    # reemplazan f(r), f'(r) y f''(r) por la rama logaritmica f_(1).
    P1_final = {
        key: sp.simplify(value.subs(substitutions))
        for key, value in P1_ansatz.items()
    }
    P1_final_terms = [
        rf"P_1^{{{{{names[a]}}}{{{names[b]}}}{{{names[c]}}}{{{names[d]}}}}}={latex_expr(value)}"
        for (a, b, c, d), value in P1_final.items()
    ]
    P1_final_latex = (
        r"\begin{aligned}" + r",\\[2pt]".join(r"&" + term for term in P1_final_terms)
        + r"\end{aligned}"
    )
    M1_final = M1_ansatz.applyfunc(lambda z: sp.simplify(z.subs(substitutions)))
    J1_final = J1_ansatz.applyfunc(lambda z: sp.simplify(z.subs(substitutions)))
    F1_final = F1_ansatz
    Rcal1_final = ricci_solution
    ctx.put("P_case1_final", P1_final)
    ctx.put("M_case1_final", M1_final)
    ctx.put("J_case1_final", J1_final)
    ctx.put("F_case1_final", F1_final)
    ctx.put("Rcal_case1_final", Rcal1_final)
    ctx.add(
        "case1_final_P", "Momento de curvatura final",
        r"\{P_1^{abcd}\}_{\mathrm{indep}}\big|_{f=f_{(1)},\phi=p\varphi}",
        P1_final_latex, group,
        r"Aqui y en los cuatro bloques siguientes ya se uso la rama logaritmica explicita \(f_{(1)}(r)\).",
    )
    ctx.add(
        "case1_final_M", "Momento metrico final",
        r"M^{(1)}_{ab}\big|_{f=f_{(1)},\phi=p\varphi}", latex_expr(M1_final), group,
    )
    ctx.add(
        "case1_final_J", "Momento de gradiente escalar final",
        r"J_1^a\big|_{f=f_{(1)},\phi=p\varphi}", latex_expr(J1_final), group,
    )
    ctx.add(
        "case1_final_F", "Momento escalar explicito final",
        r"F_\phi^{(1)}\big|_{f=f_{(1)},\phi=p\varphi}", latex_expr(F1_final), group,
    )
    ctx.add(
        "case1_final_Rcal", "Ricci generalizado final",
        r"\mathcal R^{(1)}_{ab}\big|_{f=f_{(1)},\phi=p\varphi}", latex_expr(Rcal1_final), group,
    )

    ctx.add(
        "case1_bianchi_ansatz", "Bianchi metrico antes de resolver f(r)",
        r"\nabla^aE^{(1)}_{ab}\big|_{g[f],\phi=p\varphi}",
        latex_expr(bianchi),
        group,
        check=sum((entry**2 for entry in bianchi), sp.S.Zero),
    )
    ctx.add(
        "case1_noether_ansatz", "Identidad Bianchi-Noether completa",
        r"2\nabla^aE^{(1)}_{ab}+E^{(1)}_\phi u_b",
        latex_expr(noether),
        group,
        check=sum((entry**2 for entry in noether), sp.S.Zero),
    )
    ctx.add(
        "case1_field_solution", "Verificacion final de la ecuacion metrica",
        r"E^{(1)}_{ab}\big|_{f=f_{(1)},\phi=p\varphi}",
        latex_expr(E_solution),
        group,
        check=sum((entry**2 for entry in E_solution), sp.S.Zero),
    )
    ctx.add(
        "case1_scalar_solution", "Verificacion final de la ecuacion escalar",
        r"E^{(1)}_\phi\big|_{f=f_{(1)},\phi=p\varphi}",
        latex_expr(Ephi),
        group,
        check=Ephi,
    )
    ctx.add(
        "case1_R_solution", "Escalar de Ricci de la rama logaritmica",
        r"R\big|_{f=f_{(1)}}",
        latex_expr(R_solution),
        group,
        check=sp.simplify(R_solution + 6 / ell**2 - alpha1 * p**2 / r**2),
    )
    ctx.add(
        "case1_Ricci_solution", "Tensor de Ricci de la rama logaritmica",
        r"R_{ab}\big|_{f=f_{(1)}}",
        latex_expr(ricci_solution),
        group,
    )
    ctx.add(
        "case1_Ricci2_solution", "Invariante cuadratico de Ricci",
        r"R_{ab}R^{ab}\big|_{f=f_{(1)}}",
        latex_expr(sp.factor(ricci_squared)),
        group,
    )
    ctx.add(
        "case1_K_solution", "Invariante de Kretschmann en tres dimensiones",
        r"R_{abcd}R^{abcd}\big|_{f=f_{(1)}}",
        latex_expr(kretschmann),
        group,
        r"Se usa la identidad tridimensional \(R_{abcd}R^{abcd}=4R_{ab}R^{ab}-R^2\).",
    )
    ctx.add(
        "case1_boundary_ansatz", "Flujo escalar normal al borde radial",
        r"n^au_a\big|_{\phi=p\varphi,\,r=\mathrm{cte}}",
        r"0",
        group,
        "El gradiente es puramente tangencial; el termino de frontera escalar se anula sobre este fondo.",
        check=sp.S.Zero,
    )
    return ctx
