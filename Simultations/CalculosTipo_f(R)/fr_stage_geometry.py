
import sympy as sp
from IPython.display import display, Markdown
from sympy.tensor.tensor import tensor_indices


def configure_input(ctx, L_input):
    R = ctx.R
    S = ctx.S

    L_input = sp.sympify(L_input)

    if L_input.has(sp.Derivative):
        raise ValueError("El input debe depender de R, sin derivadas explícitas de R.")

    f1 = sp.simplify(sp.diff(L_input, R))
    f2 = sp.simplify(sp.diff(L_input, R, 2))
    f3 = sp.simplify(sp.diff(L_input, R, 3))

    ctx.L_input = L_input
    ctx.f1 = f1
    ctx.f2 = f2
    ctx.f3 = f3

    S.put("L_input", L_input)
    S.put("f_R", f1)
    S.put("f_RR", f2)
    S.put("f_RRR", f3)

    display(Markdown("### Datos escalares calculados desde el input"))
    for key in ("L_input", "f_R", "f_RR", "f_RRR"):
        S.show(key)


def stage_1_build_P(ctx):
    M, S = ctx.M, ctx.S
    f1 = ctx.f1

    a, b, c, d = tensor_indices("a b c d", M)

    q0, q1, q2, q3 = ctx.curvature_projector(
        a, b, c, d, return_steps=True
    )

    S.put("Q_naive", q0)
    S.put("Q_antisym_ab", q1)
    S.put("Q_antisym_ab_cd", q2)
    S.put("dR_dRiemann", q3)

    S.put("P_abcd", f1 * q3)

    for key, title in [
        ("Q_naive", "Coeficiente antes de imponer simetrías"),
        ("Q_antisym_ab", "Después de antisimetrizar el primer par"),
        ("Q_antisym_ab_cd", "Después de antisimetrizar ambos pares"),
        ("dR_dRiemann", "Después de simetrizar el intercambio de pares"),
        ("P_abcd", "P^{abcd} calculado desde L_input"),
    ]:
        S.show(key, title)

    ctx.set_P_template((a, b, c, d), q3)

    S.check_zero(
        "check_P_antisym_ab",
        ctx.P_up(a, b, c, d) + ctx.P_up(b, a, c, d),
        "P^{abcd}+P^{bacd}=0"
    )
    S.check_zero(
        "check_P_antisym_cd",
        ctx.P_up(a, b, c, d) + ctx.P_up(a, b, d, c),
        "P^{abcd}+P^{abdc}=0"
    )
    return S.check_zero(
        "check_P_pair_exchange",
        ctx.P_up(a, b, c, d) - ctx.P_up(c, d, a, b),
        "P^{abcd}-P^{cdab}=0"
    )


def stage_2_metric_derivative(ctx):
    M, g, Riem, S = ctx.M, ctx.g, ctx.Riem, ctx.S
    f1 = ctx.f1

    p, q = tensor_indices("p q", M)
    i, j, k, l = tensor_indices("i j k l", M)

    S.put(
        "R_scalar_tensor",
        g(i, k) * g(j, l) * Riem(-i, -j, -k, -l)
    )

    S.put(
        "dR_dg_cov_raw",
        ctx.dginv_dgcov(i, k, p, q) * g(j, l) * Riem(-i, -j, -k, -l)
        + g(i, k) * ctx.dginv_dgcov(j, l, p, q) * Riem(-i, -j, -k, -l),
        simplify=False
    )

    S.put("dR_dg_cov", S["dR_dg_cov_raw"])
    S.put("P_metric_ab", f1 * S["dR_dg_cov"])

    S.show("R_scalar_tensor", "R escrito como contracción tensorial real")
    S.show("dR_dg_cov_raw", "Derivada métrica antes de canonizar")
    S.show("dR_dg_cov", "Derivada métrica canonizada")
    S.show("P_metric_ab", "P^{ab} calculado directamente desde L_input")

    S.check_zero(
        "check_P_metric_symmetry",
        S["P_metric_ab"] - S["P_metric_ab"].xreplace({p:q, q:p}),
        "P^{ab}-P^{ba}=0"
    )

    ctx.set_P_metric_template((p, q), S["P_metric_ab"])


def stage_3_lie_derivative(ctx):
    M, g, Riem, DRiem, xi, Dxi, S = (
        ctx.M, ctx.g, ctx.Riem, ctx.DRiem, ctx.xi, ctx.Dxi, ctx.S
    )
    f1 = ctx.f1

    # ------------------------------------------------
    # 3A. Primera ruta: L es escalar
    # ------------------------------------------------
    m, i, j, k, l = tensor_indices("m i j k l", M)

    S.put(
        "nabla_R_from_Riemann",
        ctx.curvature_projector(i, j, k, l) * DRiem(-m, -i, -j, -k, -l)
    )

    S.put(
        "nabla_L",
        f1 * S["nabla_R_from_Riemann"]
    )

    S.put(
        "Lie_L_route_1",
        xi(m) * S["nabla_L"]
    )

    S.show("nabla_R_from_Riemann", "∇_m R calculado desde la contracción de Riemann")
    S.show("nabla_L", "∇_m L calculado por regla de la cadena")
    S.show("Lie_L_route_1", "Primera ruta para la derivada de Lie")

    # ------------------------------------------------
    # 3B. Derivada de Lie de la métrica
    # ------------------------------------------------
    a, b, m = tensor_indices("a b m", M)

    S.put(
        "Lie_metric_ab",
        g(-m, -b) * Dxi(-a, m)
        + g(-a, -m) * Dxi(-b, m)
    )

    S.put(
        "Lie_metric_contraction",
        ctx.P_metric_up(a, b) * S["Lie_metric_ab"]
    )

    S.show("Lie_metric_ab", "L_xi g_ab construido tensorialmente")
    S.show("Lie_metric_contraction", "P^{ab} L_xi g_ab canonizado")

    # ------------------------------------------------
    # 3C. Derivada de Lie del Riemann: cinco términos reales
    # ------------------------------------------------
    i, j, k, l, m = tensor_indices("i j k l m", M)

    S.put(
        "lie_curv_transport",
        ctx.P_up(i, j, k, l) * xi(m) * DRiem(-m, -i, -j, -k, -l)
    )

    curv_terms = [
        ctx.P_up(i, j, k, l) * Riem(-m, -j, -k, -l) * Dxi(-i, m),
        ctx.P_up(i, j, k, l) * Riem(-i, -m, -k, -l) * Dxi(-j, m),
        ctx.P_up(i, j, k, l) * Riem(-i, -j, -m, -l) * Dxi(-k, m),
        ctx.P_up(i, j, k, l) * Riem(-i, -j, -k, -m) * Dxi(-l, m),
    ]

    for nterm, term in enumerate(curv_terms, 1):
        S.put(f"lie_curv_term_{nterm}_raw", term, simplify=False)
        S.put(f"lie_curv_term_{nterm}", term)
        S.show(
            f"lie_curv_term_{nterm}_raw",
            f"Término de curvatura {nterm}, antes de canonizar"
        )
        S.show(
            f"lie_curv_term_{nterm}",
            f"Término de curvatura {nterm}, canonizado"
        )

    for nterm in range(2, 5):
        S.check_zero(
            f"check_lie_curv_term_{nterm}_equals_1",
            S[f"lie_curv_term_{nterm}"] - S["lie_curv_term_1"],
            f"T_{nterm}-T_1=0"
        )

    S.put("lie_curv_four_sum", sum(curv_terms, sp.S.Zero))
    S.put(
        "Lie_Riemann_contraction",
        S["lie_curv_transport"] + S["lie_curv_four_sum"]
    )

    S.show("lie_curv_four_sum", "Suma realmente calculada de los cuatro términos")
    S.show("Lie_Riemann_contraction", "P^{ijkl} L_xi R_{ijkl}")

    # ------------------------------------------------
    # 3D. Segunda ruta completa y comparación
    # ------------------------------------------------
    S.put(
        "Lie_L_route_2",
        S["Lie_metric_contraction"] + S["Lie_Riemann_contraction"]
    )

    S.show("Lie_L_route_2", "Segunda ruta completa")

    return S.check_zero(
        "check_two_Lie_routes",
        S["Lie_L_route_2"] - S["Lie_L_route_1"],
        "Ruta 2 - Ruta 1 = 0"
    )


def stage_4_Rcal(ctx):
    M, g, Riem, S = ctx.M, ctx.g, ctx.Riem, ctx.S

    a, b, i, j, k = tensor_indices("a b i j k", M)

    S.put(
        "Rcal_up_ab",
        ctx.P_up(a, i, j, k) * Riem(b, -i, -j, -k)
    )

    p, q = tensor_indices("p q", M)
    Rcal_up_pq = S["Rcal_up_ab"].xreplace({a:p, b:q})
    S.put(
        "Rcal_down_ab",
        g(-a, -p) * g(-b, -q) * Rcal_up_pq
    )

    S.show("Rcal_up_ab", "Rcal^{ab} calculado por contracción")
    S.show("Rcal_down_ab", "Rcal_ab calculado bajando índices")

    S.check_zero(
        "check_main_identity",
        ctx.P_metric_up(a, b) + 2*S["Rcal_up_ab"],
        "P^{ab}+2 Rcal^{ab}=0"
    )

    return S.check_zero(
        "check_Rcal_symmetry",
        S["Rcal_up_ab"] - S["Rcal_up_ab"].xreplace({a:b, b:a}),
        "Rcal^{ab}-Rcal^{ba}=0"
    )
