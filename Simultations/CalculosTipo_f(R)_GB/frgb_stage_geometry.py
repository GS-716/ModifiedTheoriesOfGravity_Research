import sympy as sp
from IPython.display import display, Markdown
from sympy.tensor.tensor import tensor_indices


def configure_input(ctx, L_input):
    R, GB, S = ctx.R, ctx.GB, ctx.S

    L_input = sp.sympify(L_input)

    if L_input.has(sp.Derivative):
        raise ValueError(
            "El input debe depender de R y GB, sin derivadas explícitas de los invariantes."
        )

    L_R = sp.simplify(sp.diff(L_input, R))
    L_GB = sp.simplify(sp.diff(L_input, GB))

    ctx.L_input = L_input
    ctx.L_R = L_R
    ctx.L_GB = L_GB

    S.put("L_input", L_input)
    S.put("L_R", L_R)
    S.put("L_GB", L_GB)
    S.put("L_RR", sp.diff(L_input, R, 2))
    S.put("L_RGB", sp.diff(L_input, R, GB))
    S.put("L_GBGB", sp.diff(L_input, GB, 2))

    display(Markdown("### Datos escalares calculados desde el input"))
    for key in ("L_input", "L_R", "L_GB", "L_RR", "L_RGB", "L_GBGB"):
        S.show(key)


def stage_1_build_P(ctx):
    M, g, Riem, S = ctx.M, ctx.g, ctx.Riem, ctx.S
    R, L_R, L_GB = ctx.R, ctx.L_R, ctx.L_GB

    a, b, c, d = tensor_indices("a b c d", M)

    # ------------------------------------------------
    # 1A. dR/dR_abcd por proyección algebraica
    # ------------------------------------------------
    q0, q1, q2, qR = ctx.curvature_projector(
        a, b, c, d, return_steps=True
    )

    S.put("Q_naive", q0)
    S.put("Q_antisym_ab", q1)
    S.put("Q_antisym_ab_cd", q2)
    S.put("dR_dRiemann", qR)

    # ------------------------------------------------
    # 1B. d(Riemann^2)/dR_abcd
    # Producto de dos factores idénticos -> dos aportes iguales.
    # ------------------------------------------------
    S.put("P_Riemann2_term_1", Riem(a, b, c, d))
    S.put("P_Riemann2_term_2", Riem(a, b, c, d))
    S.put(
        "P_Riemann2_abcd",
        S["P_Riemann2_term_1"] + S["P_Riemann2_term_2"]
    )

    # ------------------------------------------------
    # 1C. d(Ricci^2)/dR_abcd
    # delta(R_mn R^mn) = 2 R^mn g^pq delta R_pm qn.
    # El coeficiente bruto se proyecta a las simetrías de Riemann.
    # ------------------------------------------------
    ricci_raw = 2 * g(a, c) * ctx.Ricci_up(b, d)
    ric0, ric1, ric2, pRicci2 = ctx.riemann_project(
        ricci_raw, a, b, c, d, return_steps=True
    )

    S.put("P_Ricci2_raw", ric0)
    S.put("P_Ricci2_antisym_ab", ric1)
    S.put("P_Ricci2_antisym_ab_cd", ric2)
    S.put("P_Ricci2_abcd", pRicci2)

    # ------------------------------------------------
    # 1D. d(R^2)/dR_abcd por regla de la cadena
    # ------------------------------------------------
    S.put("P_R2_abcd", 2 * R * qR)

    # ------------------------------------------------
    # 1E. dGB/dR_abcd y P total del input
    # ------------------------------------------------
    S.put(
        "dGB_dRiemann",
        S["P_Riemann2_abcd"]
        - 4*S["P_Ricci2_abcd"]
        + S["P_R2_abcd"]
    )

    S.put(
        "P_abcd",
        L_R*S["dR_dRiemann"] + L_GB*S["dGB_dRiemann"]
    )

    for key, title in [
        ("Q_naive", "Coeficiente de R antes de imponer simetrías"),
        ("Q_antisym_ab", "dR/dRiemann después de antisimetrizar el primer par"),
        ("Q_antisym_ab_cd", "dR/dRiemann después de antisimetrizar ambos pares"),
        ("dR_dRiemann", "dR/dRiemann después de simetrizar el intercambio de pares"),
        ("P_Riemann2_abcd", "Derivada de Riemann^2 respecto de R_abcd"),
        ("P_Ricci2_raw", "Coeficiente bruto de la variación de Ricci^2"),
        ("P_Ricci2_abcd", "Derivada de Ricci^2 después de proyectar simetrías"),
        ("P_R2_abcd", "Derivada de R^2 por regla de la cadena"),
        ("dGB_dRiemann", "dGB/dR_abcd calculado término a término"),
        ("P_abcd", "P^{abcd} completo calculado desde L_input"),
    ]:
        S.show(key, title)

    ctx.set_P_components([
        ("R", L_R, (a, b, c, d), S["dR_dRiemann"]),
        ("GB", L_GB, (a, b, c, d), S["dGB_dRiemann"]),
    ])

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
    S.check_zero(
        "check_P_pair_exchange",
        ctx.P_up(a, b, c, d) - ctx.P_up(c, d, a, b),
        "P^{abcd}-P^{cdab}=0"
    )

    # Comprobación adicional: la estructura GB también tiene simetrías de Riemann.
    S.check_zero(
        "check_P_GB_antisym_ab",
        S["dGB_dRiemann"] + S["dGB_dRiemann"].xreplace({a:b, b:a}),
        "P_GB^{abcd}+P_GB^{bacd}=0"
    )

    return S.check_zero(
        "check_P_GB_pair_exchange",
        S["dGB_dRiemann"] - S["dGB_dRiemann"].xreplace({a:c, b:d, c:a, d:b}),
        "P_GB^{abcd}-P_GB^{cdab}=0"
    )


def _metric_product_derivative(ctx, metric_pairs, tensor_product, p, q):
    """Deriva un producto de métricas inversas por la regla del producto."""
    result = sp.S.Zero
    for target in range(len(metric_pairs)):
        term = tensor_product
        for pos, (u, v) in enumerate(metric_pairs):
            if pos == target:
                term *= ctx.dginv_dgcov(u, v, p, q)
            else:
                term *= ctx.g(u, v)
        result += term
    return ctx.tsimplify(result)


def stage_2_metric_derivative(ctx):
    M, g, Riem, S = ctx.M, ctx.g, ctx.Riem, ctx.S
    R, L_R, L_GB = ctx.R, ctx.L_R, ctx.L_GB

    p, q = tensor_indices("p q", M)

    # ------------------------------------------------
    # 2A. R como contracción tensorial y dR/dg_ab
    # ------------------------------------------------
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

    # ------------------------------------------------
    # 2B. Riemann^2 y su derivada métrica
    # ------------------------------------------------
    A, B, C, D, E, F, G, H = tensor_indices(
        "A B C D E F G H", M
    )
    riemann2_metrics = [(A,E), (B,F), (C,G), (D,H)]
    riemann2_product = (
        Riem(-A,-B,-C,-D) * Riem(-E,-F,-G,-H)
    )
    S.put(
        "Riemann2_scalar_tensor",
        g(A,E)*g(B,F)*g(C,G)*g(D,H)*riemann2_product
    )
    S.put(
        "dRiemann2_dg_cov",
        _metric_product_derivative(
            ctx, riemann2_metrics, riemann2_product, p, q
        )
    )

    # ------------------------------------------------
    # 2C. Ricci^2 escrito solo con R_abcd y métricas, y derivado
    # ------------------------------------------------
    a, b, c, d, e, f, x, y = tensor_indices(
        "a b c d e f x y", M
    )
    ricci2_metrics = [(a,x), (b,y), (c,d), (e,f)]
    ricci2_product = (
        Riem(-c,-a,-d,-b) * Riem(-e,-x,-f,-y)
    )
    S.put(
        "Ricci2_scalar_tensor",
        g(a,x)*g(b,y)*g(c,d)*g(e,f)*ricci2_product
    )
    S.put(
        "dRicci2_dg_cov",
        _metric_product_derivative(
            ctx, ricci2_metrics, ricci2_product, p, q
        )
    )

    # ------------------------------------------------
    # 2D. R^2 y Gauss--Bonnet
    # ------------------------------------------------
    S.put("dR2_dg_cov", 2*R*S["dR_dg_cov"])
    S.put(
        "GB_scalar_tensor",
        S["Riemann2_scalar_tensor"]
        - 4*S["Ricci2_scalar_tensor"]
        + R**2,
    )
    S.put(
        "dGB_dg_cov",
        S["dRiemann2_dg_cov"]
        - 4*S["dRicci2_dg_cov"]
        + S["dR2_dg_cov"]
    )

    # ------------------------------------------------
    # 2E. P^{ab} total por regla de la cadena del L_input
    # ------------------------------------------------
    S.put(
        "P_metric_ab",
        L_R*S["dR_dg_cov"] + L_GB*S["dGB_dg_cov"]
    )
    S["P_metric_template_indices"] = (p, q)

    for key, title in [
        ("R_scalar_tensor", "R escrito como contracción tensorial real"),
        ("GB_scalar_tensor", "GB escrito a partir de Riemann^2, Ricci^2 y R^2"),
        ("dR_dg_cov_raw", "Derivada métrica de R antes de canonizar"),
        ("dR_dg_cov", "Derivada métrica de R canonizada"),
        ("dRiemann2_dg_cov", "Derivada métrica de Riemann^2"),
        ("dRicci2_dg_cov", "Derivada métrica de Ricci^2"),
        ("dR2_dg_cov", "Derivada métrica de R^2"),
        ("dGB_dg_cov", "Derivada métrica de GB calculada término a término"),
        ("P_metric_ab", "P^{ab} completo calculado directamente desde L_input"),
    ]:
        S.show(key, title)

    return S.check_zero(
        "check_P_metric_symmetry",
        S["P_metric_ab"] - S["P_metric_ab"].xreplace({p:q, q:p}),
        "P^{ab}-P^{ba}=0"
    )


def stage_3_lie_derivative(ctx):
    M, g, Riem, DRiem, xi, Dxi, S = (
        ctx.M, ctx.g, ctx.Riem, ctx.DRiem, ctx.xi, ctx.Dxi, ctx.S
    )
    L_R, L_GB = ctx.L_R, ctx.L_GB

    # ------------------------------------------------
    # 3A. Primera ruta: L es escalar
    # ------------------------------------------------
    m, i, j, k, l = tensor_indices("m i j k l", M)

    S.put(
        "nabla_R_from_Riemann",
        ctx.P_component_up("R", i, j, k, l)
        * DRiem(-m, -i, -j, -k, -l)
    )
    S.put(
        "nabla_GB_from_Riemann",
        ctx.P_component_up("GB", i, j, k, l)
        * DRiem(-m, -i, -j, -k, -l)
    )
    S.put(
        "nabla_L",
        L_R*S["nabla_R_from_Riemann"]
        + L_GB*S["nabla_GB_from_Riemann"]
    )
    S.put("Lie_L_route_1", xi(m) * S["nabla_L"])

    S.show("nabla_R_from_Riemann", "∇_m R calculado desde Riemann")
    S.show("nabla_GB_from_Riemann", "∇_m GB calculado desde dGB/dRiemann")
    S.show("nabla_L", "∇_m L por regla de la cadena en R y GB")
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
    S.show("lie_curv_four_sum", "Suma calculada de los cuatro términos")
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
