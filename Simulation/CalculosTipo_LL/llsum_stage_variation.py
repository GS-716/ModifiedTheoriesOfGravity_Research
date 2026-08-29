import sympy as sp
from sympy.tensor.tensor import TensorHead, TensorSymmetry, tensor_indices


def stage_3_lie_derivative(ctx):
    M, g, Riem, DRiem, xi, Dxi, S = (
        ctx.M, ctx.g, ctx.Riem, ctx.DRiem, ctx.xi, ctx.Dxi, ctx.S
    )

    m, i, j, k, l = tensor_indices("m i j k l", M)

    S.put(
        "nabla_L",
        ctx.P_up(i,j,k,l) * DRiem(-m,-i,-j,-k,-l),
    )
    S.put("Lie_L_route_1", xi(m)*S["nabla_L"])

    S.show("nabla_L", "∇_m L calculado desde el P total construido")
    S.show("Lie_L_route_1", "Primera ruta: el Lagrangiano total es escalar")

    a, b, m = tensor_indices("a b m", M)
    S.put(
        "Lie_metric_ab",
        g(-m,-b)*Dxi(-a,m) + g(-a,-m)*Dxi(-b,m),
    )
    S.put(
        "Lie_metric_contraction",
        ctx.P_metric_up(a,b)*S["Lie_metric_ab"],
    )

    S.show("Lie_metric_ab", "L_xi g_ab")
    S.show("Lie_metric_contraction", "P^{ab} L_xi g_ab")

    i, j, k, l, m = tensor_indices("i j k l m", M)
    S.put(
        "lie_curv_transport",
        ctx.P_up(i,j,k,l)*xi(m)*DRiem(-m,-i,-j,-k,-l),
    )

    curv_terms = [
        ctx.P_up(i,j,k,l)*Riem(-m,-j,-k,-l)*Dxi(-i,m),
        ctx.P_up(i,j,k,l)*Riem(-i,-m,-k,-l)*Dxi(-j,m),
        ctx.P_up(i,j,k,l)*Riem(-i,-j,-m,-l)*Dxi(-k,m),
        ctx.P_up(i,j,k,l)*Riem(-i,-j,-k,-m)*Dxi(-l,m),
    ]

    for nterm, term in enumerate(curv_terms, 1):
        S.put(f"lie_curv_term_{nterm}_raw", term, simplify=False)
        S.put(f"lie_curv_term_{nterm}", term)
        S.show(f"lie_curv_term_{nterm}", f"Término de curvatura {nterm}, canonizado")

    for nterm in range(2,5):
        S.check_zero(
            f"check_lie_curv_term_{nterm}_equals_1",
            S[f"lie_curv_term_{nterm}"] - S["lie_curv_term_1"],
            f"T_{nterm}-T_1=0",
        )

    S.put("lie_curv_four_sum", sum(curv_terms, sp.S.Zero))

    # La primera de las cuatro contribuciones es, por definición de Rcal,
    # Rcal^{ab} nabla_a xi_b. Como las cuatro ya fueron calculadas y
    # verificadas iguales, sustituimos la abreviatura derivada.
    a, b = tensor_indices("a b", M)
    S.put(
        "lie_curv_four_sum_as_Rcal",
        4*ctx.Rcal_up(a,b)*Dxi(-a,-b),
    )
    S.put(
        "Lie_Riemann_contraction",
        S["lie_curv_transport"] + S["lie_curv_four_sum_as_Rcal"],
    )
    S.put(
        "Lie_L_route_2",
        S["Lie_metric_contraction"] + S["Lie_Riemann_contraction"],
    )

    S.show("lie_curv_four_sum", "Suma tensorial cruda de los cuatro términos")
    S.show("lie_curv_four_sum_as_Rcal", "Misma suma usando la abreviatura Rcal ya derivada")
    S.show("Lie_L_route_2", "Segunda ruta completa")

    S.check_zero(
        "check_metric_lie_cancels_curvature_indices",
        S["Lie_metric_contraction"] + S["lie_curv_four_sum_as_Rcal"],
        "término métrico + cuatro términos de índice = 0",
    )

    return S.check_zero(
        "check_two_Lie_routes",
        S["Lie_L_route_2"] - S["Lie_L_route_1"],
        "Ruta 2 - Ruta 1 = 0",
    )


def stage_4_Rcal(ctx):
    M, g, S = ctx.M, ctx.g, ctx.S

    a, b = tensor_indices("a b", M)
    S.put("Rcal_contraction_raw", ctx.Rcal_contraction_raw(a,b))
    S.put("Rcal_up_ab", ctx.Rcal_up(a,b))

    S.put("Rcal_down_ab", ctx.Rcal_down(a,b))

    S.show("Rcal_contraction_raw", "Contracción cruda P^{aijk}R^b{}_{ijk}")
    S.show("Rcal_up_ab", "Rcal^{ab}: abreviatura registrada de la contracción")
    S.show("Rcal_down_ab", "Rcal_ab")

    S.check_zero(
        "check_main_identity_tensor",
        ctx.P_metric_up(a,b) + 2*S["Rcal_up_ab"],
        "P^{ab}+2 Rcal^{ab}=0",
    )

    return S.check_zero(
        "check_Rcal_symmetry",
        S["Rcal_up_ab"] - S["Rcal_up_ab"].xreplace({a:b,b:a}),
        "Rcal^{ab}-Rcal^{ba}=0",
    )


def stage_5_vary_density(ctx):
    M, g, H, dRiem, sqrtg, S = (
        ctx.M, ctx.g, ctx.H, ctx.dRiem, ctx.sqrtg, ctx.S
    )
    L_input = ctx.L_total_expr

    a, b, p, q = tensor_indices("a b p q", M)
    S.put(
        "P_metric_contravariant_variable_ab",
        -g(-a,-p)*g(-b,-q)*ctx.P_metric_up(p,q),
    )
    S.put(
        "delta_L_metric",
        S["P_metric_contravariant_variable_ab"]*H(a,b),
    )

    a, b, c, d = tensor_indices("a b c d", M)
    S.put(
        "delta_L_curvature_unsplit",
        ctx.P_up(a,b,c,d)*dRiem(-a,-b,-c,-d),
    )
    S.put(
        "delta_L_total_unsplit",
        S["delta_L_metric"] + S["delta_L_curvature_unsplit"],
    )

    a, b = tensor_indices("a b", M)
    S.put(
        "delta_sqrt_minus_g",
        -sp.Rational(1,2)*sqrtg*g(-a,-b)*H(a,b),
    )
    S.put(
        "delta_density_unsplit",
        S["delta_sqrt_minus_g"]*L_input
        + sqrtg*S["delta_L_total_unsplit"],
    )

    for key, title in [
        ("P_metric_contravariant_variable_ab", "∂L/∂g^{ab}"),
        ("delta_L_metric", "Parte métrica de δL"),
        ("delta_L_curvature_unsplit", "Parte de curvatura de δL"),
        ("delta_L_total_unsplit", "δL antes de Palatini"),
        ("delta_sqrt_minus_g", "δ√(-g)"),
        ("delta_density_unsplit", "δ(√(-g)L) antes de separar δR"),
    ]:
        S.show(key, title)

    return S.check_zero(
        "check_metric_variation_equals_2Rcal",
        S["P_metric_contravariant_variable_ab"] - 2*ctx.Rcal_down(a,b),
        "∂L/∂g^{ab} - 2 Rcal_ab = 0",
    )


def stage_6_split_delta_R(ctx):
    M, g, Riem, H, S = ctx.M, ctx.g, ctx.Riem, ctx.H, ctx.S

    a, b, c, d, e = tensor_indices("a b c d e", M)
    S.put(
        "delta_R_split_metric_piece_raw",
        ctx.P_up(a,b,c,d)*ctx.h_from_H(a,e)*Riem(e,-b,-c,-d),
    )

    # Usamos la abreviatura Rcal derivada de esa contracción.
    p, q = tensor_indices("p q", M)
    S.put(
        "delta_R_split_metric_piece",
        -ctx.Rcal_down(p,q)*H(p,q),
    )

    dRm = TensorHead(
        r"\delta\mathrm{R}_{\mathrm{mix}}",
        [M]*4,
        TensorSymmetry.no_symmetry(4),
    )
    S.put(
        "delta_R_split_connection_piece",
        ctx.P_up(a,b,c,d)*g(-a,-e)*dRm(e,-b,-c,-d),
    )

    S.show("delta_R_split_metric_piece_raw", "Primera pieza cruda de P δR")
    S.show("delta_R_split_metric_piece", "Primera pieza usando la abreviatura Rcal")
    S.show("delta_R_split_connection_piece", "Segunda pieza de P δR")

    a, b = tensor_indices("a b", M)
    return S.check_zero(
        "check_split_metric_piece",
        S["delta_R_split_metric_piece"] + ctx.Rcal_down(a,b)*H(a,b),
        "P δg R + Rcal_ab H^ab = 0",
    )


def stage_7_palatini(ctx):
    M, g, DGamma, S = ctx.M, ctx.g, ctx.DGamma, ctx.S

    a, b, c, d, e = tensor_indices("a b c d e", M)
    palatini_1 = ctx.P_up(a,b,c,d)*g(-a,-e)*DGamma(e,-c,-d,-b)
    palatini_2 = -ctx.P_up(a,b,c,d)*g(-a,-e)*DGamma(e,-d,-c,-b)

    S.put("palatini_term_1_raw", palatini_1, simplify=False)
    S.put("palatini_term_2_raw", palatini_2, simplify=False)
    S.put("palatini_term_1", palatini_1)
    S.put("palatini_term_2", palatini_2)
    S.put("palatini_sum", palatini_1 + palatini_2)

    for key in (
        "palatini_term_1",
        "palatini_term_2",
        "palatini_sum",
    ):
        S.show(key)

    S.check_zero(
        "check_palatini_two_terms_equal",
        S["palatini_term_2"] - S["palatini_term_1"],
        "segundo término - primer término = 0",
    )
    return S.check_zero(
        "check_palatini_sum_is_twice",
        S["palatini_sum"] - 2*S["palatini_term_1"],
        "suma - 2×primer término = 0",
    )


def stage_8_substitute_dGamma(ctx):
    M, DDh, S = ctx.M, ctx.DDh, ctx.S

    a, b, c, d, e = tensor_indices("a b c d e", M)
    expanded = 2*ctx.P_up(a,b,c,d)*ctx.g(-a,-e)*ctx.DGamma_from_h(e,c,d,b)

    S.put("after_dGamma_full_raw", expanded, simplify=False)
    S.put("after_dGamma_full", expanded)

    i = tensor_indices("i", M)
    dg1 = ctx.P_up(i,b,c,d)*DDh(-c,-d,-b,-i)
    dg2 = ctx.P_up(i,b,c,d)*DDh(-c,-b,-d,-i)
    dg3 = -ctx.P_up(i,b,c,d)*DDh(-c,-i,-d,-b)

    for nterm, term in enumerate((dg1,dg2,dg3), 1):
        S.put(f"dGamma_piece_{nterm}_raw", term, simplify=False)
        S.put(f"dGamma_piece_{nterm}", term)
        S.show(f"dGamma_piece_{nterm}", f"Pieza {nterm} canonizada")

    S.check_zero(
        "check_dGamma_piece_1_vanishes",
        S["dGamma_piece_1"],
        "primera pieza = 0",
    )
    S.check_zero(
        "check_dGamma_piece_2_equals_3",
        S["dGamma_piece_2"] - S["dGamma_piece_3"],
        "pieza 2 - pieza 3 = 0",
    )

    j = tensor_indices("j", M)
    S.put(
        "palatini_metric_second_derivative",
        2*ctx.P_up(i,b,j,d)*DDh(-j,-b,-d,-i),
    )
    S.check_zero(
        "check_after_dGamma_reduction",
        S["after_dGamma_full"] - S["palatini_metric_second_derivative"],
        "expansión completa - combinación reducida = 0",
    )
    return S.show(
        "palatini_metric_second_derivative",
        "Resultado tras sustituir δΓ",
    )
