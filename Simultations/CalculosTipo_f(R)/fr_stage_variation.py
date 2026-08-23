
import sympy as sp
from sympy.tensor.tensor import TensorHead, TensorSymmetry, tensor_indices


def stage_5_vary_density(ctx):
    M, g, H, dRiem, sqrtg, S = (
        ctx.M, ctx.g, ctx.H, ctx.dRiem, ctx.sqrtg, ctx.S
    )
    L_input = ctx.L_input

    a, b, p, q = tensor_indices("a b p q", M)

    S.put(
        "P_metric_contravariant_variable_ab",
        -g(-a, -p) * g(-b, -q) * ctx.P_metric_up(p, q)
    )

    S.put(
        "delta_L_metric",
        S["P_metric_contravariant_variable_ab"] * H(a, b)
    )

    a, b, c, d = tensor_indices("a b c d", M)
    S.put(
        "delta_L_curvature_unsplit",
        ctx.P_up(a, b, c, d) * dRiem(-a, -b, -c, -d)
    )

    S.put(
        "delta_L_total_unsplit",
        S["delta_L_metric"] + S["delta_L_curvature_unsplit"]
    )

    a, b = tensor_indices("a b", M)
    S.put(
        "delta_sqrt_minus_g",
        -sp.Rational(1,2) * sqrtg * g(-a, -b) * H(a, b)
    )

    S.put(
        "delta_density_unsplit",
        S["delta_sqrt_minus_g"] * L_input
        + sqrtg * S["delta_L_total_unsplit"]
    )

    for key, title in [
        ("P_metric_contravariant_variable_ab", "∂L/∂g^{ab} obtenido por cambio de variable"),
        ("delta_L_metric", "Parte métrica de δL"),
        ("delta_L_curvature_unsplit", "Parte de curvatura de δL"),
        ("delta_L_total_unsplit", "δL completo antes de Palatini"),
        ("delta_sqrt_minus_g", "δ√(-g) como objeto tensorial"),
        ("delta_density_unsplit", "δ(√(-g)L) antes de separar δR"),
    ]:
        S.show(key, title)

    a, b = tensor_indices("a b", M)
    return S.check_zero(
        "check_metric_variation_equals_2Rcal",
        S["P_metric_contravariant_variable_ab"] - 2*S["Rcal_down_ab"],
        "∂L/∂g^{ab} - 2 Rcal_ab = 0"
    )


def stage_6_split_delta_R(ctx):
    M, g, Riem, H, S = ctx.M, ctx.g, ctx.Riem, ctx.H, ctx.S

    a, b, c, d, e = tensor_indices("a b c d e", M)

    S.put(
        "delta_R_split_metric_piece",
        ctx.P_up(a, b, c, d)
        * ctx.h_from_H(a, e)
        * Riem(e, -b, -c, -d)
    )

    dRm = TensorHead(
        r"\delta\mathrm{R}_{\mathrm{mix}}",
        [M]*4,
        TensorSymmetry.no_symmetry(4),
    )

    S.put(
        "delta_R_split_connection_piece",
        ctx.P_up(a, b, c, d) * g(-a, -e) * dRm(e, -b, -c, -d)
    )

    S.show("delta_R_split_metric_piece", "Primera pieza de P δR, realmente contraída")
    S.show("delta_R_split_connection_piece", "Segunda pieza de P δR")

    a, b = tensor_indices("a b", M)
    return S.check_zero(
        "check_split_metric_piece",
        S["delta_R_split_metric_piece"] + S["Rcal_down_ab"]*H(a,b),
        "P δg R + Rcal_ab H^ab = 0"
    )


def stage_7_palatini(ctx):
    M, g, DGamma, S = ctx.M, ctx.g, ctx.DGamma, ctx.S

    a, b, c, d, e = tensor_indices("a b c d e", M)

    palatini_1 = ctx.P_up(a,b,c,d) * g(-a,-e) * DGamma(e,-c,-d,-b)
    palatini_2 = -ctx.P_up(a,b,c,d) * g(-a,-e) * DGamma(e,-d,-c,-b)

    S.put("palatini_term_1_raw", palatini_1, simplify=False)
    S.put("palatini_term_2_raw", palatini_2, simplify=False)
    S.put("palatini_term_1", palatini_1)
    S.put("palatini_term_2", palatini_2)
    S.put("palatini_sum", palatini_1 + palatini_2)

    for key in (
        "palatini_term_1_raw",
        "palatini_term_2_raw",
        "palatini_term_1",
        "palatini_term_2",
        "palatini_sum",
    ):
        S.show(key)

    S.check_zero(
        "check_palatini_two_terms_equal",
        S["palatini_term_2"] - S["palatini_term_1"],
        "segundo término - primer término = 0"
    )

    return S.check_zero(
        "check_palatini_sum_is_twice",
        S["palatini_sum"] - 2*S["palatini_term_1"],
        "suma - 2×primer término = 0"
    )


def stage_8_substitute_dGamma(ctx):
    M, DDh, S = ctx.M, ctx.DDh, ctx.S

    a, b, c, d, e = tensor_indices("a b c d e", M)
    expanded = (
        2 * ctx.P_up(a,b,c,d)
        * ctx.g(-a,-e)
        * ctx.DGamma_from_h(e,c,d,b)
    )

    S.put("after_dGamma_full_raw", expanded, simplify=False)
    S.put("after_dGamma_full", expanded)

    i = tensor_indices("i", M)
    dg1 = ctx.P_up(i,b,c,d) * DDh(-c,-d,-b,-i)
    dg2 = ctx.P_up(i,b,c,d) * DDh(-c,-b,-d,-i)
    dg3 = -ctx.P_up(i,b,c,d) * DDh(-c,-i,-d,-b)

    for nterm, term in enumerate((dg1,dg2,dg3),1):
        S.put(f"dGamma_piece_{nterm}_raw", term, simplify=False)
        S.put(f"dGamma_piece_{nterm}", term)
        S.show(f"dGamma_piece_{nterm}_raw", f"Pieza {nterm} antes de canonizar")
        S.show(f"dGamma_piece_{nterm}", f"Pieza {nterm} canonizada")

    S.show("after_dGamma_full_raw")
    S.show("after_dGamma_full")

    S.check_zero(
        "check_dGamma_piece_1_vanishes",
        S["dGamma_piece_1"],
        "primera pieza = 0 por antisimetría/simetría"
    )

    S.check_zero(
        "check_dGamma_piece_2_equals_3",
        S["dGamma_piece_2"] - S["dGamma_piece_3"],
        "pieza 2 - pieza 3 = 0"
    )

    j = tensor_indices("j", M)
    S.put(
        "palatini_metric_second_derivative",
        2*ctx.P_up(i,b,j,d)*DDh(-j,-b,-d,-i)
    )

    S.check_zero(
        "check_after_dGamma_reduction",
        S["after_dGamma_full"] - S["palatini_metric_second_derivative"],
        "expansión completa - combinación reducida = 0"
    )

    return S.show(
        "palatini_metric_second_derivative",
        "Resultado calculado tras sustituir δΓ"
    )
