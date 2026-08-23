
import sympy as sp
from IPython.display import display, Markdown
from sympy.tensor.tensor import tensor_indices


def stage_12_field_tensor(ctx):
    M, g, H, DR, DDR, sqrtg, S = (
        ctx.M, ctx.g, ctx.H, ctx.DR, ctx.DDR, ctx.sqrtg, ctx.S
    )
    L_input, f2, f3 = ctx.L_input, ctx.f2, ctx.f3

    a, b = tensor_indices("a b", M)

    S.put(
        "E_ab_raw",
        S["Rcal_down_ab"]
        - sp.Rational(1,2)*g(-a,-b)*L_input
        + S["minus2_double_divergence_P_ab"]
    )

    S.show("E_ab_raw", "Tensor de campo construido paso a paso")

    S.put(
        "delta_action_bulk_integrand",
        sqrtg * S["E_ab_raw"] * H(a,b)
    )

    S.show("delta_action_bulk_integrand", "Integrando bulk final de δA")

    m = tensor_indices("m", M)
    box_f1 = (
        f2 * DDR(m,-m)
        + f3 * DR(m)*DR(-m)
    )
    hess_f1_ab = (
        f2 * DDR(-a,-b)
        + f3 * DR(-a)*DR(-b)
    )
    S.put(
        "derivative_piece_chain_rule_check",
        g(-a,-b)*box_f1 - hess_f1_ab
    )

    return S.check_zero(
        "check_double_divergence_chain_rule",
        S["minus2_double_divergence_P_ab"]
        - S["derivative_piece_chain_rule_check"],
        "(-2∇∇P)_ab - expansión por regla de la cadena = 0"
    )


def stage_13_order_diagnostic(ctx):
    M, S = ctx.M, ctx.S
    f2 = ctx.f2

    a, b, c, d = tensor_indices("a b c d", M)

    S.put(
        "divergence_P_bcd",
        ctx.DP_up(a,a,b,c,d)
    )

    S.show("divergence_P_bcd", "∇_a P^{abcd} calculado")

    display(Markdown("### Diagnóstico automático"))
    display(sp.Eq(sp.Symbol("f_RR"), f2))

    if sp.simplify(f2) == 0:
        display(Markdown(
            "**Segundo orden:** el resultado calculado para `S['divergence_P_bcd']` "
            "se anula identitariamente."
        ))
        assert S["divergence_P_bcd"] == 0
    else:
        display(Markdown(
            "**Genéricamente cuarto orden:** `S['divergence_P_bcd']` contiene "
            "gradientes de $R$, y `S['minus2_double_divergence_P_ab']` contiene "
            "el Hessiano de $R$."
        ))


def show_stored_objects(ctx):
    S = ctx.S

    display(Markdown(f"### Se almacenaron {len(S)} objetos simbólicos"))

    for key in S.keys():
        print(key)
