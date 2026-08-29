import sympy as sp
from IPython.display import display, Markdown
from sympy.tensor.tensor import tensor_indices


def stage_12_field_tensor(ctx):
    M, g, H, sqrtg, S = (
        ctx.M, ctx.g, ctx.H, ctx.sqrtg, ctx.S
    )
    L_input = ctx.L_input

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

    # ------------------------------------------------
    # Comprobación independiente del término derivativo:
    # componente R + componente Gauss--Bonnet.
    # ------------------------------------------------
    m = tensor_indices("m", M)
    hess_LR_ab = ctx.scalar_hessian(ctx.L_R, a, b)
    box_LR = ctx.scalar_hessian(ctx.L_R, m, -m)

    S.put(
        "derivative_piece_R_component",
        g(-a,-b)*box_LR - hess_LR_ab
    )

    # Para la plantilla GB, ∇P_GB=0. Por tanto el doble divergente
    # solo actúa sobre el coeficiente escalar L_GB.
    m, n, p, q, r, s, t, u = tensor_indices(
        "GBm GBn GBp GBq GBr GBs GBt GBu", M
    )
    S.put(
        "derivative_piece_GB_component",
        -2
        * g(m,p) * g(n,q)
        * g(-a,-r) * g(-m,-s) * g(-n,-t) * g(-b,-u)
        * ctx.scalar_hessian(ctx.L_GB, p, q)
        * ctx.P_component_up("GB", r, s, t, u)
    )

    S.put(
        "derivative_piece_chain_rule_check",
        S["derivative_piece_R_component"]
        + S["derivative_piece_GB_component"]
    )

    S.show(
        "derivative_piece_R_component",
        "Contribución derivativa asociada a L_R"
    )
    S.show(
        "derivative_piece_GB_component",
        "Contribución derivativa asociada a L_GB"
    )

    return S.check_zero(
        "check_double_divergence_chain_rule",
        S["minus2_double_divergence_P_ab"]
        - S["derivative_piece_chain_rule_check"],
        "(-2∇∇P)_ab - suma de las contribuciones R y GB = 0"
    )


def stage_13_order_diagnostic(ctx):
    M, S = ctx.M, ctx.S

    a, b, c, d = tensor_indices("a b c d", M)

    S.put("gradient_L_R", ctx.scalar_covd(ctx.L_R, a))
    S.put("gradient_L_GB", ctx.scalar_covd(ctx.L_GB, a))
    S.put("divergence_P_bcd", ctx.DP_up(a,a,b,c,d))

    S.show("gradient_L_R", "∇_a L_R")
    S.show("gradient_L_GB", "∇_a L_GB")
    S.show("divergence_P_bcd", "∇_a P^{abcd} calculado")

    display(Markdown("### Diagnóstico automático"))
    display(sp.Eq(sp.Symbol("L_R"), ctx.L_R))
    display(sp.Eq(sp.Symbol("L_GB"), ctx.L_GB))

    grad_R_zero = S["gradient_L_R"] == 0
    grad_GB_zero = S["gradient_L_GB"] == 0

    if grad_R_zero and grad_GB_zero:
        display(Markdown(
            "**Sector de segundo orden:** los coeficientes de las estructuras "
            "$\partial R/\partial R_{abcd}$ y $\partial\mathcal G/\partial R_{abcd}$ "
            "son constantes en el espacio-tiempo, por lo que "
            "`S['divergence_P_bcd']` se anula identitariamente."
        ))
        assert S["divergence_P_bcd"] == 0

        if ctx.L_GB != 0:
            display(Markdown(
                "Para un término lineal $\alpha\mathcal G$, esta es precisamente la "
                "propiedad de Lovelock. En $D=4$, además, el término Gauss--Bonnet "
                "lineal es topológico; el motor tensorial abstracto no fija una "
                "dimensión concreta, así que esa cancelación específica de $D=4$ "
                "no se impone automáticamente."
            ))
    else:
        fuentes = []
        if not grad_R_zero:
            fuentes.append("$L_R$ no es constante")
        if not grad_GB_zero:
            fuentes.append("$L_{\mathcal G}$ no es constante")

        display(Markdown(
            "**Genéricamente de orden superior:** " + " y ".join(fuentes) + ". "
            "Por eso $\nabla_aP^{abcd}$ no se anula en general y el doble "
            "divergente de $P$ introduce derivadas adicionales de los invariantes."
        ))


def show_stored_objects(ctx):
    S = ctx.S

    display(Markdown(f"### Se almacenaron {len(S)} objetos simbólicos"))

    for key in S.keys():
        print(key)
