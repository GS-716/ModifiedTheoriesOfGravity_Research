from sympy.tensor.tensor import tensor_indices


def stage_9_ibp_first(ctx):
    M, Dh, DDh, S = ctx.M, ctx.Dh, ctx.DDh, ctx.S
    i, b, j, d = tensor_indices("i b j d", M)

    S.put("ibp_start", 2*ctx.P_up(i,b,j,d)*DDh(-j,-b,-d,-i))
    S.put("ibp1_boundary_vector", 2*ctx.P_up(i,b,j,d)*Dh(-b,-d,-i))
    S.put(
        "ibp1_divergence",
        2*ctx.DP_up(j,i,b,j,d)*Dh(-b,-d,-i)
        + 2*ctx.P_up(i,b,j,d)*DDh(-j,-b,-d,-i),
    )
    S.put(
        "ibp1_residual_positive",
        2*ctx.DP_up(j,i,b,j,d)*Dh(-b,-d,-i),
    )

    for key in (
        "ibp_start",
        "ibp1_boundary_vector",
        "ibp1_divergence",
        "ibp1_residual_positive",
    ):
        S.show(key)

    S.check_zero(
        "check_ibp1",
        S["ibp_start"] - (S["ibp1_divergence"] - S["ibp1_residual_positive"]),
        "integrando inicial - (divergencia - residuo) = 0",
    )

    return S.check_zero(
        "check_LL_ibp1_residual_vanishes",
        S["ibp1_residual_positive"],
        "el residuo de la primera IBP se anula porque ∇P=0",
    )


def stage_10_ibp_second(ctx):
    M, h, Dh, S = ctx.M, ctx.h, ctx.Dh, ctx.S
    c, i, j, d = tensor_indices("c i j d", M)

    S.put(
        "ibp1_residual_negative_renamed",
        -2*ctx.DP_up(c,i,j,c,d)*Dh(-j,-d,-i),
    )
    S.put(
        "ibp2_boundary_vector",
        2*h(-d,-i)*ctx.DP_up(c,i,j,c,d),
    )
    S.put(
        "ibp2_divergence",
        2*Dh(-j,-d,-i)*ctx.DP_up(c,i,j,c,d)
        + 2*h(-d,-i)*ctx.DDP_up(j,c,i,j,c,d),
    )
    S.put(
        "ibp2_bulk_hcov",
        2*h(-d,-i)*ctx.DDP_up(j,c,i,j,c,d),
    )

    for key in (
        "ibp1_residual_negative_renamed",
        "ibp2_boundary_vector",
        "ibp2_divergence",
        "ibp2_bulk_hcov",
    ):
        S.show(key)

    return S.check_zero(
        "check_ibp2",
        S["ibp1_residual_negative_renamed"]
        - (-S["ibp2_divergence"] + S["ibp2_bulk_hcov"]),
        "segunda IBP aplicada al residuo nulo",
    )


def stage_11_boundary_and_bulk(ctx):
    M, g, H, S = ctx.M, ctx.g, ctx.H, ctx.S

    S.put(
        "delta_v_vector",
        S["ibp1_boundary_vector"] - S["ibp2_boundary_vector"],
    )
    S.show("delta_v_vector", "δv^j para la suma completa de Lovelock")

    d, i, j, c = tensor_indices("d i j c", M)
    S.put(
        "ibp2_bulk_Hup",
        2*ctx.h_from_H(d,i)*ctx.DDP_up(j,c,i,j,c,d),
    )
    S.show("ibp2_bulk_Hup", "Bulk tras las dos IBP")

    a, b, m, n, p, q, r, s, t, u = tensor_indices(
        "a b m n p q r s t u", M
    )
    S.put(
        "minus2_double_divergence_P_ab",
        -2*g(m,p)*g(n,q)*g(-a,-r)*g(-m,-s)*g(-n,-t)*g(-b,-u)
        * ctx.DDP_up(p,q,r,s,t,u),
    )
    S.show(
        "minus2_double_divergence_P_ab",
        "-2 ∇^m∇^n P_amnb",
    )

    S.check_zero(
        "check_ibp_bulk_equals_double_divergence",
        S["ibp2_bulk_Hup"] - S["minus2_double_divergence_P_ab"]*H(a,b),
        "bulk de IBP - (-2∇∇P)_ab H^ab = 0",
    )

    return S.check_zero(
        "check_LL_double_divergence_vanishes",
        S["minus2_double_divergence_P_ab"],
        "el doble divergente se anula para la suma completa de Lovelock",
    )
