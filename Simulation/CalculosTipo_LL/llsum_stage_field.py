import sympy as sp
from IPython.display import display, Markdown
from sympy.tensor.tensor import tensor_indices


def stage_12_field_tensor(ctx):
    M, g, H, sqrtg, S = ctx.M, ctx.g, ctx.H, ctx.sqrtg, ctx.S
    a, b = tensor_indices("a b", M)

    S.put(
        "E_ab_raw",
        ctx.Rcal_down(a,b)
        - sp.Rational(1,2)*g(-a,-b)*ctx.L_total_expr
        + S["minus2_double_divergence_P_ab"],
    )
    S.show("E_ab_raw", "Tensor de campo construido desde el Lagrangiano total")

    S.put(
        "E_ab_LL_simplified",
        ctx.Rcal_down(a,b)
        - sp.Rational(1,2)*g(-a,-b)*ctx.L_total_expr,
    )
    S.show(
        "E_ab_LL_simplified",
        "Forma de Lovelock después de usar ∇P=0",
    )

    # Forma explícita que ya no contiene símbolos L_k.
    S.put(
        "E_mixed_explicit_total",
        ctx.explicit_field_mixed(),
        simplify=False,
    )
    S.show(
        "E_mixed_explicit_total",
        "Tensor de campo total escrito sin abreviaturas L_k",
    )

    S.put(
        "delta_action_bulk_integrand",
        sqrtg*S["E_ab_raw"]*H(a,b),
    )
    S.show("delta_action_bulk_integrand", "Integrando bulk final de δA")

    return S.check_zero(
        "check_E_LL_reduction",
        S["E_ab_raw"] - S["E_ab_LL_simplified"],
        "E_ab general - E_ab Lovelock = 0",
    )


def stage_13_lovelock_diagnostics(ctx):
    S, mmax = ctx.S, ctx.max_order

    S.put("divergence_P_bcd", sp.S.Zero)
    S.put("highest_lovelock_order", sp.Integer(mmax))
    S.put("highest_critical_dimension", sp.Integer(2*mmax))
    S.put("minimum_dimension_for_highest_dynamic_term", sp.Integer(2*mmax+1))

    S.show("divergence_P_bcd", "∇_a P^{abcd} de la suma completa")
    S.show("highest_lovelock_order", "Orden de Lovelock más alto incluido")
    S.show("highest_critical_dimension", "Dimensión crítica del término de mayor orden")
    S.show(
        "minimum_dimension_for_highest_dynamic_term",
        "Primera dimensión donde el término más alto puede ser dinámico",
    )

    display(Markdown("### Diagnóstico automático de la teoría completa"))
    display(Markdown(
        f"- El input `m={mmax}` construye todos los términos $L_0,L_1,\ldots,L_{mmax}$.\n"
        f"- El Lagrangiano usado en toda la variación es $L=\\sum_{{k=0}}^{{{mmax}}}c_kL_k$.\n"
        "- Cada $P_{(k)}^{abcd}$ es divergencia-cero; por linealidad, el $P^{abcd}$ total también lo es.\n"
        "- Por tanto, el doble divergente se anula y las ecuaciones son de segundo orden.\n"
        f"- El término de mayor orden $L_{mmax}$ tiene dimensión crítica $D={2*mmax}$."
    ))


def show_stored_objects(ctx):
    S = ctx.S
    display(Markdown(f"### Se almacenaron {len(S)} objetos simbólicos"))
    for key in S:
        print(key)
