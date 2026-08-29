import sympy as sp
from IPython.display import display, Markdown
from sympy.tensor.tensor import tensor_indices

from llsum_core import (
    LLCurvatureContribution,
    LLMetricContribution,
    LLRcalContribution,
    LLBianchiContribution,
    LLBianchiRepresentative,
    ll_compact_normal_form,
)


def configure_max_order(ctx, m_input):
    ctx.configure_max_order(m_input)
    S = ctx.S

    S.put("m_input", sp.Integer(ctx.max_order))
    for k, ck in enumerate(ctx.couplings):
        S.put(f"coupling_c_{k}", ck, simplify=False)
    for k in range(ctx.max_order + 1):
        S.put(f"L_{k}_constructed", ctx.L_terms[k], simplify=False)
    S.put("L_total_constructed", ctx.L_total_expr, simplify=False)
    S.put("L_input", ctx.L_total_expr, simplify=False)

    display(Markdown("### Input y construcción de todos los términos hasta orden m"))
    S.show("m_input", "Orden máximo elegido")

    for k in range(ctx.max_order + 1):
        S.show(
            f"L_{k}_constructed",
            rf"Término $L_{k}$ construido",
        )

    S.show(
        "L_total_constructed",
        rf"Lagrangiano final $L=\sum_{{k=0}}^{{{ctx.max_order}}}c_kL_k$",
    )

    display(Markdown(
        "Los coeficientes se crean automáticamente como "
        + ", ".join(rf"$c_{k}$" for k in range(ctx.max_order + 1))
        + "."
    ))


def stage_1_build_P_and_divergence(ctx):
    S = ctx.S
    mmax = ctx.max_order

    display(Markdown(
        "### 1A. Derivar cada $L_k$ y sumar $P^{abcd}=\\sum_k c_kP_{(k)}^{abcd}$"
    ))

    P_total_raw = sp.S.Zero
    P_total_canonical = sp.S.Zero
    all_term_checks = []

    for k in range(mmax + 1):
        Ck = ctx.normalization(k)
        if k == 0:
            raw_k = sp.S.Zero
            canonical_k = sp.S.Zero
        else:
            contributions = [
                LLCurvatureContribution(k, r) for r in range(1, k+1)
            ]
            raw_k = Ck * sp.Add(*contributions)
            canonical_k = ll_compact_normal_form(raw_k)
            checks = [
                ll_compact_normal_form(
                    LLCurvatureContribution(k, r)
                    - LLCurvatureContribution(k, 1)
                )
                for r in range(2, k+1)
            ]
            all_term_checks.extend(checks)

        S.put(f"P_{k}_product_rule_raw", raw_k, simplify=False)
        S.put(f"P_{k}_compact", canonical_k, simplify=False)
        P_total_raw += ctx.couplings[k] * raw_k
        P_total_canonical += ctx.couplings[k] * canonical_k

        S.show(f"P_{k}_product_rule_raw", rf"$P_{{({k})}}^{{abcd}}$ antes de canonizar")
        S.show(f"P_{k}_compact", rf"$P_{{({k})}}^{{abcd}}$ canonizado")

    P_total_canonical = ll_compact_normal_form(P_total_canonical)

    S.put("P_total_product_rule_raw", P_total_raw, simplify=False)
    S.put("P_total_compact", P_total_canonical, simplify=False)
    S.show("P_total_product_rule_raw", "Suma de todas las contribuciones de P")
    S.show("P_total_compact", "P total después de canonizar")

    all_zero = all(x == 0 for x in all_term_checks)
    S.put(
        "check_all_P_slot_contributions_equal",
        sp.S.Zero if all_zero else sp.S.One,
    )
    S.show(
        "check_all_P_slot_contributions_equal",
        "Verificación conjunta de equivalencia entre slots",
    )
    if not all_zero:
        raise AssertionError("Alguna contribución por slot de P no canonizó igual.")

    # Registrar el tensor total que usa la cadena variacional genérica.
    if mmax == 0:
        S.put("P_abcd_tensor", sp.S.Zero)
    else:
        a, b, c, d = tensor_indices("a b c d", ctx.M)
        S.put("P_abcd_tensor", ctx.P_up(a,b,c,d))
    S.show("P_abcd_tensor", "Tensor P total registrado para las etapas posteriores")

    # Simetrías.
    if mmax >= 1:
        a, b, c, d = tensor_indices("a b c d", ctx.M)
        S.check_zero(
            "check_P_antisym_ab",
            ctx.P_up(a,b,c,d) + ctx.P_up(b,a,c,d),
            "P^{abcd}+P^{bacd}=0",
        )
        S.check_zero(
            "check_P_antisym_cd",
            ctx.P_up(a,b,c,d) + ctx.P_up(a,b,d,c),
            "P^{abcd}+P^{abdc}=0",
        )
        S.check_zero(
            "check_P_pair_exchange",
            ctx.P_up(a,b,c,d) - ctx.P_up(c,d,a,b),
            "P^{abcd}-P^{cdab}=0",
        )
    else:
        for key in (
            "check_P_antisym_ab",
            "check_P_antisym_cd",
            "check_P_pair_exchange",
        ):
            S.put(key, sp.S.Zero)

    # ------------------------------------------------------------------
    # 1B. Demostrar divergencia nula término a término y para la suma.
    # ------------------------------------------------------------------
    display(Markdown("### 1B. Derivar cada $P_{(k)}$ y aplicar Bianchi"))

    total_div_raw = sp.S.Zero
    for k in range(mmax + 1):
        Ck = ctx.normalization(k)
        if k <= 1:
            raw_div_k = sp.S.Zero
        else:
            raw_div_k = Ck*k*sp.Add(*[
                LLBianchiContribution(k, r) for r in range(2, k+1)
            ])
        canonical_div_k = ll_compact_normal_form(raw_div_k)

        S.put(f"Bianchi_representative_{k}", LLBianchiRepresentative(k), simplify=False)
        S.put(f"divergence_P_{k}_raw", raw_div_k, simplify=False)
        S.put(f"divergence_P_{k}", canonical_div_k, simplify=False)

        S.show(f"divergence_P_{k}_raw", rf"$\nabla P_{{({k})}}$ antes de Bianchi")
        S.show(f"divergence_P_{k}", rf"$\nabla P_{{({k})}}$ después de Bianchi")
        S.check_zero(
            f"check_divergence_P_{k}",
            canonical_div_k,
            rf"$\nabla_aP_{{({k})}}^{{abcd}}=0$",
        )

        total_div_raw += ctx.couplings[k] * raw_div_k

    total_div = ll_compact_normal_form(total_div_raw)
    S.put("divergence_P_total_raw", total_div_raw, simplify=False)
    S.put("divergence_P_total", total_div, simplify=False)
    S.show("divergence_P_total_raw", "Divergencia de la suma completa antes de Bianchi")
    S.show("divergence_P_total", "Divergencia de la suma completa")
    S.check_zero(
        "check_divergence_P_total",
        total_div,
        "∇_aP^{abcd}=0 para la suma completa",
    )

    ctx.divergence_free_derived = True


def stage_2_metric_derivative(ctx):
    S = ctx.S
    mmax = ctx.max_order

    display(Markdown(
        "### 2A. Derivar métricamente cada $L_k$ y construir la suma completa"
    ))

    metric_total = sp.S.Zero
    rcal_total = sp.S.Zero

    for k in range(mmax + 1):
        Ck = ctx.normalization(k)
        if k == 0:
            metric_raw_k = sp.S.Zero
            rcal_raw_k = sp.S.Zero
        else:
            metric_raw_k = Ck*sp.Add(*[
                LLMetricContribution(k, r, leg)
                for r in range(1, k+1)
                for leg in (1,2)
            ])
            rcal_raw_k = Ck*sp.Add(*[
                LLRcalContribution(k, r)
                for r in range(1, k+1)
            ])

        metric_k = ll_compact_normal_form(metric_raw_k)
        rcal_k = ll_compact_normal_form(rcal_raw_k)

        S.put(f"P_metric_{k}_raw", metric_raw_k, simplify=False)
        S.put(f"P_metric_{k}_compact", metric_k, simplify=False)
        S.put(f"Rcal_{k}_raw", rcal_raw_k, simplify=False)
        S.put(f"Rcal_{k}_compact", rcal_k, simplify=False)

        S.show(f"P_metric_{k}_compact", rf"Derivada métrica de $L_{k}$")
        S.show(f"Rcal_{k}_compact", rf"$\mathcal R_{{({k})}}^{{ab}}$")

        S.check_zero(
            f"check_metric_identity_{k}",
            ll_compact_normal_form(metric_k + 2*rcal_k),
            rf"$P_{{({k})}}^{{ab}}+2\mathcal R_{{({k})}}^{{ab}}=0$",
        )

        metric_total += ctx.couplings[k] * metric_k
        rcal_total += ctx.couplings[k] * rcal_k

    metric_total = ll_compact_normal_form(metric_total)
    rcal_total = ll_compact_normal_form(rcal_total)

    S.put("P_metric_total_compact", metric_total, simplify=False)
    S.put("Rcal_total_compact", rcal_total, simplify=False)
    S.show("P_metric_total_compact", "Derivada métrica total")
    S.show("Rcal_total_compact", "Rcal total")

    S.check_zero(
        "check_metric_identity_total",
        ll_compact_normal_form(metric_total + 2*rcal_total),
        "P^{ab}+2Rcal^{ab}=0 para la suma completa",
    )

    ctx.metric_identity_derived = True

    if mmax == 0:
        S.put("P_metric_ab", sp.S.Zero)
    else:
        a, b = tensor_indices("a b", ctx.M)
        S.put("P_metric_ab", ctx.P_metric_up(a,b))
    S.show("P_metric_ab", "P^{ab} total registrado para la variación")
