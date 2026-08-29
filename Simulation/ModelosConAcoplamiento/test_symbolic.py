"""Pruebas de regresion exactas para el motor simbolico."""

import sympy as sp

from mc_pipeline import run_pipeline


def main() -> None:
    ctx = run_pipeline()
    assert len(ctx.steps) >= 175
    assert ctx.checks
    assert all(sp.simplify(value) == 0 for value in ctx.checks.values())
    assert all(sp.simplify(value) == 0 for value in ctx.objects["E_btz"])
    assert all(sp.simplify(value) == 0 for value in ctx.objects["E_case1_solution"])
    assert sp.simplify(ctx.objects["Ephi_case1_ansatz"]) == 0
    assert sp.simplify(ctx.objects["R_btz"] + 6 / ctx.objects["ell"]**2) == 0
    assert ctx.objects["J_case0_ansatz"] == sp.zeros(3, 1)
    r = ctx.objects["coordinates"][1]
    expected_Jphi = -2 * ctx.objects["alpha1"] * ctx.objects["p"] / r**2
    assert sp.simplify(ctx.objects["J_case1_ansatz"][2] - expected_Jphi) == 0
    f = ctx.objects["f"]
    assert all(not value.has(f) for value in ctx.objects["P_case0_final"].values())
    assert all(not value.has(f) for value in ctx.objects["P_case1_final"].values())
    for key in ("M_case0_final", "Rcal_case0_final", "M_case1_final", "Rcal_case1_final"):
        assert not ctx.objects[key].has(f)
    assert all(sp.simplify(value) == 0 for value in ctx.objects["E_case2_solution"])
    assert sp.simplify(ctx.objects["Ephi_case2_ansatz"]) == 0
    assert all(not value.has(f) for value in ctx.objects["P_case2_final"].values())
    for key in ("M_case2_final", "Rcal_case2_final"):
        assert not ctx.objects[key].has(f)
    f2 = ctx.objects["f_case2_solution"]
    H2 = ctx.objects["H_case2"]
    expected_N2 = r**2 / ctx.objects["ell"]**2 - ctx.objects["lambda"]
    assert sp.simplify(H2 * f2 - expected_N2) == 0
    # Generalizacion EQT: alpha_1+alpha_2+beta_1 en una sola corrida.
    assert all(sp.simplify(value) == 0 for value in ctx.objects["eqt_E_solution"])
    assert sp.simplify(ctx.objects["eqt_Ephi_ansatz"]) == 0
    assert sp.simplify(
        ctx.objects["eqt_H"] * ctx.objects["eqt_f_solution"] - ctx.objects["eqt_N"]
    ) == 0
    assert ctx.objects["eqt_radial_factor"] == 1 / (2 * r * f)
    assert all(not value.has(f) for value in ctx.objects["eqt_P_final"].values())
    for key in ("eqt_M_final", "eqt_J_final", "eqt_Rcal_final"):
        assert not ctx.objects[key].has(f)
    spec = ctx.objects["eqt_spec"]
    assert spec.alpha_orders == (1, 2)
    assert spec.beta_orders == (1,)
    print(f"OK: {len(ctx.steps)} pasos y {len(ctx.checks)} verificaciones exactas.")


if __name__ == "__main__":
    main()
