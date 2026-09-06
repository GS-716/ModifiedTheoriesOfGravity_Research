from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from tensor_engine import (
    AnsatzSpecialization,
    DimensionSpec,
    EngineOptions,
    LagrangianSourceSpec,
    ParameterSpec,
    Scalar,
    TensorEngine,
    draft4_circular_ansatz,
)
from field_equations_solver import SolverSearchPolicy, solveFieldEquations
from field_equations_solver.reporting import solution_latex


QUARTIC_RIEMANN_GRADIENT = (
    'contract(Riemann("a","b","c","d"), metric("a","e"), gradient("e"), '
    'metric("c","f"), gradient("f"), metric("b","g"), gradient("g"), '
    'metric("d","h"), gradient("h"))'
)


def _run(expression: str, name: str):
    model = LagrangianSourceSpec(
        name=name,
        expression=expression,
        dimension=DimensionSpec(3),
        parameters=tuple(ParameterSpec(value) for value in ("ell", "alpha", "beta0")),
    ).compile()
    return TensorEngine(options=EngineOptions(include_noether=False, include_export=False)).run(
        model, ansatz=draft4_circular_ansatz()
    )


def _constant_policy(**changes):
    values = dict(
        factor_branches=False,
        singular_branches=True,
        polynomial_degrees=(),
        power_exponents=(),
        use_wolfram=False,
        max_candidates=64,
    )
    values.update(changes)
    return SolverSearchPolicy(**values)


@pytest.mark.parametrize(
    "expression,name,constant_is_solution",
    [
        ("R", "constant_r", True),
        ("R + alpha*R**2", "constant_r2", True),
        ("RicciSq", "constant_ricci_sq", True),
        ("RiemannSq", "constant_riemann_sq", True),
        (QUARTIC_RIEMANN_GRADIENT, "constant_quartic", True),
        ("R + 2/ell**2 + ell**2*beta0*(3*RicciUU-X*R)", "constant_case2", False),
    ],
)
def test_constant_and_degenerate_branches_are_explicit(expression, name, constant_is_solution):
    result = solveFieldEquations(
        _run(expression, name),
        specialization=AnsatzSpecialization(scalar_field=Scalar("q") * Scalar("varphi")),
        search_policy=_constant_policy(),
    )
    constants = [candidate for candidate in result.solutions if candidate.origin == "constant_branch"]
    degenerates = [candidate for candidate in result.solutions
                   if candidate.origin == "degenerate_metric_branch"]
    assert constants and degenerates
    assert any("Ne(C_f, 0)" in candidate.branch_conditions for candidate in constants)
    assert all(candidate.status == "rejected" for candidate in degenerates)
    assert all(any("Dominio singular" in reason for reason in candidate.unresolved)
               for candidate in degenerates)
    assert any(candidate.status == "verified_on_domain" for candidate in constants) is constant_is_solution
    for candidate in constants:
        if candidate.status == "verified_on_domain":
            assert len(candidate.residuals) == 10
            assert len(candidate.mixed_residuals) == 9
            assert all(value.to_data()["type"] == "number" for _, value in candidate.mixed_residuals)


def test_parameter_scenarios_denominators_and_global_classification():
    result = solveFieldEquations(
        _run("R + 2/ell**2 + ell**2*beta0*(3*RicciUU-X*R)", "case2_branches"),
        specialization=AnsatzSpecialization(scalar_field=Scalar("q") * Scalar("varphi")),
        search_policy=_constant_policy(),
    )
    scenarios = {item["name"] for item in result.search_summary["parameter_scenarios"]}
    assert any("q=0" in item for item in scenarios)
    assert any("q!=0" in item for item in scenarios)
    assert any("beta0=0" in item for item in scenarios)
    assert any("beta0!=0" in item for item in scenarios)
    assert all("ell!=0" in item for item in scenarios)
    singular = [candidate for candidate in result.solutions
                if candidate.origin == "singular_denominator_branch"]
    assert singular and all(candidate.status == "rejected" for candidate in singular)
    assert all(len(candidate.residuals) == 10 and len(candidate.mixed_residuals) == 9
               for candidate in singular)
    assert result.search_summary["completeness_proven"] is False
    assert result.status in {"partially_solved", "no_verified_candidate"}


@pytest.mark.parametrize("expression", ["R + alpha*RicciUU", "R + alpha*R*X"])
def test_constant_branch_is_also_explored_for_existing_scalar_curvature_cases(expression):
    result = solveFieldEquations(
        _run(expression, "scalar_curvature_constant_branch"),
        specialization=AnsatzSpecialization(scalar_field=Scalar("q") * Scalar("varphi")),
        search_policy=_constant_policy(),
    )
    assert any(candidate.origin == "constant_branch" for candidate in result.solutions)
    assert any(branch["kind"] == "constant" for branch in result.search_summary["branches"])


def test_verified_and_nonaccepted_candidates_are_separated_in_report():
    result = solveFieldEquations(
        _run("R", "accepted_and_rejected"),
        specialization=AnsatzSpecialization(scalar_field=Scalar("q") * Scalar("varphi")),
        search_policy=_constant_policy(),
    )
    assert result.status == "verified_with_pending_branches"
    assert result.search_summary["has_verified_family"] is True
    assert result.search_summary["candidate_status_counts"]["rejected"] > 0
    tex, presentation = solution_latex(result)
    assert tex.index("Lagrangiano:") < tex.index("Ecuaciones combinadas")
    assert tex.index("Modelo:") < tex.index("Lagrangiano:")
    main = tex.split(r"\section*{Familias verificadas}", 1)[1].split(
        r"\section*{Auditoría de ramas no aceptadas}", 1
    )[0]
    assert "no verificado" not in main and "rejected" not in main
    assert "lagrangian" in presentation["expressions"]
    payload = json.loads(json.dumps(result.to_data()))
    assert any(item["status"] == "verified_on_domain" for item in payload["solutions"])
    assert any(item["status"] == "rejected" for item in payload["solutions"])
    assert all("method" in item and "mixed_residuals" in item for item in payload["solutions"])


def test_alpha_zero_and_nonzero_are_both_explored():
    result = solveFieldEquations(
        _run("R + alpha*R**2", "alpha_branches"),
        search_policy=_constant_policy(singular_branches=False),
    )
    names = {item["name"] for item in result.search_summary["parameter_scenarios"]}
    assert any("alpha=0" in name for name in names)
    assert any("alpha!=0" in name for name in names)


def test_polynomial_power_and_factor_searches_are_recorded_and_verified():
    policy = SolverSearchPolicy(
        constant_branches=False,
        singular_branches=False,
        factor_branches=True,
        polynomial_degrees=(2,),
        power_exponents=(2,),
        use_wolfram=False,
        max_candidates=32,
    )
    result = solveFieldEquations(_run("R + 2/ell**2", "radial_families"), search_policy=policy)
    verified_methods = {candidate.origin for candidate in result.solutions
                        if candidate.status == "verified_on_domain"}
    assert "polynomial_degree_2" in verified_methods
    assert "power_2" in verified_methods
    branch_kinds = {branch["kind"] for branch in result.search_summary["branches"]}
    assert {"factor", "polynomial", "power"}.issubset(branch_kinds)
    assert result.status == "verified_with_pending_branches"

    tex, presentation = solution_latex(result)
    # Solver-generated identifiers remain canonical internally but are never
    # exposed as mathematical constants in the human-facing report.
    assert "polyfD2C0" not in tex
    assert "powerfE2B" not in tex
    assert r"C_{1}" in tex
    assert "ansatz polinomial de grado 2" in tex
    assert "ansatz de potencia con exponente 2" in tex
    assert any(
        "polyfD2C0" in aliases or "powerfE2B" in aliases
        for aliases in presentation["display_aliases"].values()
    )
    serialized = json.dumps(result.to_data(), sort_keys=True)
    assert "polyfD2C0" in serialized and "powerfE2B" in serialized


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="LaTeX no disponible")
def test_results_tex_and_pdf_belong_to_the_same_solver_bundle(tmp_path: Path):
    result = solveFieldEquations(
        _run("R", "report_coherence"), search_policy=_constant_policy()
    )
    directory = result.export(tmp_path, compile_pdf=True)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    data = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    tex = (directory / "report.tex").read_text(encoding="utf-8")
    assert data["source_run_id"] == manifest["source_run_id"]
    assert manifest["files"]["report.tex"] == hashlib.sha256(
        (directory / "report.tex").read_bytes()
    ).hexdigest()
    if not (directory / "report.pdf").is_file():
        pytest.skip("El compilador LaTeX no está operativo: " + str(manifest.get("pdf_diagnostic", "")))
    assert manifest["files"]["report.pdf"] == hashlib.sha256(
        (directory / "report.pdf").read_bytes()
    ).hexdigest()
    assert "Lagrangiano:" in tex and r"report\_coherence" in tex
