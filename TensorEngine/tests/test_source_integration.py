from __future__ import annotations

import json
import os

import pytest
import sympy as sp

from tensor_engine import (
    DimensionSpec, EngineOptions, LagrangianSourceSpec, ParameterSpec,
    RunExporter, RunPackage, TensorEngine, WolframXActBridge,
    draft4_circular_ansatz, spatially_flat_flrw_ansatz,
    delta_count,
)
from tensor_engine.components import ir_scalar_to_sympy


@pytest.fixture(scope="module", params=("draft4", "flrw"))
def compiled_run(request):
    draft4 = request.param == "draft4"
    source = LagrangianSourceSpec(
        name="source_integration_" + request.param,
        expression="R + 2/ell**2 + ell**2*beta0*(3*RicciUU - X*R)",
        dimension=DimensionSpec(3 if draft4 else 4),
        parameters=tuple(ParameterSpec(name) for name in ("ell", "beta0", "p")),
    )
    live = os.environ.get("TENSOR_ENGINE_RUN_WOLFRAM_TESTS") == "1"
    run = TensorEngine(options=EngineOptions(include_noether=live)).run(
        source.compile(),
        ansatz=draft4_circular_ansatz() if draft4 else spatially_flat_flrw_ansatz(),
        wolfram_bridge=WolframXActBridge(timeout_seconds=300) if live else None,
    )
    return request.param, source, run


def test_case2_runs_on_both_ansatz_without_hiding_limitations(compiled_run):
    kind, source, run = compiled_run
    assert run.abstract is not None and run.projected is not None
    assert len(run.projected.quantities) == 11
    assert run.package.verification.summary["failed"] == 0
    assert run.projected.ansatz_name == ("draft4_circular" if kind == "draft4" else "flat_flrw")
    assert run.projected.lagrangian.status.value == "completed"
    for quantity in run.projected.quantities:
        if quantity.components is None:
            assert quantity.reason
            assert getattr(run.abstract, quantity.key) is not None
        else:
            assert quantity.components.dimension == source.dimension.value
    if os.environ.get("TENSOR_ENGINE_RUN_WOLFRAM_TESTS") == "1":
        assert run.package.verification.external_bindings
        assert run.abstract.record("curvature_momentum").xact_status.value == "validated_with_xact"
    # In 4D a nonzero rank-six tensor exceeds the existing 2048 component budget.
    if kind == "flrw":
        assert run.projected.nabla_nabla_P.status.value == "unavailable"
        assert "4096" in run.projected.nabla_nabla_P.reason


def test_projected_lagrangian_matches_coordinate_reference(compiled_run):
    kind, _, run = compiled_run
    ell, beta, p = sp.symbols("ell beta0 p")
    if kind == "draft4":
        r = sp.Symbol("r")
        f = sp.Function("f")(r)
        expected = (
            -sp.diff(f, r, 2) - 2*sp.diff(f, r)/r + 2/ell**2
            + beta*ell**2*p**2*(sp.diff(f, r, 2)/r**2 - sp.diff(f, r)/r**3)
        )
    else:
        t = sp.Symbol("t")
        a, phi = sp.Function("a")(t), sp.Function("phi")(t)
        expected = (
            6*(sp.diff(a, t, 2)/a + sp.diff(a, t)**2/a**2) + 2/ell**2
            + beta*ell**2*sp.diff(phi, t)**2*(
                6*sp.diff(a, t)**2/a**2 - 3*sp.diff(a, t, 2)/a
            )
        )
    assert sp.simplify(ir_scalar_to_sympy(run.projected.lagrangian.scalar) - expected) == 0


def test_source_views_manifest_and_latex_survive_roundtrip(compiled_run, tmp_path):
    _, source, run = compiled_run
    rebuilt = RunPackage.from_data(json.loads(json.dumps(run.package.to_data())))
    assert rebuilt.run_id == run.package.run_id
    assert rebuilt.abstract == run.abstract and rebuilt.projected == run.projected
    assert rebuilt.delta_contractions == run.delta_contractions
    assert dict(rebuilt.model.metadata)["source_expression"] == source.expression
    assert "RicciUU" in json.loads(dict(rebuilt.model.metadata)["source_invariants"])
    bundle = RunExporter(tmp_path, compile_pdf=False).export(rebuilt)
    report = (bundle.output_directory / "report.tex").read_text(encoding="utf-8")
    assert report.count(r"\section*{") == 2
    assert report.count(r"\subsection*{") == 22
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["result_views"]["abstract"]) == 11
    assert len(manifest["result_views"]["projected"]) == 11
    audit = json.loads((bundle.output_directory / "delta_contractions.json").read_text(encoding="utf-8"))
    assert audit["passes"] == [a.to_data() for a in run.delta_contractions]
    assert all(count == 0 for count in audit["final_abstract_counts"].values())
    for quantity in run.projected.quantities:
        if quantity.components is None:
            assert "Motivo:" in report


def test_case2_canonical_deltas_are_resolved_before_projection_and_validation(compiled_run):
    from tensor_engine import Number
    kind, _, run = compiled_run
    assert run.delta_contractions
    assert any(e.action == "substitute" for a in run.delta_contractions for e in a.events)
    for _, expression in run.abstract.expression_items():
        assert delta_count(expression) == 0
    assert run.projected.metric_euler.status.value == "completed"
    assert run.projected.scalar_euler.status.value == "completed"
    if kind == "draft4":
        assert run.projected.scalar_euler.scalar == Number(0)


def test_presentation_is_read_only_and_more_compact(compiled_run):
    from tensor_engine import DisplayPolicy, build_presentation
    from tensor_engine.exporting import display_expr_to_latex, expr_to_latex, latex_report
    kind, _, run = compiled_run
    package = run.package
    before = json.dumps(package.to_data(), sort_keys=True)
    source_id = package.run_id
    ansatz = draft4_circular_ansatz() if kind == "draft4" else spatially_flat_flrw_ansatz()
    view = build_presentation(package, projected_assumptions=ansatz.assumptions)
    assert view.run_id == source_id
    assert not any(r.status == "fallback" for _, r in view.expressions)
    expected_count = len(package.abstract.records) + 1
    for q in package.projected.quantities:
        expected_count += max(1, len(q.components.values)) if q.components is not None else 1
    assert len(view.expressions) == expected_count  # includes undisplayed sparse components
    for key, record in view.expressions:
        if key.startswith("projected.") and not key.endswith("abstract_fallback"):
            assert sp.cancel(ir_scalar_to_sympy(record.canonical) - ir_scalar_to_sympy(record.presentation)) == 0
    records = [r for key, r in view.expressions if key.startswith("abstract.")]
    canonical_size = sum(len(expr_to_latex(r.canonical)) for r in records)
    presented_size = sum(len(display_expr_to_latex(r.presentation)) for r in records)
    assert presented_size < canonical_size * 0.85
    tex = latex_report(package, presentation=view)
    assert tex.count(r"\section*{") == 2
    assert tex.count(r"\subsection*{") == 22
    assert "% display:" in tex
    assert "presentation.json" in tex
    assert json.dumps(package.to_data(), sort_keys=True) == before
    assert package.run_id == source_id
    unchanged = build_presentation(package, DisplayPolicy(enabled=False))
    assert all(r.canonical is r.presentation for _, r in unchanged.expressions)


def test_presentation_does_not_change_canonical_files_or_manifest_bindings(compiled_run, tmp_path):
    from tensor_engine import DisplayPolicy, RunManifest
    kind, _, run = compiled_run
    ansatz = draft4_circular_ansatz() if kind == "draft4" else spatially_flat_flrw_ansatz()
    bundles = [RunExporter(tmp_path / name, compile_pdf=False,
                           display_policy=policy, projected_assumptions=ansatz.assumptions).export(
                               run.package, created_at_utc="2026-08-30T00:00:00Z")
               for name, policy in (("raw", DisplayPolicy(enabled=False)), ("readable", DisplayPolicy()))]
    raw, readable = bundles
    for filename in ("results.json", "verification.json", "delta_contractions.json"):
        assert (raw.output_directory / filename).read_bytes() == (readable.output_directory / filename).read_bytes()
    assert raw.manifest.expressions == readable.manifest.expressions
    assert raw.manifest.projected_quantities == readable.manifest.projected_quantities
    assert raw.manifest.external_bindings == readable.manifest.external_bindings
    assert raw.manifest.external_sources == readable.manifest.external_sources
    assert raw.manifest.run_id == readable.manifest.run_id
    for bundle in bundles:
        loaded = RunManifest.from_data(json.loads(bundle.manifest_path.read_text(encoding="utf-8")))
        assert loaded.verify_files(bundle.output_directory) == ()
        audit = json.loads((bundle.output_directory / "presentation.json").read_text(encoding="utf-8"))
        assert audit["purpose"] == "presentation_only"
        for record in audit["expressions"].values():
            assert "latex" in record and "assumptions_used" in record and "operations" in record
