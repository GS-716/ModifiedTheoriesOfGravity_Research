from __future__ import annotations

from dataclasses import replace
import json

from tensor_engine import (
    ComponentProjectionStatus,
    BackendExecutionError,
    DerivedQuantities,
    DimensionSpec,
    EngineOptions,
    FunctionSpec,
    LagrangianSourceSpec,
    RunExporter,
    RunPackage,
    StructuralTensorBackend,
    SympyComponentBackend,
    SymbolicEvaluationStatus,
    TensorEngine,
    VerificationRecord,
    VerificationStatus,
    XActValidationStatus,
    derive_intermediate_quantities,
    draft4_circular_ansatz,
    latex_report,
    verify_run,
)


def _simple_model():
    return LagrangianSourceSpec(
        name="derived_simple",
        expression="R - X/2 - V(phi)",
        dimension=DimensionSpec(3),
        functions=(FunctionSpec("V", 1),),
    ).compile()


def _abstract_calculation():
    model = _simple_model()
    backend = StructuralTensorBackend.from_model(model)
    momenta = backend.derive_momenta(model.lagrangian)
    euler = backend.derive_euler_lagrange(model.lagrangian, momenta)
    verification = verify_run(model, momenta, euler)
    return model, backend, momenta, euler, verification


def test_derived_quantities_are_first_class_and_json_roundtrip() -> None:
    model, backend, momenta, euler, verification = _abstract_calculation()
    derived = derive_intermediate_quantities(
        model,
        momenta,
        euler,
        verification,
        backend,
    )
    assert derived.curvature_derivative_metric_term == (
        euler.curvature_derivative_metric_term
    )
    assert derived.record("riemann_tensor").symbolic_status is (
        SymbolicEvaluationStatus.GEOMETRIC_INPUT
    )
    assert derived.record("nabla_P").component_status is (
        ComponentProjectionStatus.NOT_REQUESTED
    )
    rebuilt = DerivedQuantities.from_data(json.loads(json.dumps(derived.to_data())))
    assert rebuilt == derived


def test_derived_projection_respects_the_supplied_draft4_ansatz() -> None:
    model = _simple_model()
    result = TensorEngine(options=EngineOptions(include_noether=False)).run(
        model,
        ansatz=draft4_circular_ansatz(),
    )
    assert result.derived is result.package.derived
    assert result.derived is not None
    assert result.abstract is result.package.abstract
    assert result.projected is result.package.projected
    assert result.abstract is not None
    assert result.projected is not None
    ricci = result.derived.record("ricci_scalar")
    assert ricci.component_status is ComponentProjectionStatus.PROJECTED
    assert ricci.components is not None
    assert ricci.components.dimension == 3
    assert ricci.components.scalar != 0
    nabla_p = result.derived.record("nabla_P")
    assert nabla_p.component_status is ComponentProjectionStatus.PROJECTED
    assert nabla_p.components is not None
    assert len(nabla_p.components.free_indices) == 5
    assert all(
        item.status.value == "completed" for item in result.projected.quantities
    )


def test_component_budget_preserves_abstract_expression_instead_of_failing() -> None:
    model, backend, momenta, euler, verification = _abstract_calculation()
    components = SympyComponentBackend.from_model(model, draft4_circular_ansatz())
    derived = derive_intermediate_quantities(
        model,
        momenta,
        euler,
        verification,
        backend,
        components,
        component_budget=10,
    )
    riemann = derived.record("riemann_tensor")
    assert riemann.component_status is ComponentProjectionStatus.BACKEND_LIMITATION
    assert riemann.components is None
    assert derived.riemann_tensor is not None


def test_xact_status_is_bound_to_the_specific_derivative_term_check() -> None:
    model, backend, momenta, euler, verification = _abstract_calculation()
    verification = replace(
        verification,
        checks=verification.checks
        + (
            VerificationRecord(
                "external.model.metric_euler_curvature_derivative_term",
                VerificationStatus.PASSED,
            ),
        ),
        external_bindings=(("verify_model", "a" * 64, "b" * 64),),
    )
    derived = derive_intermediate_quantities(
        model,
        momenta,
        euler,
        verification,
        backend,
    )
    assert derived.record("curvature_derivative_metric_term").xact_status is (
        XActValidationStatus.VALIDATED
    )
    assert derived.record("nabla_P").xact_status is XActValidationStatus.NOT_VALIDATED


def test_simple_lagrangian_integration_exports_derived_section(tmp_path) -> None:
    result = TensorEngine(options=EngineOptions(include_noether=False)).run(
        _simple_model(),
        ansatz=draft4_circular_ansatz(),
    )
    assert result.derived is not None
    report = latex_report(result.package)
    assert report.count(r"\section*{") == 2
    assert r"\section*{Expresiones tensoriales abstractas}" in report
    assert r"\section*{Expresiones proyectadas mediante el ansatz}" in report
    assert r"\section*{Resultados covariantes}" not in report
    assert r"\left.E_{ab}\right|_{\nabla\nabla P}" in report
    rebuilt = RunPackage.from_data(json.loads(json.dumps(result.package.to_data())))
    assert rebuilt.abstract == result.abstract
    assert rebuilt.projected == result.projected
    bundle = RunExporter(tmp_path, compile_pdf=False).export(result.package)
    data = json.loads((bundle.output_directory / "results.json").read_text(encoding="utf-8"))
    assert set(data["derived_quantities"]["expressions"]) == {
        "ricci_scalar",
        "riemann_tensor",
        "nabla_P",
        "nabla_nabla_P",
        "curvature_derivative_metric_term",
    }
    assert set(data["abstract_results"]["expressions"]) == {
        "lagrangian",
        "metric_momentum",
        "curvature_momentum",
        "scalar_gradient_momentum",
        "scalar_derivative",
        "metric_euler",
        "scalar_euler",
        "ricci_scalar",
        "riemann_tensor",
        "nabla_P",
        "nabla_nabla_P",
    }
    assert len(data["projected_results"]["quantities"]) == 11
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["result_views"]["abstract"]) == 11
    assert len(manifest["result_views"]["projected"]) == 11


def test_run_without_ansatz_preserves_all_projected_entries_as_symbolic() -> None:
    result = TensorEngine(options=EngineOptions(include_noether=False)).run(
        _simple_model()
    )
    assert result.projected is not None
    assert result.projected.ansatz_name is None
    assert all(item.status.value == "symbolic" for item in result.projected.quantities)


def test_one_projection_failure_does_not_abort_other_quantities(monkeypatch) -> None:
    model = _simple_model()
    target = StructuralTensorBackend.from_model(model).derive_momenta(
        model.lagrangian
    ).metric
    original = SympyComponentBackend.evaluate

    def selective_failure(self, expression):
        if expression == target:
            raise BackendExecutionError("limitación selectiva de prueba")
        return original(self, expression)

    monkeypatch.setattr(SympyComponentBackend, "evaluate", selective_failure)
    result = TensorEngine(options=EngineOptions(include_noether=False)).run(
        model,
        ansatz=draft4_circular_ansatz(),
    )
    assert result.projected is not None
    assert result.projected.metric_momentum.status.value == "unavailable"
    assert "limitación selectiva" in result.projected.metric_momentum.reason
    assert result.projected.lagrangian.status.value == "completed"
    assert result.projected.curvature_momentum.status.value == "completed"
