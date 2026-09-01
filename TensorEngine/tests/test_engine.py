from __future__ import annotations

import json

import pytest

from tensor_engine import (
    DEFAULT_PIPELINE,
    DimensionSpec,
    EngineOptions,
    ModelBuilder,
    ModelSpec,
    Number,
    ParameterSpec,
    PipelineExecutionError,
    RunEvent,
    Scalar,
    StageStatus,
    TensorEngine,
    Tensor,
    load_ansatz,
    load_model,
    save_ansatz,
    save_model,
    spatially_flat_flrw_ansatz,
    validate_stage_result,
    walk,
    WolframValidationReport,
    calculation_fingerprint,
    model_fingerprint,
)


def constant_model(*, normalization=Number(1)) -> ModelSpec:
    parameters = (ParameterSpec("kappa"),) if normalization == Scalar("kappa") else ()
    return ModelSpec(
        "phase10_constant",
        Number(1),
        dimension=DimensionSpec(4),
        normalization=normalization,
        parameters=parameters,
    )


def generic_report(
    model,
    momenta,
    euler,
    noether=None,
    *,
    calculation_hash=None,
    checks=None,
):
    normalized = model.lagrangian
    return WolframValidationReport.from_data({
        "schema_version": "1.2",
        "status": "success",
        "operation": "verify_model",
        "subject": {
            "model_name": model.name,
            "model_fingerprint": model_fingerprint(model),
            "calculation_fingerprint": calculation_hash or calculation_fingerprint(
                model, normalized, momenta, euler, noether
            ),
        },
        "runtime": {
            "wolfram_version": "15.0 test",
            "wolfram_version_number": 15.0,
            "wolfram_release_number": 0,
            "system_id": "Windows-x86-64",
        },
        "components": {
            name: {"available": True, "version": "test", "release_date": None}
            for name in ("xact_xtensor", "xact_xpert", "xact_xtras", "xact_xcoba")
        },
        "conventions": {},
        "checks": checks or [{
            "key": "bound_transport",
            "status": "passed",
            "message": "coincide",
            "residual": None,
        }],
    })


def test_engine_runs_required_pipeline_with_one_call() -> None:
    result = TensorEngine().run(constant_model())
    assert result.status is StageStatus.SUCCESS
    assert [stage.stage_key for stage in result.stages] == [
        "validate_model",
        "normalize_lagrangian",
        "derive_momenta",
        "raw_variation",
        "integrate_by_parts",
        "noether",
        "verify",
        "derive_intermediate_quantities",
        "organize_result_views",
    ]
    assert result.skipped_stages == ("components", "wolfram_model_validation", "export")
    specifications = {stage.key: stage for stage in DEFAULT_PIPELINE}
    for stage in result.stages:
        validate_stage_result(specifications[stage.stage_key], stage)


def test_engine_applies_global_action_normalization() -> None:
    result = TensorEngine().run(constant_model(normalization=Scalar("kappa")))
    assert result.package.model.lagrangian == Number(1)
    assert result.package.normalized_lagrangian == Scalar("kappa")
    lagrangian_stage = next(
        stage for stage in result.stages if stage.stage_key == "normalize_lagrangian"
    )
    assert lagrangian_stage.outputs[0].expression == Scalar("kappa")


def test_normalization_preserves_the_input_riemann_variance_contract() -> None:
    builder = ModelBuilder()
    ricci_scalar = (
        builder.metric("a", "c")
        * builder.metric("b", "d")
        * builder.riemann("a", "b", "c", "d")
    )
    model = ModelSpec("phase10_riemann_input", ricci_scalar, dimension=DimensionSpec(4))
    result = TensorEngine(options=EngineOptions(include_noether=False)).run(model)
    riemann_nodes = [
        node
        for node in walk(result.package.lagrangian)
        if isinstance(node, Tensor) and node.name == model.symbols.curvature
    ]
    assert riemann_nodes
    assert all(index.variance.value == "down" for node in riemann_nodes for index in node.indices)


def test_engine_can_disable_optional_noether_branch() -> None:
    engine = TensorEngine(options=EngineOptions(include_noether=False))
    result = engine.run(constant_model())
    assert result.package.noether is None
    assert "noether" in result.skipped_stages
    assert result.acceptable(strict=True)


def test_engine_exports_and_preserves_stage_trace(tmp_path) -> None:
    result = TensorEngine().run(constant_model(), output_root=tmp_path)
    assert result.export_bundle is not None
    assert result.stages[-1].stage_key == "export"
    assert result.export_bundle.manifest.verify_files(
        result.export_bundle.output_directory
    ) == ()
    summary = result.summary_data()
    assert summary["output_directory"] == str(result.export_bundle.output_directory.resolve())


def test_engine_projects_components_when_ansatz_is_supplied() -> None:
    ansatz = spatially_flat_flrw_ansatz()
    result = TensorEngine().run(
        constant_model(),
        ansatz=ansatz,
    )
    assert result.package.components is not None
    assert result.package.components.ansatz_name == ansatz.name
    assert "components" in [stage.stage_key for stage in result.stages]
    assert "components" not in result.skipped_stages


def test_engine_emits_ordered_observable_events() -> None:
    events: list[RunEvent] = []
    TensorEngine(event_handler=events.append).run(constant_model())
    assert events[0] == RunEvent("validate_model", "started")
    completed = [event.stage_key for event in events if event.state == "completed"]
    assert completed == [stage for stage in (
        "validate_model",
        "normalize_lagrangian",
        "derive_momenta",
        "raw_variation",
        "integrate_by_parts",
        "noether",
        "verify",
        "derive_intermediate_quantities",
        "organize_result_views",
    )]
    assert all(event.duration_seconds >= 0 for event in events)


def test_model_and_ansatz_file_roundtrip(tmp_path) -> None:
    model_path = save_model(constant_model(), tmp_path / "model.json")
    ansatz_path = save_ansatz(spatially_flat_flrw_ansatz(), tmp_path / "ansatz.json")
    assert load_model(model_path) == constant_model()
    assert load_ansatz(ansatz_path) == spatially_flat_flrw_ansatz()
    assert isinstance(json.loads(model_path.read_text(encoding="utf-8")), dict)


def test_partial_policy_is_explicit() -> None:
    # El modelo constante es success; se comprueban ambos modos sin ocultar el estado.
    result = TensorEngine().run(constant_model())
    assert result.acceptable()
    assert result.acceptable(strict=True)


def test_engine_integrates_a_bound_generic_wolfram_report() -> None:
    model = constant_model()

    class FakeBridge:
        def validate_model(self, active_model, momenta, euler, **options):
            return generic_report(active_model, momenta, euler, options.get("noether"))

    result = TensorEngine().run(model, wolfram_bridge=FakeBridge())
    assert "wolfram_model_validation" in [stage.stage_key for stage in result.stages]
    assert "wolfram_model_validation" not in result.skipped_stages
    assert result.package.verification.external_bindings[0][0] == "verify_model"
    assert "external.model.bound_transport" in {
        check.key for check in result.package.verification.checks
    }


def test_engine_adjudicates_only_explicit_bound_xact_evidence() -> None:
    from tests.test_model import scalar_tensor_model

    model = scalar_tensor_model()

    class FakeBridge:
        def validate_model(self, active_model, momenta, euler, **options):
            checks = [
                {
                    "key": key,
                    "status": "passed",
                    "message": "xAct redujo el residual a cero.",
                    "residual": None,
                    "strategy": "differential",
                    "adjudicates": [key],
                }
                for key in (
                    "noether_current_decomposition",
                    "diffeomorphism_noether_identity",
                )
            ]
            return generic_report(
                active_model,
                momenta,
                euler,
                options.get("noether"),
                checks=checks,
            )

    result = TensorEngine().run(model, wolfram_bridge=FakeBridge())
    verification = result.package.verification
    self_checks = {
        item.key: item.status.value
        for item in verification.checks
        if not item.key.startswith("external.")
    }
    assert verification.status is StageStatus.SUCCESS
    assert self_checks["noether_current_decomposition"] == "passed"
    assert self_checks["diffeomorphism_noether_identity"] == "passed"
    assert len(verification.adjudications) == 2


def test_engine_rejects_generic_evidence_from_another_calculation() -> None:
    model = constant_model()
    baseline = TensorEngine().run(model)
    mismatched = generic_report(
        model,
        baseline.package.momenta,
        baseline.package.euler,
        baseline.package.noether,
        calculation_hash="c" * 64,
    )
    with pytest.raises(PipelineExecutionError, match="otro modelo o cálculo"):
        TensorEngine().run(model, external_reports=(mismatched,))
