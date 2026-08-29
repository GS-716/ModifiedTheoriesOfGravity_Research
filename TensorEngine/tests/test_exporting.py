from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from tensor_engine import (
    DEFAULT_PIPELINE,
    DimensionSpec,
    FunctionSpec,
    ModelBuilder,
    ModelSpec,
    Number,
    RunExporter,
    RunManifest,
    RunPackage,
    Scalar,
    StructuralTensorBackend,
    Tensor,
    VerificationRecord,
    VerificationReport,
    VerificationStatus,
    Variance,
    Variation,
    expr_to_latex,
    function,
    latex_report,
    validate_stage_result,
    verify_run,
)
from tensor_engine.ir import Index, CovariantDerivative


def make_package() -> RunPackage:
    builder = ModelBuilder()
    lagrangian = function("V", builder.phi)
    model = ModelSpec(
        name="phase9_export_test",
        lagrangian=lagrangian,
        dimension=DimensionSpec(4),
        functions=(FunctionSpec("V"),),
    )
    backend = StructuralTensorBackend.from_model(model)
    momenta = backend.derive_momenta(model.lagrangian)
    raw = backend.raw_lagrangian_variation(momenta)
    euler = backend.derive_euler_lagrange(model.lagrangian, momenta)
    report = verify_run(model, momenta, euler, raw_variation=raw)
    return RunPackage(
        model,
        momenta,
        raw,
        euler,
        report,
        duration_seconds=1.25,
        stage_durations=(("derive_momenta", 0.25), ("verify", 1.0)),
    )


def test_latex_printer_covers_tensor_derivative_and_variation() -> None:
    a_down = Index("a", Variance.DOWN)
    b_up = Index("b", Variance.UP)
    tensor = Tensor("Riemann", (a_down, b_up))
    rendered = expr_to_latex(CovariantDerivative(a_down, tensor))
    assert rendered == r"\nabla_{a}\!\left(R{}_{a}{}^{b}\right)"
    assert expr_to_latex(Variation(Scalar("phi"))) == r"\delta\!\left(\phi\right)"
    assert expr_to_latex(Number(-1, 2)) == r"-\frac{1}{2}"


def test_latex_report_escapes_machine_identifiers() -> None:
    package = make_package()
    package = replace(
        package,
        verification=replace(package.verification, backend_name="structural_python"),
    )
    report = latex_report(package)
    assert r"run\_" in report
    assert r"structural\_python" in report


def test_run_id_is_content_addressed_and_ignores_timing() -> None:
    package = make_package()
    changed_timing = replace(package, duration_seconds=99.0, stage_durations=())
    assert changed_timing.run_id == package.run_id
    changed_model = replace(package.model, name="another_model")
    changed_report = replace(package.verification, model_name="another_model")
    assert replace(package, model=changed_model, verification=changed_report).run_id != package.run_id


def test_run_package_json_roundtrip_preserves_ir_and_identity() -> None:
    package = make_package()
    rebuilt = RunPackage.from_data(json.loads(json.dumps(package.to_data())))
    assert rebuilt == package
    assert rebuilt.run_id == package.run_id


def test_run_package_rejects_tampered_run_id() -> None:
    data = make_package().to_data()
    data["run_id"] = "run_tampered"
    with pytest.raises(ValueError, match="run_id"):
        RunPackage.from_data(data)


def test_export_creates_auditable_files_with_valid_hashes(tmp_path) -> None:
    package = make_package()
    bundle = RunExporter(tmp_path).export(package, created_at_utc="2026-08-28T12:00:00Z")
    assert bundle.manifest_path.is_file()
    assert {item.relative_path for item in bundle.manifest.files} == {
        "results.json",
        "verification.json",
        "report.tex",
    }
    for item in bundle.manifest.files:
        content = (bundle.output_directory / item.relative_path).read_bytes()
        assert hashlib.sha256(content).hexdigest() == item.sha256
        assert len(content) == item.size_bytes
    loaded = RunManifest.from_data(json.loads(bundle.manifest_path.read_text(encoding="utf-8")))
    assert loaded == bundle.manifest
    assert loaded.verify_files(bundle.output_directory) == ()


def test_manifest_detects_a_modified_artifact(tmp_path) -> None:
    bundle = RunExporter(tmp_path).export(make_package())
    (bundle.output_directory / "verification.json").write_text("altered", encoding="utf-8")
    incidents = bundle.manifest.verify_files(bundle.output_directory)
    assert any("tamaño" in item for item in incidents)
    assert any("SHA-256" in item for item in incidents)


def test_export_is_deterministic_with_injected_timestamp(tmp_path) -> None:
    exporter = RunExporter(tmp_path)
    first = exporter.export(make_package(), created_at_utc="2026-08-28T12:00:00Z")
    first_manifest = first.manifest_path.read_bytes()
    second = exporter.export(make_package(), created_at_utc="2026-08-28T12:00:00Z")
    assert second.output_directory == first.output_directory
    assert second.manifest_path.read_bytes() == first_manifest


def test_export_directory_uses_safe_model_slug(tmp_path) -> None:
    package = make_package()
    model = replace(package.model, name="Model_Name_99")
    report = replace(package.verification, model_name=model.name)
    bundle = RunExporter(tmp_path).export(replace(package, model=model, verification=report))
    assert bundle.output_directory.name.startswith("model-name-99-")
    assert bundle.output_directory.parent == tmp_path.resolve()


def test_export_stage_result_satisfies_declared_contract(tmp_path) -> None:
    bundle = RunExporter(tmp_path).export(make_package())
    result = bundle.to_stage_result(duration_seconds=0.2)
    export_stage = next(stage for stage in DEFAULT_PIPELINE if stage.key == "export")
    validate_stage_result(export_stage, result)
    assert {item.key for item in result.artifacts} == {"run_manifest", "exported_artifacts"}


def test_partial_or_failed_verification_is_not_promoted_to_success(tmp_path) -> None:
    package = make_package()
    partial_report = VerificationReport(
        package.model.name,
        "test",
        "1",
        (VerificationRecord("unknown", VerificationStatus.UNDETERMINED, Number(1)),),
    )
    partial = RunExporter(tmp_path / "partial").export(
        replace(package, verification=partial_report)
    ).to_stage_result()
    assert partial.status.value == "partial"
    assert partial.diagnostics[0].severity.value == "warning"

    failed_report = replace(
        partial_report,
        checks=(VerificationRecord("failure", VerificationStatus.FAILED, Number(1)),),
    )
    failed = RunExporter(tmp_path / "failed").export(
        replace(package, verification=failed_report)
    ).to_stage_result()
    assert failed.status.value == "failed"
    assert failed.diagnostics[0].severity.value == "error"


def test_results_json_is_the_canonical_reconstructible_source(tmp_path) -> None:
    package = make_package()
    bundle = RunExporter(tmp_path).export(package)
    data = json.loads((bundle.output_directory / "results.json").read_text(encoding="utf-8"))
    rebuilt = RunPackage.from_data(data)
    assert rebuilt.model == package.model
    assert rebuilt.momenta == package.momenta
    assert rebuilt.euler == package.euler
    assert rebuilt.verification == package.verification
