from __future__ import annotations

import json
import unittest

from tensor_engine import (
    ComponentFieldEquations,
    DimensionSpec,
    EulerLagrangeResult,
    LagrangianMomenta,
    ModelSpec,
    Number,
    Scalar,
    StageStatus,
    StructuralTensorBackend,
    SympyComponentBackend,
    VerificationReport,
    VerificationRecord,
    VerificationStatus,
    WolframValidationReport,
    adjudicate_external_evidence,
    add,
    evaluate_field_equations,
    spatially_flat_flrw_ansatz,
    verify_run,
    DEFAULT_PIPELINE,
    validate_stage_result,
)
from tests.test_model import scalar_tensor_model


def constant_run():
    model = ModelSpec("constant_reference", Number(1), dimension=DimensionSpec(4))
    backend = StructuralTensorBackend.from_model(model)
    momenta = backend.derive_momenta(model.lagrangian)
    euler = backend.derive_euler_lagrange(model.lagrangian, momenta)
    noether = backend.derive_noether_wald(model.lagrangian, momenta, euler)
    raw = backend.raw_lagrangian_variation(momenta)
    return model, backend, momenta, euler, noether, raw


class VerificationSuiteTests(unittest.TestCase):
    def test_constant_model_passes_all_available_checks(self) -> None:
        model, _, momenta, euler, noether, raw = constant_run()
        report = verify_run(
            model,
            momenta,
            euler,
            noether=noether,
            raw_variation=raw,
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.status, StageStatus.SUCCESS)
        self.assertEqual(report.summary["failed"], 0)
        self.assertEqual(report.summary["undetermined"], 0)

    def test_general_scalar_tensor_preserves_undetermined_identities(self) -> None:
        model = scalar_tensor_model()
        backend = StructuralTensorBackend.from_model(model)
        momenta = backend.derive_momenta(model.lagrangian)
        euler = backend.derive_euler_lagrange(model.lagrangian, momenta)
        noether = backend.derive_noether_wald(model.lagrangian, momenta, euler)
        report = verify_run(model, momenta, euler, noether=noether)
        self.assertEqual(report.status, StageStatus.PARTIAL)
        self.assertEqual(report.summary["failed"], 0)
        self.assertGreaterEqual(report.summary["undetermined"], 2)
        self.assertTrue(all(
            item.residual is not None
            for item in report.checks
            if item.status is VerificationStatus.UNDETERMINED
        ))

    def test_corrupted_boundary_is_reported_as_failure(self) -> None:
        model, _, momenta, euler, _, _ = constant_run()
        corrupted = EulerLagrangeResult(
            euler.metric_euler,
            euler.scalar_euler,
            euler.boundary_metric,
            euler.boundary_scalar,
            Number(1),
            euler.full_variation,
            euler.density_variation,
        )
        report = verify_run(model, momenta, corrupted)
        self.assertEqual(report.status, StageStatus.FAILED)
        failed = {item.key for item in report.checks if item.status is VerificationStatus.FAILED}
        self.assertIn("boundary.total", failed)
        self.assertIn("recompute_euler.boundary_total", failed)

    def test_undeclared_symbol_is_not_silently_accepted(self) -> None:
        model, _, momenta, euler, _, _ = constant_run()
        corrupted = LagrangianMomenta(
            momenta.metric,
            momenta.curvature,
            momenta.scalar_gradient,
            Scalar("undeclared"),
        )
        report = verify_run(model, corrupted, euler)
        check = next(item for item in report.checks if item.key == "symbols.declared")
        self.assertEqual(check.status, VerificationStatus.FAILED)
        self.assertIsNotNone(check.residual)

    def test_component_projection_is_recomputed(self) -> None:
        model, _, momenta, euler, _, _ = constant_run()
        component_backend = SympyComponentBackend.from_model(
            model,
            spatially_flat_flrw_ansatz(),
        )
        components = evaluate_field_equations(
            euler.metric_euler,
            euler.scalar_euler,
            component_backend,
        )
        report = verify_run(
            model,
            momenta,
            euler,
            components=components,
            component_backend=component_backend,
        )
        component_checks = [item for item in report.checks if item.key.startswith("components.")]
        self.assertEqual(len(component_checks), 3)
        self.assertTrue(all(item.status is VerificationStatus.PASSED for item in component_checks))

    def test_component_arguments_must_be_provided_together(self) -> None:
        model, _, momenta, euler, _, _ = constant_run()
        empty_components = ComponentFieldEquations(
            "unused",
            SympyComponentBackend.from_model(model, spatially_flat_flrw_ansatz()).evaluate(
                euler.metric_euler
            ),
            (),
            SympyComponentBackend.from_model(model, spatially_flat_flrw_ansatz()).evaluate(
                euler.scalar_euler
            ),
        )
        with self.assertRaises(ValueError):
            verify_run(model, momenta, euler, components=empty_components)

    def test_report_json_roundtrip(self) -> None:
        model, _, momenta, euler, _, _ = constant_run()
        report = verify_run(model, momenta, euler)
        encoded = json.loads(json.dumps(report.to_data()))
        self.assertEqual(VerificationReport.from_data(encoded), report)

    def test_report_satisfies_verify_stage_contract(self) -> None:
        model, _, momenta, euler, _, _ = constant_run()
        report = verify_run(model, momenta, euler)
        stage_result = report.to_stage_result(duration_seconds=0.5)
        specification = next(item for item in DEFAULT_PIPELINE if item.key == "verify")
        validate_stage_result(specification, stage_result)
        self.assertEqual(stage_result.status, StageStatus.SUCCESS)

    def test_external_wolfram_checks_are_namespaced(self) -> None:
        model, _, momenta, euler, _, _ = constant_run()
        data = {
            "schema_version": "1.1",
            "status": "success",
            "operation": "verify_phase7",
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
            "checks": [{
                "key": "flrw_metric_inverse",
                "status": "passed",
                "message": "coincide",
                "residual": None,
            }],
        }
        external = WolframValidationReport.from_data(data)
        report = verify_run(model, momenta, euler, external_reports=(external,))
        self.assertIn(
            "external.phase7.flrw_metric_inverse",
            {item.key for item in report.checks},
        )
        self.assertEqual(report.external_sources[0][0], "verify_phase7")

    def test_adjudication_requires_all_explicit_evidence_to_pass(self) -> None:
        internal = VerificationReport(
            "bound_model",
            "structural",
            "test",
            (
                VerificationRecord(
                    "diffeomorphism_noether_identity",
                    VerificationStatus.UNDETERMINED,
                    Scalar("internal_residual"),
                    "Requiere Bianchi diferencial.",
                ),
            ),
        )

        def external(status: str, residual):
            return WolframValidationReport.from_data({
                "schema_version": "1.3",
                "status": "success" if status == "passed" else "partial",
                "operation": "verify_model",
                "subject": {
                    "model_name": "bound_model",
                    "model_fingerprint": "a" * 64,
                    "calculation_fingerprint": "b" * 64,
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
                "checks": [{
                    "key": "diffeomorphism_noether_identity",
                    "status": status,
                    "message": "prueba xAct",
                    "residual": residual,
                    "strategy": "differential",
                    "adjudicates": ["diffeomorphism_noether_identity"],
                }],
            })

        passed = adjudicate_external_evidence(internal, (external("passed", None),))
        self.assertEqual(passed.status, StageStatus.SUCCESS)
        self.assertEqual(len(passed.adjudications), 1)
        undecided = adjudicate_external_evidence(
            internal,
            (external("passed", None), external("undetermined", "residual")),
        )
        self.assertEqual(undecided.status, StageStatus.PARTIAL)
        self.assertEqual(undecided.adjudications, ())


if __name__ == "__main__":
    unittest.main()
