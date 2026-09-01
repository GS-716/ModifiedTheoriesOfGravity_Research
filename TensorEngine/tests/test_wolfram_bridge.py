from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from tensor_engine import (
    BackendUnavailableError,
    DimensionSpec,
    ModelSpec,
    ModelBuilder,
    Number,
    StructuralTensorBackend,
    TensorDeclaration,
    TensorSymmetry,
    Variance,
    VerificationStatus,
    WolframPhase5Report,
    WolframXActBridge,
    calculation_fingerprint,
    detect_wolfram_runtime,
    model_fingerprint,
)


class WolframBridgeTests(unittest.TestCase):
    @staticmethod
    def constant_calculation():
        model = ModelSpec("phase11_constant", Number(1), dimension=DimensionSpec(4))
        backend = StructuralTensorBackend.from_model(model)
        momenta = backend.derive_momenta(model.lagrangian)
        euler = backend.derive_euler_lagrange(model.lagrangian, momenta)
        return model, momenta, euler

    def test_missing_runtime_is_reported_without_crashing(self) -> None:
        status = detect_wolfram_runtime("C:\\missing-wolfram\\wolframscript.exe")
        self.assertFalse(status.available)
        self.assertIsNotNone(status.reason)

    def test_request_is_json_serializable(self) -> None:
        bridge = WolframXActBridge(executable="C:\\missing-wolfram\\wolframscript.exe")
        expression = ModelBuilder().metric("a", "b")
        declaration = TensorDeclaration(
            "g",
            (Variance.UP, Variance.UP),
            TensorSymmetry.SYMMETRIC,
        )
        request = bridge.build_request("canonicalize", expression, (declaration,))
        encoded = json.dumps(request)
        self.assertIn('"operation": "canonicalize"', encoded)

    def test_unavailable_runtime_raises_controlled_error(self) -> None:
        bridge = WolframXActBridge(executable="C:\\missing-wolfram\\wolframscript.exe")
        with self.assertRaises(BackendUnavailableError):
            bridge.ping()

    def test_bridge_script_exists_in_source_tree(self) -> None:
        bridge = WolframXActBridge(executable="C:\\missing-wolfram\\wolframscript.exe")
        self.assertTrue(Path(bridge.script_path).is_file())

    def test_invalid_operation_is_rejected(self) -> None:
        bridge = WolframXActBridge(executable="C:\\missing-wolfram\\wolframscript.exe")
        with self.assertRaises(ValueError):
            bridge.build_request("Invalid Operation")

    def test_phase_five_report_has_typed_versions_and_checks(self) -> None:
        data = {
            "schema_version": "1.1",
            "status": "success",
            "operation": "verify_phase5",
            "runtime": {
                "wolfram_version": "15.0 test",
                "wolfram_version_number": 15.0,
                "wolfram_release_number": 0,
                "system_id": "Windows-x86-64",
            },
            "components": {
                "xact_xtensor": {
                    "available": True,
                    "version": "1.3.0",
                    "release_date": [2025, 12, 29],
                },
                "xact_xpert": {
                    "available": True,
                    "version": "1.0.6",
                    "release_date": [2018, 2, 28],
                },
                "xact_xtras": {
                    "available": True,
                    "version": "1.4.2",
                    "release_date": [2014, 10, 30],
                },
            },
            "conventions": {
                "tensor_engine_riemann_map": "R_TE^a_bcd = -R_xAct_cd b^a"
            },
            "checks": [
                {
                    "key": "palatini_mixed_xpert",
                    "status": "passed",
                    "message": "Palatini coincide.",
                    "residual": None,
                }
            ],
        }
        report = WolframPhase5Report.from_data(data)
        self.assertTrue(report.passed)
        self.assertEqual(report.wolfram_version_number, 15.0)
        self.assertEqual(report.xact_xtensor.version, "1.3.0")
        self.assertEqual(report.xact_xpert.version, "1.0.6")
        self.assertEqual(report.xact_xtras.version, "1.4.2")
        self.assertEqual(report.summary["passed"], 1)
        self.assertEqual(
            report.verification_records[0].status,
            VerificationStatus.PASSED,
        )
        self.assertEqual(WolframPhase5Report.from_data(report.to_data()), report)

    def test_model_and_calculation_fingerprints_are_stable_and_distinct(self) -> None:
        model, momenta, euler = self.constant_calculation()
        self.assertEqual(model_fingerprint(model), model_fingerprint(model))
        self.assertEqual(len(model_fingerprint(model)), 64)
        calculation = calculation_fingerprint(
            model, model.lagrangian, momenta, euler
        )
        self.assertEqual(len(calculation), 64)
        changed = ModelSpec("phase11_changed", Number(1), dimension=DimensionSpec(4))
        self.assertNotEqual(model_fingerprint(model), model_fingerprint(changed))

    def test_generic_request_is_bound_to_model_and_calculation(self) -> None:
        model, momenta, euler = self.constant_calculation()
        bridge = WolframXActBridge(executable="C:\\missing-wolfram\\wolframscript.exe")
        request = bridge.build_model_validation_request(model, momenta, euler)
        self.assertEqual(request["operation"], "verify_model")
        subject = request["options"]["subject"]
        self.assertEqual(subject["model_name"], model.name)
        self.assertEqual(subject["model_fingerprint"], model_fingerprint(model))
        self.assertEqual(
            subject["calculation_fingerprint"],
            calculation_fingerprint(model, model.lagrangian, momenta, euler),
        )
        self.assertEqual(len(request["options"]["checks"]), 9)
        strategies = {item["strategy"] for item in request["options"]["checks"]}
        self.assertEqual(strategies, {"algebraic", "riemann_bianchi", "differential"})
        json.dumps(request)

    def test_check_strategy_and_adjudication_round_trip(self) -> None:
        data = {
            "schema_version": "1.3",
            "status": "partial",
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
                "status": "undetermined",
                "message": "identidad diferencial",
                "residual": "residual",
                "strategy": "differential",
                "adjudicates": ["diffeomorphism_noether_identity"],
            }],
        }
        report = WolframPhase5Report.from_data(data)
        check = report.checks[0]
        self.assertEqual(check.strategy, "differential")
        self.assertEqual(check.adjudicates, ("diffeomorphism_noether_identity",))
        self.assertEqual(WolframPhase5Report.from_data(report.to_data()), report)

    def test_generic_report_requires_valid_subject_fingerprints(self) -> None:
        data = {
            "schema_version": "1.2",
            "status": "success",
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
                "key": "transport",
                "status": "passed",
                "message": "coincide",
                "residual": None,
            }],
        }
        report = WolframPhase5Report.from_data(data)
        self.assertEqual(report.operation, "verify_model")
        self.assertEqual(report.model_fingerprint, "a" * 64)
        self.assertEqual(WolframPhase5Report.from_data(report.to_data()), report)
        data["subject"]["model_fingerprint"] = "invalid"
        with self.assertRaisesRegex(Exception, "fingerprints"):
            WolframPhase5Report.from_data(data)

    @unittest.skipUnless(
        os.environ.get("TENSOR_ENGINE_RUN_WOLFRAM_TESTS") == "1",
        "La integración Wolfram/xAct es opt-in.",
    )
    def test_live_phase_five_validation_passes(self) -> None:
        bridge = WolframXActBridge(timeout_seconds=180)
        report = bridge.validate_phase5()
        self.assertTrue(report.xact_xtensor.available)
        self.assertTrue(report.xact_xpert.available)
        self.assertTrue(report.xact_xtras.available)
        self.assertTrue(report.passed, report.summary)

    @unittest.skipUnless(
        os.environ.get("TENSOR_ENGINE_RUN_WOLFRAM_TESTS") == "1",
        "La integración Wolfram/xAct es opt-in.",
    )
    def test_live_phase_six_validation_passes(self) -> None:
        report = WolframXActBridge(timeout_seconds=180).validate_phase6()
        self.assertEqual(report.operation, "verify_phase6")
        self.assertTrue(report.xact_xtensor.available)
        self.assertTrue(report.xact_xtras.available)
        self.assertTrue(report.passed, report.summary)

    @unittest.skipUnless(
        os.environ.get("TENSOR_ENGINE_RUN_WOLFRAM_TESTS") == "1",
        "La integración Wolfram/xAct es opt-in.",
    )
    def test_live_phase_seven_validation_passes(self) -> None:
        report = WolframXActBridge(timeout_seconds=180).validate_phase7()
        self.assertEqual(report.operation, "verify_phase7")
        self.assertTrue(report.xact_xtensor.available)
        self.assertTrue(report.xact_xcoba.available)
        self.assertTrue(report.passed, report.summary)

    @unittest.skipUnless(
        os.environ.get("TENSOR_ENGINE_RUN_WOLFRAM_TESTS") == "1",
        "La integración Wolfram/xAct es opt-in.",
    )
    def test_live_generic_model_validation_is_bound_and_has_no_failures(self) -> None:
        from tests.test_model import scalar_tensor_model

        model = scalar_tensor_model()
        backend = StructuralTensorBackend.from_model(model)
        momenta = backend.derive_momenta(model.lagrangian)
        euler = backend.derive_euler_lagrange(model.lagrangian, momenta)
        noether = backend.derive_noether_wald(model.lagrangian, momenta, euler)
        report = WolframXActBridge(timeout_seconds=180).validate_model(
            model,
            momenta,
            euler,
            noether=noether,
        )
        self.assertEqual(report.operation, "verify_model")
        self.assertEqual(report.model_name, model.name)
        self.assertEqual(report.summary["failed"], 0, report.to_data())
        self.assertGreater(report.summary["passed"], 0)
        self.assertEqual(len(report.checks), 12)
        self.assertEqual(
            {item.strategy for item in report.checks},
            {"algebraic", "riemann_bianchi", "differential"},
        )


if __name__ == "__main__":
    unittest.main()
