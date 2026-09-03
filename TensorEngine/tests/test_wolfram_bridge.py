from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from tensor_engine import (
    BackendUnavailableError,
    BackendExecutionError,
    DimensionSpec,
    LagrangianSourceSpec,
    ModelSpec,
    ModelBuilder,
    Number,
    ParameterSpec,
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

    @staticmethod
    def case2_underscored_calculation(*, include_noether: bool = False):
        source = LagrangianSourceSpec(
            name="eqt_case2_draft4",
            expression=(
                "R + 2/ell**2 - alpha_1*X "
                "+ ell**2*beta0*(3*RicciUU - X*R)"
            ),
            dimension=DimensionSpec(3),
            parameters=tuple(
                ParameterSpec(name) for name in ("alpha_1", "ell", "beta0", "p")
            ),
            assumptions=("ell != 0", "beta0 != 0"),
        )
        model = source.compile()
        backend = StructuralTensorBackend.from_model(model)
        momenta = backend.derive_momenta(model.lagrangian)
        euler = backend.derive_euler_lagrange(model.lagrangian, momenta)
        noether = (
            backend.derive_noether_wald(model.lagrangian, momenta, euler)
            if include_noether
            else None
        )
        return model, momenta, euler, noether

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
        data["checks"][0]["residual"] = "residual no reducido"
        with self.assertRaisesRegex(
            BackendExecutionError, "aprobada no puede conservar residual"
        ):
            WolframPhase5Report.from_data(data)

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

    def test_generic_request_preserves_underscored_ir_names_and_json(self) -> None:
        model, momenta, euler, _ = self.case2_underscored_calculation()
        bridge = WolframXActBridge(executable="C:\\missing-wolfram\\wolframscript.exe")
        request = bridge.build_model_validation_request(model, momenta, euler)
        encoded = json.dumps(request, ensure_ascii=False, sort_keys=True)
        self.assertIn('"name": "alpha_1"', encoded)
        self.assertEqual(
            [item["name"] for item in request["options"]["model"]["parameters"]],
            ["alpha_1", "ell", "beta0", "p"],
        )
        self.assertNotIn("IR decode failed", encoded)

    def test_quadratic_curvature_ir_is_transportable_and_json_serializable(self) -> None:
        bridge = WolframXActBridge(executable="C:\\missing-wolfram\\wolframscript.exe")
        for alias in ("RicciSq", "RiemannSq"):
            model = LagrangianSourceSpec(
                "quadratic_" + alias,
                f"R + alpha*{alias}",
                parameters=(ParameterSpec("alpha"),),
            ).compile()
            backend = StructuralTensorBackend.from_model(model)
            momenta = backend.derive_momenta(model.lagrangian)
            euler = backend.derive_euler_lagrange(model.lagrangian, momenta)
            request = bridge.build_model_validation_request(model, momenta, euler)
            encoded = json.dumps(request, ensure_ascii=False, sort_keys=True)
            encoded_expression = json.dumps(
                request["expression"], ensure_ascii=False, sort_keys=True
            )

            self.assertIn('"name": "Riemann"', encoded)
            self.assertIn('"name": "g"', encoded)
            # La procedencia conserva el alias, pero el árbol matemático enviado
            # a xAct contiene únicamente nodos de la IR tensorial canónica.
            self.assertNotIn(alias, encoded_expression)
            self.assertNotIn("IR decode failed", encoded)

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
                "transport_diagnostic": {
                    "code": "undeclared_tensor",
                    "reason": "tensor no declarado: Mystery",
                    "category": "tensor",
                    "path": ["residual", "terms", "1"],
                    "node_type": "tensor",
                    "symbol": "Mystery",
                    "fragment": {
                        "type": "tensor",
                        "name": "Mystery",
                        "indices": [],
                    },
                },
            }],
        }
        report = WolframPhase5Report.from_data(data)
        check = report.checks[0]
        self.assertEqual(check.strategy, "differential")
        self.assertEqual(check.adjudicates, ("diffeomorphism_noether_identity",))
        self.assertEqual(check.diagnostic.code, "undeclared_tensor")
        self.assertEqual(check.diagnostic.path, ("residual", "terms", "1"))
        self.assertEqual(check.diagnostic.fragment["name"], "Mystery")
        record = check.to_verification_record()
        self.assertEqual(record.diagnostic, check.diagnostic)
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

    @unittest.skipUnless(
        os.environ.get("TENSOR_ENGINE_RUN_WOLFRAM_TESTS") == "1",
        "La integración Wolfram/xAct es opt-in.",
    )
    def test_live_case2_underscored_parameter_transports_all_checks(self) -> None:
        model, momenta, euler, noether = self.case2_underscored_calculation(
            include_noether=True
        )
        report = WolframXActBridge(timeout_seconds=300).validate_model(
            model,
            momenta,
            euler,
            noether=noether,
        )
        self.assertEqual(report.summary, {"passed": 11, "failed": 0, "undetermined": 1})
        self.assertTrue(
            all(
                item.status is VerificationStatus.PASSED
                for item in report.checks
                if item.strategy in {"algebraic", "riemann_bianchi"}
            ),
            report.to_data(),
        )
        self.assertFalse(
            any("IR decode failed" in (item.residual or "") for item in report.checks)
        )
        self.assertTrue(all(item.diagnostic is None for item in report.checks))

    @unittest.skipUnless(
        os.environ.get("TENSOR_ENGINE_RUN_WOLFRAM_TESTS") == "1",
        "La integración Wolfram/xAct es opt-in.",
    )
    def test_live_quadratic_curvature_models_have_no_transport_failures(self) -> None:
        bridge = WolframXActBridge(timeout_seconds=300)
        for alias in ("RicciSq", "RiemannSq"):
            model = LagrangianSourceSpec(
                "quadratic_" + alias,
                f"R + alpha*{alias}",
                parameters=(ParameterSpec("alpha"),),
            ).compile()
            backend = StructuralTensorBackend.from_model(model)
            momenta = backend.derive_momenta(model.lagrangian)
            euler = backend.derive_euler_lagrange(model.lagrangian, momenta)
            report = bridge.validate_model(model, momenta, euler)

            self.assertEqual(report.summary["failed"], 0, report.to_data())
            self.assertGreater(report.summary["passed"], 0, report.to_data())
            self.assertFalse(
                any("IR decode failed" in (item.residual or "") for item in report.checks)
            )
            self.assertTrue(all(item.diagnostic is None for item in report.checks))

    @unittest.skipUnless(
        os.environ.get("TENSOR_ENGINE_RUN_WOLFRAM_TESTS") == "1",
        "La integración Wolfram/xAct es opt-in.",
    )
    def test_live_unsupported_ir_is_never_reported_as_passed(self) -> None:
        model, momenta, euler = self.constant_calculation()
        bridge = WolframXActBridge(timeout_seconds=180)
        request = bridge.build_model_validation_request(model, momenta, euler)
        request["options"]["checks"] = [
            {
                "key": "unsupported_node",
                "message": "Un nodo desconocido no constituye una identidad validada.",
                "residual": {
                    "type": "add",
                    "terms": [
                        {"type": "number", "numerator": 1, "denominator": 1},
                        {"type": "future_ir_node", "payload": {"source": "test"}},
                    ],
                },
                "on_nonzero": "failed",
                "strategy": "algebraic",
                "adjudicates": [],
            },
            {
                "key": "undeclared_tensor",
                "message": "Un tensor desconocido debe identificarse.",
                "residual": {"type": "tensor", "name": "Mystery", "indices": []},
                "on_nonzero": "failed",
                "strategy": "algebraic",
                "adjudicates": [],
            },
            {
                "key": "invalid_index",
                "message": "Un índice desconocido debe identificarse.",
                "residual": {
                    "type": "tensor",
                    "name": "g",
                    "indices": [
                        {"name": "ghost", "variance": "up", "space": "M"},
                        {"name": "teFallbackA", "variance": "down", "space": "M"},
                    ],
                },
                "on_nonzero": "failed",
                "strategy": "algebraic",
                "adjudicates": [],
            },
            {
                "key": "malformed_expression",
                "message": "Una expresión malformada debe conservarse.",
                "residual": ["not", "an", "association"],
                "on_nonzero": "failed",
                "strategy": "algebraic",
                "adjudicates": [],
            },
        ]
        report = WolframPhase5Report.from_data(bridge.execute(request))
        self.assertEqual(report.summary, {"passed": 0, "failed": 0, "undetermined": 4})
        self.assertTrue(
            all(item.status is VerificationStatus.UNDETERMINED for item in report.checks)
        )
        diagnostics = {item.key: item.diagnostic for item in report.checks}
        self.assertEqual(diagnostics["unsupported_node"].code, "unsupported_node_type")
        self.assertEqual(
            diagnostics["unsupported_node"].path,
            ("residual", "terms", "1"),
        )
        self.assertEqual(
            diagnostics["unsupported_node"].fragment["payload"],
            {"source": "test"},
        )
        self.assertEqual(diagnostics["undeclared_tensor"].category, "tensor")
        self.assertEqual(diagnostics["undeclared_tensor"].symbol, "Mystery")
        self.assertEqual(diagnostics["invalid_index"].category, "index")
        self.assertEqual(diagnostics["invalid_index"].symbol, "ghost")
        self.assertEqual(
            diagnostics["invalid_index"].path,
            ("residual", "indices", "0"),
        )
        self.assertEqual(
            diagnostics["malformed_expression"].fragment,
            ["not", "an", "association"],
        )
        self.assertTrue(
            all("IR transport rejected at" in (item.residual or "") for item in report.checks)
        )


if __name__ == "__main__":
    unittest.main()
