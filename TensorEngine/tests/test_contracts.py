from __future__ import annotations

import json
import unittest

from tensor_engine import (
    ArtifactRecord,
    ContractValidationError,
    Diagnostic,
    Number,
    Severity,
    StageResult,
    StageStatus,
    VerificationRecord,
    VerificationStatus,
    validate_pipeline,
    validate_stage_result,
)
from tensor_engine.contracts import StageSpec


class ContractTests(unittest.TestCase):
    def test_default_pipeline_is_valid(self) -> None:
        validate_pipeline()

    def test_failed_verification_requires_residual(self) -> None:
        with self.assertRaises(ContractValidationError):
            VerificationRecord("test", VerificationStatus.FAILED)

    def test_undetermined_verification_preserves_residual(self) -> None:
        record = VerificationRecord(
            "test",
            VerificationStatus.UNDETERMINED,
            residual=Number(1),
        )
        self.assertEqual(record.residual, Number(1))

    def test_failed_stage_requires_diagnostic(self) -> None:
        with self.assertRaises(ContractValidationError):
            StageResult("stage", StageStatus.FAILED, "test", "operation")

    def test_failed_stage_with_diagnostic_is_valid(self) -> None:
        result = StageResult(
            "stage",
            StageStatus.FAILED,
            "test",
            "operation",
            diagnostics=(Diagnostic("E_TEST", "fallo controlado", Severity.ERROR),),
        )
        self.assertEqual(result.status, StageStatus.FAILED)

    def test_successful_result_must_produce_declared_outputs(self) -> None:
        spec = StageSpec("validate", ("model_spec",), ("validated_model",))
        result = StageResult(
            "validate",
            StageStatus.SUCCESS,
            "python",
            "validate",
            inputs=("model_spec",),
        )
        with self.assertRaises(ContractValidationError):
            validate_stage_result(spec, result)

    def test_structured_artifact_satisfies_stage_contract(self) -> None:
        spec = StageSpec("validate", ("model_spec",), ("validated_model",))
        artifact = ArtifactRecord(
            "validated_model",
            "model_spec",
            (("schema_version", "1.0"),),
            ("model_spec",),
        )
        result = StageResult(
            "validate",
            StageStatus.SUCCESS,
            "python",
            "validate",
            inputs=("model_spec",),
            artifacts=(artifact,),
        )
        validate_stage_result(spec, result)

    def test_stage_result_json_roundtrip(self) -> None:
        result = StageResult(
            "validate",
            StageStatus.SUCCESS,
            "python",
            "validate",
            inputs=("model_spec",),
            artifacts=(
                ArtifactRecord(
                    "validated_model",
                    "model_spec",
                    (("schema_version", "1.0"),),
                ),
            ),
            duration_seconds=0.25,
        )
        encoded = json.loads(json.dumps(result.to_data()))
        self.assertEqual(StageResult.from_data(encoded), result)


if __name__ == "__main__":
    unittest.main()
