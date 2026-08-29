from __future__ import annotations

import json
import unittest

from tensor_engine import (
    Capability,
    CovariantDerivative,
    DifferentialContext,
    FunctionDerivative,
    Index,
    Number,
    Scalar,
    StructuralTensorBackend,
    Tensor,
    TensorDeclaration,
    TensorSymmetry,
    Variance,
    VerificationStatus,
    all_indices,
    expr_from_data,
    function,
    mul,
)


class DifferentialGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.down = lambda name: Index(name, Variance.DOWN, "M")
        self.up = lambda name: Index(name, Variance.UP, "M")
        declarations = (
            TensorDeclaration("g", (Variance.UP, Variance.UP), TensorSymmetry.SYMMETRIC),
            TensorDeclaration(
                "Riemann",
                (Variance.DOWN,) * 4,
                TensorSymmetry.RIEMANN,
            ),
        )
        context = DifferentialContext(constant_scalars=frozenset({"D", "alpha"}))
        self.backend = StructuralTensorBackend(declarations, differential_context=context)

    def test_gradient_of_phi_is_u(self) -> None:
        self.assertEqual(
            self.backend.gradient(Scalar("phi"), self.down("a")),
            Tensor("u", (self.down("a"),)),
        )

    def test_declared_parameter_is_covariantly_constant(self) -> None:
        self.assertEqual(
            self.backend.covariant_derivative(Scalar("alpha"), self.down("a")),
            Number(0),
        )

    def test_metric_compatibility(self) -> None:
        metric = Tensor("g", (self.up("a"), self.up("b")))
        self.assertEqual(
            self.backend.covariant_derivative(metric, self.down("c")),
            Number(0),
        )

    def test_leibniz_rule_for_phi_squared(self) -> None:
        phi = Scalar("phi")
        result = self.backend.covariant_derivative(mul(phi, phi), self.down("a"))
        expected = self.backend.canonicalize(
            mul(2, phi, Tensor("u", (self.down("a"),)))
        )
        self.assertEqual(result, expected)

    def test_function_chain_rule(self) -> None:
        phi = Scalar("phi")
        result = self.backend.gradient(function("F", phi), self.down("a"))
        expected = self.backend.canonicalize(
            mul(
                FunctionDerivative("F", (1,), (phi,)),
                Tensor("u", (self.down("a"),)),
            )
        )
        self.assertEqual(result, expected)

    def test_function_derivative_json_roundtrip(self) -> None:
        expression = FunctionDerivative("F", (2,), (Scalar("phi"),))
        encoded = json.loads(json.dumps(expression.to_data()))
        self.assertEqual(expr_from_data(encoded), expression)

    def test_scalar_hessian_is_symmetric(self) -> None:
        phi = Scalar("phi")
        first = self.backend.hessian(phi, self.down("a"), self.down("b"))
        second = self.backend.hessian(phi, self.down("b"), self.down("a"))
        self.assertEqual(first, second)

    def test_divergence_of_vector_is_scalar(self) -> None:
        vector = Tensor("V", (self.up("a"),))
        result = self.backend.divergence(vector, self.up("a"))
        self.assertTrue(result.is_scalar)

    def test_divergence_of_covector_is_scalar(self) -> None:
        covector = Tensor("W", (self.down("a"),))
        result = self.backend.divergence(covector, self.down("a"))
        self.assertTrue(result.is_scalar)

    def test_scalar_laplacian_is_scalar(self) -> None:
        self.assertTrue(self.backend.laplacian(Scalar("phi")).is_scalar)

    def test_scalar_commutator_vanishes(self) -> None:
        check = self.backend.check_commutator(
            Scalar("phi"),
            self.down("a"),
            self.down("b"),
        )
        self.assertEqual(check.status, VerificationStatus.PASSED)

    def test_vector_commutator_preserves_residual_when_undecidable(self) -> None:
        vector = Tensor("V", (self.up("c"),))
        check = self.backend.check_commutator(
            vector,
            self.down("a"),
            self.down("b"),
        )
        self.assertEqual(check.status, VerificationStatus.UNDETERMINED)
        self.assertIsNotNone(check.residual)

    def test_lie_derivative_of_scalar_is_scalar(self) -> None:
        result = self.backend.lie_derivative(Scalar("phi"))
        self.assertTrue(result.is_scalar)

    def test_lie_derivative_of_phi_is_xi_dot_u(self) -> None:
        result = self.backend.lie_derivative(Scalar("phi"))
        expected = self.backend.canonicalize(
            mul(
                Tensor("xi", (self.up("q"),)),
                Tensor("u", (self.down("q"),)),
            )
        )
        self.assertEqual(result, expected)

    def test_lie_derivative_preserves_vector_rank(self) -> None:
        vector = Tensor("V", (self.up("a"),))
        result = self.backend.lie_derivative(vector)
        self.assertEqual(result.free_indices, (self.up("a"),))

    def test_lie_derivative_respects_nondefault_index_space(self) -> None:
        a = Index("a", Variance.UP, "N")
        result = self.backend.lie_derivative(Tensor("V", (a,)))
        self.assertEqual(result.free_indices, (a,))
        self.assertTrue(all(index.space == "N" for index in all_indices(result)))

    def test_differential_bianchi_is_undetermined_structurally(self) -> None:
        check = self.backend.check_differential_bianchi(
            "Riemann",
            tuple(self.down(name) for name in ("e", "a", "b", "c", "d")),
        )
        self.assertEqual(check.status, VerificationStatus.UNDETERMINED)
        self.assertIsNotNone(check.residual)

    def test_differential_capabilities_are_declared(self) -> None:
        self.assertTrue(self.backend.info.supports(Capability.COVARIANT_DERIVATIVES))
        self.assertTrue(self.backend.info.supports(Capability.FUNCTION_CHAIN_RULE))
        self.assertTrue(self.backend.info.supports(Capability.LIE_DERIVATIVE))
        self.assertFalse(self.backend.info.supports(Capability.DIFFERENTIAL_BIANCHI))


if __name__ == "__main__":
    unittest.main()
