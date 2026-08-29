from __future__ import annotations

import unittest

from tensor_engine import (
    BackendCapabilityError,
    Capability,
    Index,
    Number,
    Scalar,
    StructuralTensorBackend,
    Tensor,
    TensorDeclaration,
    TensorSymmetry,
    Variance,
    VerificationStatus,
    add,
    mul,
)


class TensorCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.down = lambda name: Index(name, Variance.DOWN, "M")
        self.up = lambda name: Index(name, Variance.UP, "M")
        self.declarations = (
            TensorDeclaration("g", (Variance.UP, Variance.UP), TensorSymmetry.SYMMETRIC),
            TensorDeclaration(
                "Riemann",
                (Variance.DOWN,) * 4,
                TensorSymmetry.RIEMANN,
            ),
        )
        self.backend = StructuralTensorBackend(self.declarations)

    def riemann(self, a: str, b: str, c: str, d: str) -> Tensor:
        return Tensor("Riemann", tuple(self.down(name) for name in (a, b, c, d)))

    def test_symmetric_metric_is_canonicalized(self) -> None:
        metric_ba = Tensor("g", (self.up("b"), self.up("a")))
        metric_ab = Tensor("g", (self.up("a"), self.up("b")))
        self.assertEqual(self.backend.canonicalize(metric_ba), metric_ab)

    def test_full_canonicalization_is_idempotent(self) -> None:
        metric = Tensor("g", (self.up("a"), self.up("b")))
        vector_a = Tensor("u", (self.down("a"),))
        vector_b = Tensor("u", (self.down("b"),))
        once = self.backend.canonicalize(mul(metric, vector_a, vector_b))
        self.assertEqual(self.backend.canonicalize(once), once)

    def test_combined_like_terms_are_idempotent(self) -> None:
        vector = Tensor("u", (self.down("a"),))
        expression = add(mul(Scalar("phi"), vector), mul(vector, Scalar("phi")))
        once = self.backend.canonicalize(expression)
        self.assertEqual(self.backend.canonicalize(once), once)

    def test_riemann_antisymmetry_in_first_pair(self) -> None:
        residual = add(self.riemann("b", "a", "c", "d"), self.riemann("a", "b", "c", "d"))
        self.assertEqual(self.backend.canonicalize(residual), Number(0))

    def test_riemann_pair_exchange(self) -> None:
        left = self.backend.canonicalize(self.riemann("c", "d", "a", "b"))
        right = self.backend.canonicalize(self.riemann("a", "b", "c", "d"))
        self.assertEqual(left, right)

    def test_antisymmetrized_metric_is_zero(self) -> None:
        metric = Tensor("g", (self.up("a"), self.up("b")))
        result = self.backend.antisymmetrize(metric, (self.up("a"), self.up("b")))
        self.assertEqual(result, Number(0))

    def test_metric_raises_vector_index(self) -> None:
        metric = Tensor("g", (self.up("a"), self.up("b")))
        vector = Tensor("u", (self.down("b"),))
        expected = Tensor("u", (self.up("a"),))
        self.assertEqual(self.backend.simplify(mul(metric, vector)), expected)

    def test_metric_trace_is_dimension(self) -> None:
        metric_up = Tensor("g", (self.up("a"), self.up("b")))
        metric_down = Tensor("g", (self.down("a"), self.down("b")))
        self.assertEqual(self.backend.simplify(mul(metric_up, metric_down)), Scalar("D"))

    def test_delta_contracts_vector(self) -> None:
        delta = Tensor("delta", (self.up("a"), self.down("b")))
        vector = Tensor("v", (self.up("b"),))
        self.assertEqual(
            self.backend.simplify(mul(delta, vector)),
            Tensor("v", (self.up("a"),)),
        )

    def test_delta_trace_is_dimension(self) -> None:
        delta_trace = Tensor("delta", (self.up("a"), self.down("a")))
        self.assertEqual(self.backend.simplify(delta_trace), Scalar("D"))

    def test_raise_index_introduces_explicit_metric(self) -> None:
        vector = Tensor("u", (self.down("a"),))
        raised = self.backend.raise_index(vector, self.down("a"))
        self.assertEqual(self.backend.simplify(raised), Tensor("u", (self.up("a"),)))

    def test_lower_index_introduces_explicit_metric(self) -> None:
        vector = Tensor("v", (self.up("a"),))
        lowered = self.backend.lower_index(vector, self.up("a"))
        self.assertEqual(self.backend.simplify(lowered), Tensor("v", (self.down("a"),)))

    def test_expand_distributes_scalar_product(self) -> None:
        x, y, z = Scalar("x"), Scalar("y"), Scalar("z")
        result = self.backend.canonicalize(mul(x, add(y, z)))
        expected = self.backend.canonicalize(add(mul(x, y), mul(x, z)))
        self.assertEqual(result, expected)

    def test_bianchi_is_reported_as_undetermined(self) -> None:
        check = self.backend.check_first_bianchi(
            "Riemann",
            (self.down("a"), self.down("b"), self.down("c"), self.down("d")),
        )
        self.assertEqual(check.status, VerificationStatus.UNDETERMINED)
        self.assertIsNotNone(check.residual)

    def test_unsupported_capability_raises(self) -> None:
        with self.assertRaises(BackendCapabilityError):
            self.backend.require(Capability.MULTITERM_BIANCHI)


if __name__ == "__main__":
    unittest.main()
