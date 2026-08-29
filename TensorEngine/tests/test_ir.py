from __future__ import annotations

import json
import unittest

from tensor_engine import (
    IRValidationError,
    ModelBuilder,
    Number,
    Scalar,
    expr_from_data,
    function,
)


class IRTests(unittest.TestCase):
    def setUp(self) -> None:
        self.b = ModelBuilder()

    def test_ricci_scalar_is_fully_contracted(self) -> None:
        ricci_scalar = (
            self.b.metric("a", "c")
            * self.b.metric("b", "d")
            * self.b.riemann("a", "b", "c", "d")
        )
        self.assertTrue(ricci_scalar.is_scalar)

    def test_kinetic_scalar_is_fully_contracted(self) -> None:
        kinetic = (
            self.b.metric("a", "b")
            * self.b.scalar_gradient("a")
            * self.b.scalar_gradient("b")
        )
        self.assertTrue(kinetic.is_scalar)

    def test_scalar_subexpressions_keep_independent_dummy_scope(self) -> None:
        kinetic = (
            self.b.metric("a", "b")
            * self.b.scalar_gradient("a")
            * self.b.scalar_gradient("b")
        )
        self.assertTrue((kinetic * kinetic).is_scalar)

    def test_rejects_same_variance_contraction(self) -> None:
        with self.assertRaises(IRValidationError):
            _ = self.b.scalar_gradient("a") * self.b.scalar_gradient("a")

    def test_rejects_sum_with_different_free_indices(self) -> None:
        with self.assertRaises(IRValidationError):
            _ = self.b.scalar_gradient("a") + self.b.scalar_gradient("b")

    def test_json_roundtrip_preserves_expression(self) -> None:
        expression = function("F", self.b.phi) * Scalar("alpha") + Number(3, 2)
        encoded = json.loads(json.dumps(expression.to_data()))
        self.assertEqual(expr_from_data(encoded), expression)


if __name__ == "__main__":
    unittest.main()
