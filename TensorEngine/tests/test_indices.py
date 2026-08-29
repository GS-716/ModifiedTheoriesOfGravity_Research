from __future__ import annotations

import unittest

from tensor_engine import (
    IRValidationError,
    Index,
    ModelBuilder,
    Scalar,
    Tensor,
    TensorAlgebraError,
    Variance,
    canonicalize_dummy_indices,
    rename_free_indices,
    substitute,
    tensor_product,
    add,
    mul,
)


class IndexOperationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.b = ModelBuilder()

    def kinetic(self, first: str, second: str):
        return (
            self.b.metric(first, second)
            * self.b.scalar_gradient(first)
            * self.b.scalar_gradient(second)
        )

    def test_alpha_equivalent_dummies_have_same_canonical_form(self) -> None:
        left = canonicalize_dummy_indices(self.kinetic("a", "b"))
        right = canonicalize_dummy_indices(self.kinetic("i", "j"))
        self.assertEqual(left, right)

    def test_dummy_canonicalization_is_idempotent(self) -> None:
        once = canonicalize_dummy_indices(self.kinetic("a", "b"))
        twice = canonicalize_dummy_indices(once)
        self.assertEqual(once, twice)

    def test_tensor_product_avoids_accidental_contraction(self) -> None:
        first = self.b.scalar_gradient("a")
        second = self.b.scalar_gradient("a")
        result = tensor_product(first, second)
        self.assertEqual(len(result.free_indices), 2)
        self.assertNotEqual(result.free_indices[0].name, result.free_indices[1].name)

    def test_free_index_swap_is_simultaneous(self) -> None:
        tensor = Tensor("T", (self.b.down("a"), self.b.down("b")))
        swapped = rename_free_indices(tensor, {("M", "a"): "b", ("M", "b"): "a"})
        self.assertEqual(tuple(index.name for index in swapped.indices), ("b", "a"))

    def test_dummy_inside_add_does_not_collide_with_external_contraction(self) -> None:
        internal = self.kinetic("i", "j")
        vector_sum = add(
            mul(internal, Tensor("A", (self.b.up("a"),))),
            Tensor("B", (self.b.up("a"),)),
        )
        expression = mul(vector_sum, Tensor("C", (self.b.down("a"),)))
        canonical = canonicalize_dummy_indices(expression)
        self.assertTrue(canonical.is_scalar)
        self.assertEqual(canonicalize_dummy_indices(canonical), canonical)

    def test_substitution_preserves_free_signature(self) -> None:
        source = Tensor("A", (self.b.down("a"),))
        target = Tensor("B", (self.b.down("a"),))
        self.assertEqual(substitute(source, {source: target}), target)

    def test_substitution_rejects_different_free_signature(self) -> None:
        source = Tensor("A", (self.b.down("a"),))
        with self.assertRaises(TensorAlgebraError):
            substitute(source, {source: Scalar("x")})


if __name__ == "__main__":
    unittest.main()
