from __future__ import annotations

import json
import unittest

from tensor_engine import (
    Capability,
    CovariantDerivative,
    EulerLagrangeResult,
    FunctionDerivative,
    Index,
    Number,
    Scalar,
    StructuralTensorBackend,
    Tensor,
    TensorDeclaration,
    TensorSymmetry,
    Variance,
    Variation,
    VerificationStatus,
    VolumeElement,
    add,
    curvature_derivative_metric_term,
    function,
    mul,
    rename_free_indices,
    walk,
)
from tensor_engine.builders import ModelBuilder


class EulerLagrangeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.b = ModelBuilder()
        self.down = lambda name: Index(name, Variance.DOWN, "M")
        self.up = lambda name: Index(name, Variance.UP, "M")
        declarations = (
            TensorDeclaration("g", (Variance.UP, Variance.UP), TensorSymmetry.SYMMETRIC),
            TensorDeclaration(
                "Riemann",
                (Variance.DOWN,) * 4,
                TensorSymmetry.RIEMANN,
            ),
            TensorDeclaration("u", (Variance.DOWN,), TensorSymmetry.NONE),
        )
        self.backend = StructuralTensorBackend(declarations)

    def kinetic(self):
        return mul(
            Number(-1, 2),
            self.b.metric("i", "j"),
            self.b.scalar_gradient("i"),
            self.b.scalar_gradient("j"),
        )

    def ricci_scalar(self):
        return mul(
            self.b.metric("i", "k"),
            self.b.metric("j", "l"),
            self.b.riemann("i", "j", "k", "l"),
        )

    def test_connection_variation_is_symmetric_in_lower_indices(self) -> None:
        first = self.backend.connection_variation(
            self.up("a"), self.down("b"), self.down("c")
        )
        second = self.backend.connection_variation(
            self.up("a"), self.down("c"), self.down("b")
        )
        self.assertEqual(first, second)

    def test_connection_variation_has_correct_free_signature(self) -> None:
        result = self.backend.connection_variation(
            self.up("a"), self.down("b"), self.down("c")
        )
        self.assertEqual(
            set(result.free_indices),
            {self.up("a"), self.down("b"), self.down("c")},
        )

    def test_palatini_variation_is_antisymmetric_in_derivative_indices(self) -> None:
        first = self.backend.mixed_curvature_variation(
            self.up("a"), self.down("b"), self.down("c"), self.down("d")
        )
        second = self.backend.mixed_curvature_variation(
            self.up("a"), self.down("b"), self.down("d"), self.down("c")
        )
        self.assertEqual(self.backend.canonicalize(add(first, second)), Number(0))

    def test_all_down_curvature_variation_preserves_rank(self) -> None:
        indices = tuple(self.down(name) for name in ("a", "b", "c", "d"))
        result = self.backend.all_down_curvature_variation(indices)
        self.assertEqual(set(result.free_indices), set(indices))
        tensor_names = {node.name for node in walk(result) if isinstance(node, Tensor)}
        self.assertIn("delta_Gamma", tensor_names)
        self.assertIn("Riemann", tensor_names)

    def test_expanded_palatini_eliminates_connection_symbol(self) -> None:
        result = self.backend.mixed_curvature_variation(
            self.up("a"),
            self.down("b"),
            self.down("c"),
            self.down("d"),
            expand_connection=True,
        )
        tensor_names = {node.name for node in walk(result) if isinstance(node, Tensor)}
        self.assertNotIn("delta_Gamma", tensor_names)

    def test_cosmological_term_metric_euler(self) -> None:
        lagrangian = mul(-2, Scalar("alpha"))
        result = self.backend.derive_euler_lagrange(lagrangian)
        expected = Tensor("g", (self.down("a"), self.down("b"))) * Scalar("alpha")
        self.assertEqual(result.metric_euler, self.backend.canonicalize(expected))
        self.assertEqual(result.scalar_euler, Number(0))

    def test_kinetic_metric_euler(self) -> None:
        lagrangian = self.kinetic()
        result = self.backend.derive_euler_lagrange(lagrangian)
        expected = self.backend.canonicalize(
            add(
                mul(
                    Number(-1, 2),
                    Tensor("u", (self.down("a"),)),
                    Tensor("u", (self.down("b"),)),
                ),
                mul(
                    Number(-1, 2),
                    Tensor("g", (self.down("a"), self.down("b"))),
                    lagrangian,
                ),
            )
        )
        self.assertEqual(result.metric_euler, self.backend.simplify(expected))

    def test_kinetic_scalar_euler_is_divergence_of_u(self) -> None:
        result = self.backend.derive_euler_lagrange(self.kinetic())
        expected = CovariantDerivative(
            self.down("a"), Tensor("u", (self.up("a"),))
        )
        self.assertEqual(result.scalar_euler, self.backend.canonicalize(expected))

    def test_scalar_boundary_for_canonical_kinetic_term(self) -> None:
        result = self.backend.derive_euler_lagrange(self.kinetic())
        expected = mul(
            -1,
            Tensor("u", (self.up("a"),)),
            Variation(Scalar("phi")),
        )
        self.assertEqual(result.boundary_scalar, self.backend.canonicalize(expected))

    def test_scalar_integration_by_parts_passes(self) -> None:
        momenta = self.backend.derive_momenta(self.kinetic())
        check = self.backend.check_scalar_integration_by_parts(momenta)
        self.assertEqual(check.status, VerificationStatus.PASSED)

    def test_einstein_hilbert_curvature_derivative_term_vanishes(self) -> None:
        momenta = self.backend.derive_momenta(self.ricci_scalar())
        term = curvature_derivative_metric_term(
            momenta.curvature,
            self.backend.variational_context,
            self.backend.differential_context,
        )
        self.assertEqual(term, Number(0))

    def test_einstein_hilbert_metric_euler_is_einstein_tensor_form(self) -> None:
        lagrangian = self.ricci_scalar()
        result = self.backend.derive_euler_lagrange(lagrangian)
        ricci_ab = Tensor(
            "Riemann",
            (self.up("r"), self.down("a"), self.down("r"), self.down("b")),
        )
        ricci_ba = rename_free_indices(
            ricci_ab,
            {("M", "a"): "b", ("M", "b"): "a"},
        )
        expected = self.backend.simplify(
            add(
                mul(Number(1, 2), add(ricci_ab, ricci_ba)),
                mul(
                    Number(-1, 2),
                    Tensor("g", (self.down("a"), self.down("b"))),
                    lagrangian,
                ),
            )
        )
        self.assertEqual(result.metric_euler, expected)

    def test_einstein_hilbert_has_metric_boundary_vector(self) -> None:
        result = self.backend.derive_euler_lagrange(self.ricci_scalar())
        self.assertEqual(result.boundary_metric.free_indices, (self.up("a"),))
        self.assertTrue(any(isinstance(node, Variation) for node in walk(result.boundary_metric)))

    def test_potential_contributes_minus_derivative_to_scalar_euler(self) -> None:
        lagrangian = mul(-1, function("V", Scalar("phi")))
        result = self.backend.derive_euler_lagrange(lagrangian)
        expected = mul(
            -1,
            FunctionDerivative("V", (1,), (Scalar("phi"),)),
        )
        self.assertEqual(result.scalar_euler, expected)

    def test_metric_euler_is_manifestly_symmetric(self) -> None:
        lagrangian = add(self.ricci_scalar(), self.kinetic())
        result = self.backend.derive_euler_lagrange(lagrangian)
        swapped = rename_free_indices(
            result.metric_euler,
            {("M", "a"): "b", ("M", "b"): "a"},
        )
        self.assertEqual(result.metric_euler, self.backend.canonicalize(swapped))

    def test_full_variation_and_density_are_scalars(self) -> None:
        result = self.backend.derive_euler_lagrange(self.kinetic())
        self.assertTrue(result.full_variation.is_scalar)
        self.assertTrue(result.density_variation.is_scalar)
        self.assertTrue(any(isinstance(node, VolumeElement) for node in walk(result.density_variation)))

    def test_euler_result_json_roundtrip(self) -> None:
        result = self.backend.derive_euler_lagrange(self.kinetic())
        encoded = json.loads(json.dumps(result.to_data()))
        self.assertEqual(EulerLagrangeResult.from_data(encoded), result)

    def test_phase_five_capabilities_are_declared(self) -> None:
        for capability in (
            Capability.PALATINI_VARIATION,
            Capability.INTEGRATION_BY_PARTS,
            Capability.EULER_LAGRANGE,
            Capability.BOUNDARY_POTENTIAL,
        ):
            self.assertTrue(self.backend.info.supports(capability))


if __name__ == "__main__":
    unittest.main()
