from __future__ import annotations

import json
import unittest

from tensor_engine import (
    Capability,
    CovariantDerivative,
    Index,
    NoetherWaldResult,
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
    rename_free_indices,
    walk,
)
from tensor_engine.builders import ModelBuilder


class NoetherWaldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.b = ModelBuilder()
        self.up = lambda name: Index(name, Variance.UP, "M")
        self.down = lambda name: Index(name, Variance.DOWN, "M")
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

    def ricci_scalar(self):
        return mul(
            self.b.metric("i", "k"),
            self.b.metric("j", "l"),
            self.b.riemann("i", "j", "k", "l"),
        )

    def kinetic(self):
        return mul(
            Number(-1, 2),
            self.b.metric("i", "j"),
            self.b.scalar_gradient("i"),
            self.b.scalar_gradient("j"),
        )

    def test_inverse_metric_diffeomorphism_is_symmetric(self) -> None:
        result = self.backend.derive_noether_wald(Number(1))
        variation = result.diffeomorphism.inverse_metric
        swapped = rename_free_indices(
            variation,
            {("M", "a"): "b", ("M", "b"): "a"},
        )
        self.assertEqual(variation, self.backend.simplify(swapped))
        self.assertEqual(set(variation.free_indices), {self.up("a"), self.up("b")})

    def test_scalar_diffeomorphism_is_xi_dot_u(self) -> None:
        result = self.backend.derive_noether_wald(Number(1))
        expected = mul(
            Tensor("xi", (self.up("q"),)),
            Tensor("u", (self.down("q"),)),
        )
        self.assertEqual(
            result.diffeomorphism.scalar,
            self.backend.simplify(expected),
        )

    def test_einstein_hilbert_charge_is_komar_potential(self) -> None:
        result = self.backend.derive_noether_wald(self.ricci_scalar())
        c, d = (self.down(name) for name in ("c", "d"))
        derivative = CovariantDerivative(c, Tensor("xi", (d,)))
        expected = add(
            mul(
                -1,
                Tensor("g", (self.up("a"), c.flipped())),
                Tensor("g", (self.up("b"), d.flipped())),
                derivative,
            ),
            mul(
                Tensor("g", (self.up("a"), d.flipped())),
                Tensor("g", (self.up("b"), c.flipped())),
                derivative,
            ),
        )
        self.assertEqual(result.charge_potential, self.backend.simplify(expected))

    def test_charge_potential_is_manifestly_antisymmetric(self) -> None:
        result = self.backend.derive_noether_wald(self.ricci_scalar())
        swapped = rename_free_indices(
            result.charge_potential,
            {("M", "a"): "b", ("M", "b"): "a"},
        )
        self.assertEqual(
            self.backend.simplify(add(result.charge_potential, swapped)),
            Number(0),
        )

    def test_scalar_only_lagrangian_has_no_wald_charge(self) -> None:
        result = self.backend.derive_noether_wald(self.kinetic())
        self.assertEqual(result.charge_potential, Number(0))
        self.assertNotEqual(result.boundary_scalar, Number(0))

    def test_constant_lagrangian_decomposition_passes_off_shell(self) -> None:
        result = self.backend.derive_noether_wald(Number(1))
        self.assertEqual(result.decomposition_residual, Number(0))
        self.assertEqual(
            self.backend.check_noether_decomposition(result).status,
            VerificationStatus.PASSED,
        )
        self.assertEqual(
            self.backend.check_noether_identity(result).status,
            VerificationStatus.PASSED,
        )

    def test_einstein_hilbert_decomposition_preserves_residual(self) -> None:
        result = self.backend.derive_noether_wald(self.ricci_scalar())
        check = self.backend.check_noether_decomposition(result)
        self.assertEqual(check.status, VerificationStatus.UNDETERMINED)
        self.assertIsNotNone(check.residual)

    def test_current_and_identity_have_expected_free_indices(self) -> None:
        result = self.backend.derive_noether_wald(add(self.ricci_scalar(), self.kinetic()))
        self.assertEqual(result.noether_current.free_indices, (self.up("a"),))
        self.assertEqual(result.constraint_current.free_indices, (self.up("a"),))
        self.assertEqual(result.charge_divergence.free_indices, (self.up("a"),))
        self.assertEqual(result.noether_identity.free_indices, (self.down("b"),))

    def test_einstein_hilbert_charge_contains_only_derivatives_of_xi(self) -> None:
        result = self.backend.derive_noether_wald(self.ricci_scalar())
        derivatives = [node for node in walk(result.charge_potential) if isinstance(node, CovariantDerivative)]
        self.assertTrue(derivatives)
        self.assertTrue(
            all(
                isinstance(node.operand, Tensor) and node.operand.name == "xi"
                for node in derivatives
            )
        )

    def test_noether_result_json_roundtrip(self) -> None:
        result = self.backend.derive_noether_wald(add(self.ricci_scalar(), self.kinetic()))
        encoded = json.loads(json.dumps(result.to_data()))
        self.assertEqual(NoetherWaldResult.from_data(encoded), result)

    def test_phase_six_capabilities_are_declared(self) -> None:
        for capability in (
            Capability.DIFFEOMORPHISM_VARIATION,
            Capability.NOETHER_CURRENT,
            Capability.WALD_CHARGE,
            Capability.NOETHER_IDENTITY,
        ):
            self.assertTrue(self.backend.info.supports(capability))


if __name__ == "__main__":
    unittest.main()
