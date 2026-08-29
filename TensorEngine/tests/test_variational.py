from __future__ import annotations

import json
import unittest

from tensor_engine import (
    Capability,
    CovariantDerivative,
    DimensionSpec,
    FunctionDerivative,
    Index,
    LagrangianMomenta,
    ModelBuilder,
    ModelSpec,
    Number,
    ParameterSpec,
    Scalar,
    StructuralTensorBackend,
    Tensor,
    TensorDeclaration,
    TensorSymmetry,
    Variance,
    Variation,
    VariationalContext,
    VolumeElement,
    add,
    expr_from_data,
    function,
    mul,
    rename_free_indices,
    scalar_partial_derivative,
    tensor_partial_derivative,
    walk,
)


class VariationalCalculusTests(unittest.TestCase):
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
        context = VariationalContext(constant_scalars=frozenset({"D", "alpha"}))
        self.backend = StructuralTensorBackend(
            declarations,
            variational_context=context,
        )

    def kinetic_scalar(self):
        return (
            self.b.metric("i", "j")
            * self.b.scalar_gradient("i")
            * self.b.scalar_gradient("j")
        )

    def ricci_scalar(self):
        return (
            self.b.metric("i", "k")
            * self.b.metric("j", "l")
            * self.b.riemann("i", "j", "k", "l")
        )

    def test_variation_json_roundtrip(self) -> None:
        expression = Variation(Tensor("u", (self.down("a"),)))
        encoded = json.loads(json.dumps(expression.to_data()))
        self.assertEqual(expr_from_data(encoded), expression)

    def test_volume_element_json_roundtrip(self) -> None:
        expression = VolumeElement("g")
        encoded = json.loads(json.dumps(expression.to_data()))
        self.assertEqual(expr_from_data(encoded), expression)

    def test_scalar_partial_chain_rule(self) -> None:
        phi = Scalar("phi")
        result = scalar_partial_derivative(function("F", phi), "phi")
        self.assertEqual(result, FunctionDerivative("F", (1,), (phi,)))

    def test_scalar_partial_supports_symbolic_exponent(self) -> None:
        phi = Scalar("phi")
        result = self.backend.canonicalize(scalar_partial_derivative(phi**phi, "phi"))
        expected = self.backend.canonicalize(
            add(
                mul(phi, phi ** add(phi, -1)),
                mul(phi**phi, function("Log", phi)),
            )
        )
        self.assertEqual(result, expected)

    def test_declared_parameter_has_zero_variation(self) -> None:
        self.assertEqual(self.backend.direct_variation(Scalar("alpha")), Number(0))

    def test_kinetic_metric_momentum(self) -> None:
        momenta = self.backend.derive_momenta(mul(Number(-1, 2), self.kinetic_scalar()))
        expected = self.backend.canonicalize(
            mul(
                Number(-1, 2),
                Tensor("u", (self.down("a"),)),
                Tensor("u", (self.down("b"),)),
            )
        )
        self.assertEqual(momenta.metric, expected)

    def test_kinetic_gradient_momentum(self) -> None:
        momenta = self.backend.derive_momenta(mul(Number(-1, 2), self.kinetic_scalar()))
        expected = self.backend.canonicalize(mul(-1, Tensor("u", (self.up("a"),))))
        self.assertEqual(momenta.scalar_gradient, expected)

    def test_potential_scalar_derivative(self) -> None:
        phi = Scalar("phi")
        momenta = self.backend.derive_momenta(mul(-1, function("V", phi)))
        expected = mul(-1, FunctionDerivative("V", (1,), (phi,)))
        self.assertEqual(momenta.scalar, expected)
        self.assertEqual(momenta.metric, Number(0))
        self.assertEqual(momenta.curvature, Number(0))
        self.assertEqual(momenta.scalar_gradient, Number(0))

    def test_ricci_scalar_curvature_momentum(self) -> None:
        momentum = self.backend.derive_momenta(self.ricci_scalar()).curvature
        expected = self.backend.canonicalize(
            mul(
                Number(1, 2),
                add(
                    mul(self.b.metric("a", "c"), self.b.metric("b", "d")),
                    mul(-1, self.b.metric("a", "d"), self.b.metric("b", "c")),
                ),
            )
        )
        self.assertEqual(momentum, expected)

    def test_curvature_momentum_is_antisymmetric_in_first_pair(self) -> None:
        momentum = self.backend.derive_momenta(self.ricci_scalar()).curvature
        swapped = rename_free_indices(momentum, {("M", "a"): "b", ("M", "b"): "a"})
        self.assertEqual(self.backend.canonicalize(add(momentum, swapped)), Number(0))

    def test_curvature_momentum_satisfies_first_bianchi(self) -> None:
        momentum = self.backend.derive_momenta(self.ricci_scalar()).curvature
        second = rename_free_indices(
            momentum,
            {("M", "b"): "c", ("M", "c"): "d", ("M", "d"): "b"},
        )
        third = rename_free_indices(
            momentum,
            {("M", "b"): "d", ("M", "c"): "b", ("M", "d"): "c"},
        )
        self.assertEqual(self.backend.canonicalize(add(momentum, second, third)), Number(0))

    def test_metric_momentum_is_manifestly_symmetric(self) -> None:
        momentum = self.backend.derive_momenta(self.ricci_scalar()).metric
        swapped = rename_free_indices(momentum, {("M", "a"): "b", ("M", "b"): "a"})
        self.assertEqual(self.backend.canonicalize(add(momentum, mul(-1, swapped))), Number(0))

    def test_momenta_json_roundtrip(self) -> None:
        momenta = self.backend.derive_momenta(
            add(self.ricci_scalar(), self.kinetic_scalar(), Scalar("phi"))
        )
        encoded = json.loads(json.dumps(momenta.to_data()))
        self.assertEqual(LagrangianMomenta.from_data(encoded), momenta)

    def test_raw_variation_contains_four_independent_variations(self) -> None:
        lagrangian = add(self.ricci_scalar(), self.kinetic_scalar(), Scalar("phi"))
        raw = self.backend.raw_lagrangian_variation(self.backend.derive_momenta(lagrangian))
        variations = [node for node in walk(raw) if isinstance(node, Variation)]
        varied_names = {
            node.operand.name
            for node in variations
            if isinstance(node.operand, (Scalar, Tensor))
        }
        self.assertEqual(varied_names, {"g", "Riemann", "phi", "u"})
        self.assertTrue(raw.is_scalar)

    def test_direct_and_momentum_variation_agree_for_function_of_phi(self) -> None:
        lagrangian = function("F", Scalar("phi"))
        direct = self.backend.direct_variation(lagrangian)
        reconstructed = self.backend.raw_lagrangian_variation(
            self.backend.derive_momenta(lagrangian)
        )
        self.assertEqual(direct, reconstructed)

    def test_covariant_metric_variation_is_symmetric(self) -> None:
        result = self.backend.covariant_metric_variation(self.down("a"), self.down("b"))
        swapped = rename_free_indices(result, {("M", "a"): "b", ("M", "b"): "a"})
        self.assertEqual(self.backend.canonicalize(result), self.backend.canonicalize(swapped))

    def test_volume_element_variation_has_expected_sign(self) -> None:
        result = self.backend.volume_element_variation()
        a, b = self.down("v0"), self.down("v1")
        expected = self.backend.canonicalize(
            mul(
                Number(-1, 2),
                VolumeElement("g"),
                Tensor("g", (a, b)),
                Variation(Tensor("g", (a.flipped(), b.flipped()))),
            )
        )
        self.assertEqual(result, expected)

    def test_volume_element_metric_partial_derivative(self) -> None:
        result = self.backend.canonicalize(
            tensor_partial_derivative(
                VolumeElement("g"),
                "g",
                (Variance.UP, Variance.UP),
                (self.down("a"), self.down("b")),
                TensorSymmetry.SYMMETRIC,
            )
        )
        expected = self.backend.canonicalize(
            mul(Number(-1, 2), VolumeElement("g"), Tensor("g", (self.down("a"), self.down("b"))))
        )
        self.assertEqual(result, expected)

    def test_scalar_gradient_geometric_variation(self) -> None:
        result = self.backend.scalar_gradient_geometric_variation(self.down("a"))
        expected = CovariantDerivative(self.down("a"), Variation(Scalar("phi")))
        self.assertEqual(result, expected)

    def test_riemann_variation_is_independent_at_this_stage(self) -> None:
        indices = tuple(self.down(name) for name in ("a", "b", "c", "d"))
        result = self.backend.riemann_independent_variation(indices)
        self.assertEqual(result.free_indices, indices)

    def test_variational_context_is_derived_from_model(self) -> None:
        model = ModelSpec(
            "context_model",
            mul(Scalar("alpha"), Scalar("phi")),
            dimension=DimensionSpec("n"),
            parameters=(ParameterSpec("alpha"),),
        )
        context = VariationalContext.from_model(model)
        self.assertEqual(context.constant_scalars, frozenset({"alpha", "n"}))

    def test_variational_capabilities_are_declared(self) -> None:
        self.assertTrue(self.backend.info.supports(Capability.ELEMENTARY_VARIATION))
        self.assertTrue(self.backend.info.supports(Capability.LAGRANGIAN_MOMENTA))
        self.assertTrue(self.backend.info.supports(Capability.RIEMANN_PROJECTION))


if __name__ == "__main__":
    unittest.main()
