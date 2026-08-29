from __future__ import annotations

import json
import unittest

from tensor_engine import (
    CovariantDerivative,
    DimensionSpec,
    FunctionSpec,
    ModelBuilder,
    ModelSpec,
    ModelValidationError,
    Number,
    ParameterSpec,
    Scalar,
    Variation,
    VolumeElement,
    function,
)


def scalar_tensor_model() -> ModelSpec:
    b = ModelBuilder()
    ricci_scalar = (
        b.metric("a", "c")
        * b.metric("b", "d")
        * b.riemann("a", "b", "c", "d")
    )
    kinetic = b.metric("a", "b") * b.scalar_gradient("a") * b.scalar_gradient("b")
    lagrangian = (
        function("F", b.phi) * ricci_scalar
        - Number(1, 2) * function("Z", b.phi) * kinetic
        - function("V", b.phi)
    )
    return ModelSpec(
        name="scalar_tensor_reference",
        lagrangian=lagrangian,
        dimension=DimensionSpec("D"),
        normalization=Scalar("kappa"),
        parameters=(ParameterSpec("kappa", ("nonzero",)),),
        functions=(FunctionSpec("F"), FunctionSpec("Z"), FunctionSpec("V")),
        assumptions=("D>=2",),
        metadata=(("purpose", "phase1_reference"),),
    )


class ModelSpecTests(unittest.TestCase):
    def test_reference_model_is_valid(self) -> None:
        model = scalar_tensor_model()
        self.assertTrue(model.lagrangian.is_scalar)
        self.assertEqual(model.dimension.value, "D")

    def test_model_json_roundtrip(self) -> None:
        model = scalar_tensor_model()
        encoded = json.loads(json.dumps(model.to_data()))
        self.assertEqual(ModelSpec.from_data(encoded), model)

    def test_rejects_free_index_in_lagrangian(self) -> None:
        b = ModelBuilder()
        with self.assertRaises(ModelValidationError):
            ModelSpec("invalid_free_index", b.scalar_gradient("a"))

    def test_rejects_undeclared_scalar(self) -> None:
        with self.assertRaises(ModelValidationError):
            ModelSpec("invalid_symbol", Scalar("alpha"))

    def test_rejects_explicit_covariant_derivative(self) -> None:
        b = ModelBuilder()
        expression = CovariantDerivative(b.down("a"), b.phi)
        with self.assertRaises(ModelValidationError):
            ModelSpec("invalid_derivative", expression)

    def test_rejects_calculated_variation_node(self) -> None:
        with self.assertRaises(ModelValidationError):
            ModelSpec("invalid_variation", Variation(Scalar("phi")))

    def test_rejects_volume_element_inside_lagrangian(self) -> None:
        with self.assertRaises(ModelValidationError):
            ModelSpec("invalid_measure", VolumeElement("g"))


if __name__ == "__main__":
    unittest.main()
