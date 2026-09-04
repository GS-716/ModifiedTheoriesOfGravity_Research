from __future__ import annotations

import json
import unittest

import sympy as sp

from tensor_engine import (
    CoordinateGeometry,
    ComponentEvaluation,
    ComponentFieldEquations,
    CovariantDerivative,
    DimensionSpec,
    Function,
    FunctionDerivative,
    GeometryAnsatz,
    Index,
    ModelBuilder,
    ModelSpec,
    ModelValidationError,
    Number,
    Scalar,
    SympyComponentBackend,
    Tensor,
    Variance,
    evaluate_field_equations,
    draft4_angular_scalar_profile,
    draft4_circular_ansatz,
    ir_scalar_to_sympy,
    mul,
    spatially_flat_flrw_ansatz,
    sympy_scalar_to_ir,
)


class AnsatzContractTests(unittest.TestCase):
    def test_flrw_ansatz_json_roundtrip(self) -> None:
        ansatz = spatially_flat_flrw_ansatz()
        encoded = json.loads(json.dumps(ansatz.to_data()))
        self.assertEqual(GeometryAnsatz.from_data(encoded), ansatz)

    def test_component_stage_requires_concrete_matching_dimension(self) -> None:
        ansatz = spatially_flat_flrw_ansatz()
        symbolic = ModelSpec("symbolic", Number(1))
        with self.assertRaises(ModelValidationError):
            ansatz.validate_for_model(symbolic)
        concrete = ModelSpec("concrete", Number(1), dimension=DimensionSpec(3))
        with self.assertRaises(ModelValidationError):
            ansatz.validate_for_model(concrete)

    def test_rejects_nonsymmetric_metric(self) -> None:
        data = spatially_flat_flrw_ansatz().to_data()
        data["metric_covariant"][0][1] = Number(1).to_data()
        with self.assertRaises(ModelValidationError):
            GeometryAnsatz.from_data(data)

    def test_draft4_ansatz_json_roundtrip(self) -> None:
        ansatz = draft4_circular_ansatz()
        encoded = json.loads(json.dumps(ansatz.to_data()))
        self.assertEqual(GeometryAnsatz.from_data(encoded), ansatz)


class Draft4GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = CoordinateGeometry.build(draft4_circular_ansatz())
        cls.tau = cls.geometry.coordinates[0]
        cls.r = cls.geometry.coordinates[1]
        cls.varphi = cls.geometry.coordinates[2]
        cls.f = sp.Function("f")(cls.r)
        cls.Phi = sp.Function("Phi")(cls.r, cls.varphi)

    def test_metric_inverse_and_determinant(self) -> None:
        self.assertEqual(
            sp.simplify(
                self.geometry.metric_covariant * self.geometry.metric_contravariant
            ),
            sp.eye(3),
        )
        self.assertEqual(sp.simplify(self.geometry.determinant + self.r**2), 0)

    def test_default_scalar_profile_is_generic(self) -> None:
        gradient = self.geometry.scalar_gradient_covariant()
        self.assertEqual(gradient, (sp.S.Zero, sp.diff(self.Phi, self.r), sp.diff(self.Phi, self.varphi)))
        self.assertFalse(self.Phi.has(self.tau))
        self.assertNotIn(sp.Symbol("p"), self.geometry.scalar_field.free_symbols)

    def test_time_dependent_scalar_specialization_is_rejected(self) -> None:
        base = draft4_circular_ansatz()
        tau, radial, angle = base.chart.coordinates
        with self.assertRaisesRegex(ModelValidationError, "tau"):
            base.specialize_scalar(Function("Psi", (tau, radial, angle)))

    def test_axial_scalar_profile_is_an_explicit_later_specialization(self) -> None:
        base = draft4_circular_ansatz()
        specialized = base.specialize_scalar(
            draft4_angular_scalar_profile(),
            assumptions=("phi=p*varphi",),
        )
        geometry = CoordinateGeometry.build(specialized)
        p = sp.Symbol("p")
        self.assertEqual(
            geometry.scalar_gradient_covariant(),
            (sp.S.Zero, sp.S.Zero, p),
        )
        self.assertEqual(sp.simplify(geometry.scalar_laplacian()), 0)
        self.assertNotEqual(base.scalar_field, specialized.scalar_field)

    def test_metric_and_scalar_solutions_can_be_specialized_together(self) -> None:
        base = draft4_circular_ansatz()
        _, radial, angle = base.chart.coordinates
        f = Function("f", (radial,))
        Phi = Function("Phi", (radial, angle))
        specialized = base.specialize(
            {
                f: Number(1),
                Phi: mul(Scalar("p"), angle),
            },
            assumptions=("phi=p*varphi",),
        )
        self.assertEqual(specialized.metric_covariant[0][0], Number(-1))
        self.assertEqual(
            sp.simplify(ir_scalar_to_sympy(specialized.metric_covariant[1][1])),
            sp.S.One,
        )
        self.assertEqual(specialized.scalar_field, mul(Scalar("p"), angle))
        self.assertEqual(GeometryAnsatz.from_data(specialized.to_data()), specialized)

    def test_kinetic_scalar_only_contains_p_after_explicit_specialization(self) -> None:
        base = draft4_circular_ansatz()
        specialized = base.specialize_scalar(draft4_angular_scalar_profile())
        X = ModelBuilder().kinetic_scalar()
        generic_x = SympyComponentBackend(base).evaluate_sympy(X).scalar
        specialized_x = SympyComponentBackend(specialized).evaluate_sympy(X).scalar
        p = sp.Symbol("p")
        self.assertNotIn(p, generic_x.free_symbols)
        self.assertEqual(
            sp.simplify(
                generic_x
                - self.f * sp.diff(self.Phi, self.r) ** 2
                - sp.diff(self.Phi, self.varphi) ** 2 / self.r**2
            ),
            0,
        )
        self.assertFalse(generic_x.has(self.tau))
        self.assertEqual(sp.simplify(specialized_x - p**2 / self.r**2), 0)

    def test_generic_multivariate_projection_budget_is_structural(self) -> None:
        base = draft4_circular_ansatz()
        expression = Scalar("phi")
        for position in range(25):
            expression = CovariantDerivative(
                Index(f"q{position}", Variance.DOWN),
                expression,
            )
        reason = SympyComponentBackend(base).projection_limit_reason(expression)
        self.assertIsNotNone(reason)
        self.assertIn("25 nodos", reason)
        specialized = base.specialize_scalar(draft4_angular_scalar_profile())
        self.assertIsNone(
            SympyComponentBackend(specialized).projection_limit_reason(expression)
        )


class ScalarTranslationTests(unittest.TestCase):
    def test_function_derivative_roundtrip(self) -> None:
        t = Scalar("t")
        expression = FunctionDerivative("F", (2,), (Function("phi", (t,)),))
        translated = ir_scalar_to_sympy(expression)
        self.assertEqual(sympy_scalar_to_ir(translated), expression)

    def test_substituted_function_derivative_accepts_composite_argument(self) -> None:
        dummy = sp.Symbol("dummy")
        p, angle = sp.symbols("p varphi")
        expression = sp.Subs(
            sp.Derivative(sp.Function("V")(dummy), dummy),
            dummy,
            p * angle,
        )
        self.assertEqual(
            sympy_scalar_to_ir(expression),
            FunctionDerivative(
                "V",
                (1,),
                (mul(Scalar("p"), Scalar("varphi")),),
            ),
        )

    def test_elementary_functions_are_evaluated(self) -> None:
        theta = Scalar("theta")
        expression = Function("sin", (theta,))
        translated = ir_scalar_to_sympy(expression)
        self.assertEqual(sp.diff(translated, sp.Symbol("theta")), sp.cos(sp.Symbol("theta")))


class CoordinateGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = CoordinateGeometry.build(spatially_flat_flrw_ansatz())
        cls.t = cls.geometry.coordinates[0]
        cls.a = sp.Function("a")(cls.t)
        cls.phi = sp.Function("phi")(cls.t)

    def test_inverse_and_determinant(self) -> None:
        identity = sp.simplify(self.geometry.metric_covariant * self.geometry.metric_contravariant)
        self.assertEqual(identity, sp.eye(4))
        self.assertEqual(sp.simplify(self.geometry.determinant + self.a**6), 0)

    def test_flrw_christoffel(self) -> None:
        gamma = self.geometry.nonzero_christoffel()
        expected = {
            (0, 1, 1): self.a * sp.diff(self.a, self.t),
            (0, 2, 2): self.a * sp.diff(self.a, self.t),
            (0, 3, 3): self.a * sp.diff(self.a, self.t),
            (1, 0, 1): sp.diff(self.a, self.t) / self.a,
            (1, 1, 0): sp.diff(self.a, self.t) / self.a,
            (2, 0, 2): sp.diff(self.a, self.t) / self.a,
            (2, 2, 0): sp.diff(self.a, self.t) / self.a,
            (3, 0, 3): sp.diff(self.a, self.t) / self.a,
            (3, 3, 0): sp.diff(self.a, self.t) / self.a,
        }
        self.assertEqual(gamma, expected)

    def test_flrw_ricci_and_einstein(self) -> None:
        adot = sp.diff(self.a, self.t)
        addot = sp.diff(self.a, (self.t, 2))
        self.assertEqual(sp.simplify(self.geometry.ricci_covariant[0, 0] + 3 * addot / self.a), 0)
        self.assertEqual(
            sp.simplify(self.geometry.ricci_covariant[1, 1] - self.a * addot - 2 * adot**2),
            0,
        )
        expected_scalar = 6 * (self.a * addot + adot**2) / self.a**2
        self.assertEqual(sp.simplify(self.geometry.ricci_scalar - expected_scalar), 0)
        self.assertEqual(
            sp.simplify(self.geometry.einstein_covariant[0, 0] - 3 * adot**2 / self.a**2),
            0,
        )

    def test_homogeneous_scalar_laplacian(self) -> None:
        expected = -sp.diff(self.phi, (self.t, 2)) - (
            3 * sp.diff(self.a, self.t) * sp.diff(self.phi, self.t) / self.a
        )
        self.assertEqual(sp.simplify(self.geometry.scalar_laplacian() - expected), 0)


class AbstractProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = SympyComponentBackend(spatially_flat_flrw_ansatz())
        cls.up = staticmethod(lambda name: Index(name, Variance.UP))
        cls.down = staticmethod(lambda name: Index(name, Variance.DOWN))

    def ricci_covariant(self):
        return mul(
            Tensor("g", (self.up("c"), self.up("d"))),
            Tensor(
                "Riemann",
                (self.down("c"), self.down("a"), self.down("d"), self.down("b")),
            ),
        )

    def ricci_scalar(self):
        return mul(
            Tensor("g", (self.up("a"), self.up("b"))),
            self.ricci_covariant(),
        )

    def test_abstract_ricci_scalar_projects_to_coordinate_result(self) -> None:
        projected = self.backend.evaluate_sympy(self.ricci_scalar()).scalar
        self.assertEqual(sp.simplify(projected - self.backend.geometry.ricci_scalar), 0)

    def test_abstract_scalar_laplacian_projects_to_coordinate_result(self) -> None:
        a, b = self.down("a"), self.down("b")
        expression = mul(
            Tensor("g", (a.flipped(), b.flipped())),
            CovariantDerivative(a, CovariantDerivative(b, Scalar("phi"))),
        )
        projected = self.backend.evaluate_sympy(expression).scalar
        self.assertEqual(sp.simplify(projected - self.backend.geometry.scalar_laplacian()), 0)

    def test_einstein_equation_projection_and_independent_selection(self) -> None:
        a, b = self.down("a"), self.down("b")
        metric_euler = self.ricci_covariant() - mul(
            Number(1, 2),
            Tensor("g", (a, b)),
            self.ricci_scalar(),
        )
        equations = evaluate_field_equations(metric_euler, Number(0), self.backend)
        projected = self.backend.evaluate_sympy(metric_euler)
        for row in range(4):
            for column in range(4):
                self.assertEqual(
                    sp.simplify(
                        projected.component(row, column)
                        - self.backend.geometry.einstein_covariant[row, column]
                    ),
                    0,
                )
        self.assertEqual(len(equations.independent_metric), 10)
        self.assertEqual(equations.scalar.scalar, Number(0))
        encoded = json.loads(json.dumps(equations.to_data()))
        self.assertEqual(ComponentFieldEquations.from_data(encoded), equations)
        self.assertEqual(
            ComponentEvaluation.from_data(equations.metric.to_data()),
            equations.metric,
        )

    def test_model_bound_backend(self) -> None:
        model = ModelSpec("reference", Number(1), dimension=DimensionSpec(4))
        backend = SympyComponentBackend.from_model(model, spatially_flat_flrw_ansatz())
        self.assertEqual(backend.geometry.dimension, 4)


if __name__ == "__main__":
    unittest.main()
