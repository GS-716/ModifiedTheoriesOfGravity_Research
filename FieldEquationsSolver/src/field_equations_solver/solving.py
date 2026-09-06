"""Optional, read-only reduction of already projected field equations.

No variational operations live here. Expressions cross the component boundary
using the existing scalar IR adapters. Solver exports are separate from runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import ast
from itertools import combinations, product
import json
from pathlib import Path
from typing import Any, Mapping

import sympy as sp
from sympy.core.function import AppliedUndef

from tensor_engine.components import (
    AnsatzSpecialization, ComponentEvaluation, GeometryAnsatz,
    SympyComponentBackend, ir_scalar_to_sympy, sympy_scalar_to_ir,
)
from tensor_engine.ir import Expr, Function, Number, Power, Variance, expr_from_data, walk
from tensor_engine.wolfram_bridge import calculation_fingerprint

from .bridge import FieldEquationWolframBridge

SOLVING_SCHEMA_VERSION = "1.1"


def _ir(value):
    return value if isinstance(value, Expr) else sympy_scalar_to_ir(sp.sympify(value))


def _sp(value):
    return ir_scalar_to_sympy(value)


@dataclass(frozen=True)
class SolverSearchPolicy:
    """Bounded exhaustive catalogue of safe symbolic searches.

    Exhaustive means that every enabled search class and parameter scenario is
    attempted. It never means that the mathematical solution set is complete.
    """

    constant_branches: bool = True
    factor_branches: bool = True
    singular_branches: bool = True
    polynomial_degrees: tuple[int, ...] = (1, 2)
    power_exponents: tuple[int, ...] = (-2, -1, 1, 2)
    zero_or_nonzero_parameters: tuple[str, ...] = ("alpha", "q", "beta0")
    required_nonzero_parameters: tuple[str, ...] = ("ell",)
    max_parameter_scenarios: int = 16
    max_factor_branches: int = 24
    max_candidates: int = 96
    max_expression_ops: int = 2500
    use_wolfram: bool = True

    def to_data(self) -> dict[str, Any]:
        return {
            "constant_branches": self.constant_branches,
            "factor_branches": self.factor_branches,
            "singular_branches": self.singular_branches,
            "polynomial_degrees": list(self.polynomial_degrees),
            "power_exponents": list(self.power_exponents),
            "zero_or_nonzero_parameters": list(self.zero_or_nonzero_parameters),
            "required_nonzero_parameters": list(self.required_nonzero_parameters),
            "max_parameter_scenarios": self.max_parameter_scenarios,
            "max_factor_branches": self.max_factor_branches,
            "max_candidates": self.max_candidates,
            "max_expression_ops": self.max_expression_ops,
            "use_wolfram": self.use_wolfram,
        }


@dataclass(frozen=True)
class ReducedEquation:
    key: str
    label: str
    original: Expr
    reduced: Expr
    role: str
    status: str = "retained"
    # Exact rational linear certificate: this equation = sum(c * prior equation).
    dependencies: tuple[tuple[str, Expr], ...] = ()

    def to_data(self):
        return {"key": self.key, "label": self.label, "original": self.original.to_data(),
                "reduced": self.reduced.to_data(), "role": self.role, "status": self.status,
                "dependencies": [[key, value.to_data()] for key, value in self.dependencies]}

    @classmethod
    def from_data(cls, data):
        return cls(data["key"], data["label"], expr_from_data(data["original"]),
                   expr_from_data(data["reduced"]), data["role"], data["status"],
                   tuple((k, expr_from_data(v)) for k, v in data["dependencies"]))


@dataclass(frozen=True)
class FormalSolution:
    rules: tuple[tuple[Expr, Expr], ...]
    status: str
    residuals: tuple[tuple[str, Expr], ...] = ()
    nonzero_conditions: tuple[Expr, ...] = ()
    unresolved: tuple[str, ...] = ()
    origin: str = "candidate"
    domain_conditions: tuple[str, ...] = ()
    mixed_residuals: tuple[tuple[str, Expr], ...] = ()
    free_parameters: tuple[str, ...] = ()
    branch_conditions: tuple[str, ...] = ()

    def to_data(self):
        return {"rules": [[a.to_data(), b.to_data()] for a, b in self.rules],
                "status": self.status, "residuals": [[k, v.to_data()] for k, v in self.residuals],
                "nonzero_conditions": [v.to_data() for v in self.nonzero_conditions],
                "unresolved": list(self.unresolved), "origin": self.origin,
                "method": self.origin,
                "domain_conditions": list(self.domain_conditions),
                "mixed_residuals": [[k, v.to_data()] for k, v in self.mixed_residuals],
                "free_parameters": list(self.free_parameters),
                "branch_conditions": list(self.branch_conditions)}

    @classmethod
    def from_data(cls, data):
        return cls(tuple((expr_from_data(a), expr_from_data(b)) for a, b in data["rules"]),
                   data["status"], tuple((k, expr_from_data(v)) for k, v in data["residuals"]),
                   tuple(expr_from_data(v) for v in data["nonzero_conditions"]),
                   tuple(data["unresolved"]), data.get("origin", data.get("method", "candidate")),
                   tuple(data.get("domain_conditions", ())),
                   tuple((k, expr_from_data(v)) for k, v in data.get("mixed_residuals", ())),
                   tuple(data.get("free_parameters", ())),
                   tuple(data.get("branch_conditions", ())))


_RELATIONS = {"eq": sp.Eq, "ne": sp.Ne, "gt": sp.Gt, "ge": sp.Ge, "lt": sp.Lt, "le": sp.Le}


def _assumption_records(model, ansatz):
    """Read a small scalar predicate grammar, never eval/sympify user text."""
    texts = list(model.assumptions) + list(ansatz.assumptions)
    unsupported = []
    for parameter in model.parameters:
        for assumption in parameter.assumptions:
            suffix = {"positive": ">0", "negative": "<0", "nonzero": "!=0",
                      "nonnegative": ">=0", "nonpositive": "<=0"}.get(assumption)
            if suffix:
                texts.append(parameter.name + suffix)
            elif assumption != "real":
                unsupported.append(parameter.name + ":" + assumption)
    def scalar(node):
        if isinstance(node, ast.Name):
            if node.id == "D" and not model.dimension.is_symbolic:
                return sp.Integer(model.dimension.value)
            return sp.Symbol(node.id)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return sp.Integer(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -scalar(node.operand) if isinstance(node.op, ast.USub) else scalar(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = scalar(node.left), scalar(node.right)
            if isinstance(node.op, ast.Add): return a+b
            if isinstance(node.op, ast.Sub): return a-b
            if isinstance(node.op, ast.Mult): return a*b
            if isinstance(node.op, ast.Div): return a/b
            if isinstance(node.op, ast.Pow): return a**b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            return sp.Function(node.func.id)(*(scalar(a) for a in node.args))
        raise ValueError("Supuesto fuera de la gramática escalar.")
    predicates = []
    for text in dict.fromkeys(texts):
        try:
            tree = ast.parse(text, mode="eval").body
            if not isinstance(tree, ast.Compare) or len(tree.ops) != 1:
                raise ValueError(text)
            op = {ast.Eq: "eq", ast.NotEq: "ne", ast.Gt: "gt", ast.GtE: "ge", ast.Lt: "lt", ast.LtE: "le"}[type(tree.ops[0])]
            predicates.append({"op": op, "lhs": _ir(scalar(tree.left)).to_data(),
                               "rhs": _ir(scalar(tree.comparators[0])).to_data(), "source": text})
        except (SyntaxError, ValueError, KeyError, TypeError):
            unsupported.append(text)
    return predicates, unsupported


def raise_metric_equation(metric: ComponentEvaluation, ansatz: GeometryAnsatz) -> sp.ImmutableMatrix:
    """E^a_b = sum_c g^{ac} E_cb; also valid for non-diagonal metrics."""
    if len(metric.free_indices) != 2 or any(i.variance is not Variance.DOWN for i in metric.free_indices):
        raise ValueError("Se requieren componentes covariantes E_ab, con dos índices inferiores.")
    if metric.dimension != ansatz.dimension:
        raise ValueError("La dimensión de E_ab no coincide con la métrica.")
    g = sp.Matrix([[_sp(e) for e in row] for row in ansatz.metric_covariant])
    e = sp.Matrix(ansatz.dimension, ansatz.dimension, lambda a, b: _sp(metric.component(a, b)))
    return sp.ImmutableMatrix((g.inv() * e).applyfunc(sp.factor_terms))


def analyze_redundancy(equations: tuple[ReducedEquation, ...]) -> tuple[ReducedEquation, ...]:
    """Only certify constant rational linear dependence, never divide by parameters.

    Independence is over Q for the displayed expressions, NOT a claim of
    differential/algebraic functional independence (Noether identities included).
    """
    basis, result = [], []
    for equation in equations:
        value = sp.expand(_sp(equation.reduced))
        if value == 0:
            result.append(replace(equation, status="zero", reduced=Number(0)))
            continue
        vectors = [sp.expand(_sp(e.reduced)).as_coefficients_dict() for e in basis]
        target = value.as_coefficients_dict()
        monomials = sorted(set(target).union(*(set(v) for v in vectors)), key=sp.default_sort_key)
        certificate = None
        if basis:
            matrix = sp.Matrix([[v.get(m, 0) for v in vectors] for m in monomials])
            rhs = sp.Matrix([target.get(m, 0) for m in monomials])
            try:
                weights, parameters = matrix.gauss_jordan_solve(rhs)
                if not parameters.rows and all(w.is_Rational for w in weights):
                    certificate = tuple((e.key, _ir(w)) for e, w in zip(basis, weights) if w != 0)
            except ValueError:
                pass
        if certificate is not None:
            result.append(replace(equation, status="redundant", dependencies=certificate))
        else:
            basis.append(equation)
            result.append(replace(equation, status="linearly_independent_over_Q"))
    return tuple(result)


def classify_system(expressions, unknowns, coordinates):
    """Structural classification, without asserting nonlinear independence."""
    expressions = tuple(sp.sympify(e) for e in expressions if e != 0)
    unknowns = tuple(unknowns)
    functions = tuple(u for u in unknowns if isinstance(u, AppliedUndef))
    derivatives = set().union(*(e.atoms(sp.Derivative) for e in expressions)) if expressions else set()
    derivatives = {d for d in derivatives if any(d.has(u) for u in functions)}
    algebraic = [e for e in expressions if not e.has(sp.Derivative) and any(e.has(u) for u in functions)]
    parameter_constraints = [e for e in expressions if not any(e.has(u) for u in functions)]
    differential_variables = {v for d in derivatives for v, _ in d.variable_count}
    active_functions = [u for u in functions if any(e.has(u) for e in expressions)]
    pde = any(len(set(u.free_symbols).intersection(coordinates)) > 1 for u in active_functions)
    if not derivatives:
        kind = "algebraic"
    elif pde:
        kind = "mixed" if algebraic or any(len(u.args) == 1 for u in active_functions) else "PDE"
    elif algebraic:
        kind = "DAE"
    elif len(differential_variables) > 1:
        kind = "mixed"
    else:
        kind = "ODE"
    return {"kind": kind, "contains_pde": bool(derivatives and pde),
            "unknowns": [str(u) for u in unknowns],
            "independent_variables": [str(v) for v in coordinates if any(e.has(v) for e in expressions)],
            "max_derivative_order": int(max((sum(n for _, n in d.variable_count) for d in derivatives), default=0)),
            "orders": {str(u): int(max((sum(n for _, n in d.variable_count) for d in derivatives if d.has(u)), default=0))
                       for u in functions},
            "unconstrained_unknowns": [str(u) for u in unknowns if not any(e.has(u) for e in expressions)],
            "algebraic_constraints": [_ir(e).to_data() for e in algebraic],
            "parameter_constraints": [_ir(e).to_data() for e in parameter_constraints],
            "independence_scope": "Solo dependencia lineal exacta sobre Q; independencia diferencial no certificada."}


def _substitute(expr, rules):
    # Exactly one simultaneous replacement preserves even a reparameterization
    # f(r) -> f(r)+1. Reapplying that profile would silently change the ansatz.
    value = expr.subs(rules, simultaneous=True).doit()
    return sp.factor(sp.cancel(value))


def _resolve_solution_rules(rules):
    resolved = {}
    visiting = set()
    def resolve(key):
        if key in resolved:
            return resolved[key]
        if key in visiting:
            raise ValueError("Reglas de solución cíclicas o autorreferentes.")
        visiting.add(key)
        value = rules[key]
        dependencies = {other: resolve(other) for other in rules if value != key and value.has(other)}
        resolved[key] = value.subs(dependencies, simultaneous=True).doit()
        visiting.remove(key)
        return resolved[key]
    for key in rules:
        resolve(key)
    return resolved


@dataclass(frozen=True)
class FieldEquationSolution:
    source_run_id: str
    source_fingerprint: str
    source_results: dict[str, Any]
    ansatz: GeometryAnsatz | None
    original_equations: tuple[ReducedEquation, ...] = ()
    equations: tuple[ReducedEquation, ...] = ()
    mixed_components: tuple[tuple[tuple[int, int], Expr], ...] = ()
    unknowns: tuple[Expr, ...] = ()
    classification: dict[str, Any] = field(default_factory=dict)
    nonzero_conditions: tuple[Expr, ...] = ()
    specialization_rules: tuple[tuple[Expr, Expr], ...] = ()
    solutions: tuple[FormalSolution, ...] = ()
    backend: dict[str, Any] = field(default_factory=dict)
    search_policy: dict[str, Any] = field(default_factory=dict)
    search_summary: dict[str, Any] = field(default_factory=dict)
    status: str = "symbolic"
    diagnostics: tuple[str, ...] = ()
    output_directory: Path | None = field(default=None, compare=False)

    def to_data(self):
        return {"schema_version": SOLVING_SCHEMA_VERSION,
                "source_run_id": self.source_run_id, "source_fingerprint": self.source_fingerprint,
                "source_results": self.source_results,
                "ansatz": None if self.ansatz is None else self.ansatz.to_data(),
                "original_equations": [e.to_data() for e in self.original_equations],
                "equations": [e.to_data() for e in self.equations],
                "mixed_components": [[list(k), v.to_data()] for k, v in self.mixed_components],
                "unknowns": [v.to_data() for v in self.unknowns],
                "classification": self.classification,
                "nonzero_conditions": [v.to_data() for v in self.nonzero_conditions],
                "specialization_rules": [[a.to_data(), b.to_data()] for a, b in self.specialization_rules],
                "solutions": [s.to_data() for s in self.solutions], "backend": self.backend,
                "search_policy": self.search_policy, "search_summary": self.search_summary,
                "status": self.status, "diagnostics": list(self.diagnostics)}

    @classmethod
    def from_data(cls, data):
        if data["schema_version"] not in {"1.0", SOLVING_SCHEMA_VERSION}:
            raise ValueError("Versión de resolución no soportada.")
        return cls(
            source_run_id=data["source_run_id"], source_fingerprint=data["source_fingerprint"],
            source_results=data["source_results"],
            ansatz=None if data["ansatz"] is None else GeometryAnsatz.from_data(data["ansatz"]),
            original_equations=tuple(ReducedEquation.from_data(e) for e in data["original_equations"]),
            equations=tuple(ReducedEquation.from_data(e) for e in data["equations"]),
            mixed_components=tuple((tuple(k), expr_from_data(v)) for k, v in data["mixed_components"]),
            unknowns=tuple(expr_from_data(v) for v in data["unknowns"]),
            classification=data["classification"],
            nonzero_conditions=tuple(expr_from_data(v) for v in data["nonzero_conditions"]),
            specialization_rules=tuple((expr_from_data(a), expr_from_data(b)) for a, b in data["specialization_rules"]),
            solutions=tuple(FormalSolution.from_data(s) for s in data["solutions"]),
            backend=data["backend"], search_policy=data.get("search_policy", {}),
            search_summary=data.get("search_summary", {}), status=data["status"],
            diagnostics=tuple(data["diagnostics"]),
        )

    def verify(self, rules: Mapping, *, origin="user", required_nonzero=(),
               branch_conditions=(), partial_if_unassigned=False) -> FormalSolution:
        """Check all original covariant equations and all stored mixed components."""
        raw_rules = tuple((_ir(a), _ir(b)) for a, b in rules.items())
        converted = {_sp(a): _sp(b) for a, b in raw_rules}
        converted = _resolve_solution_rules(converted)
        allowed = {_sp(u) for u in self.unknowns}
        if self.status != "unavailable" and any(key not in allowed for key in converted):
            raise ValueError("Las reglas solo pueden sustituir las incógnitas declaradas.")
        profile = {_sp(a): _sp(b) for a, b in self.specialization_rules}
        residuals, mixed_residuals, unresolved, conditions, domain_conditions = [], [], [], [], []
        rejected = any(v.has(sp.nan, sp.zoo, sp.oo, -sp.oo) for v in converted.values())
        for equation in self.original_equations:
            try:
                residual = sp.simplify(_substitute(_substitute(_sp(equation.original), profile), converted))
                if residual.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                    rejected = True
                    unresolved.append(equation.key + ": sustitución singular")
                    residuals.append((equation.key, equation.original))
                    continue
                residuals.append((equation.key, _ir(residual)))
                if residual != 0:
                    unresolved.append(equation.key + ": residual no nulo o no demostrado nulo")
                    if residual.is_zero is False:
                        rejected = True
            except Exception as error:
                residuals.append((equation.key, equation.original))
                unresolved.append(equation.key + ": " + str(error))
        for (a, b), expression in self.mixed_components:
            key = f"mixed_{a}_{b}"
            try:
                residual = sp.simplify(_substitute(_substitute(_sp(expression), profile), converted))
                if residual.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                    rejected = True
                    unresolved.append(key + ": sustitución singular")
                    mixed_residuals.append((key, expression))
                    continue
                mixed_residuals.append((key, _ir(residual)))
                if residual != 0:
                    unresolved.append(key + ": residual no nulo o no demostrado nulo")
                    if residual.is_zero is False:
                        rejected = True
            except Exception as error:
                mixed_residuals.append((key, expression))
                unresolved.append(key + ": " + str(error))
        all_nonzero = tuple(dict.fromkeys((*self.nonzero_conditions,
                                          *tuple(_ir(v) for v in required_nonzero),
                                          *_domain(tuple(v for _, v in raw_rules)))))
        for condition in all_nonzero:
            try:
                value = sp.simplify(_substitute(_substitute(_sp(condition), profile), converted))
                if value == 0 or value.has(sp.nan, sp.zoo):
                    rejected = True
                    unresolved.append("Dominio singular: " + str(condition))
                elif value.is_zero is not False or value.free_symbols or isinstance(value, AppliedUndef):
                    conditions.append(_ir(value))
                    domain_conditions.append(str(sp.Ne(value, 0)))
            except Exception as error:
                conditions.append(condition)
                unresolved.append("Dominio no evaluable: " + str(error))
        if not self.original_equations or self.status == "unavailable":
            unresolved.append("Faltan ecuaciones originales; no se puede verificar la solución.")
        for condition in self.classification.get("domain_assumptions", ()):
            try:
                lhs = _substitute(_substitute(_sp(expr_from_data(condition["lhs"])), profile), converted)
                rhs = _substitute(_substitute(_sp(expr_from_data(condition["rhs"])), profile), converted)
                predicate = _RELATIONS[condition["op"]](lhs, rhs)
                if predicate is sp.false:
                    rejected = True
                    unresolved.append("Se viola el supuesto: " + condition["source"])
                elif predicate is not sp.true:
                    domain_conditions.append(str(predicate))
            except Exception as error:
                unresolved.append("Supuesto no verificable: " + str(error))
        unresolved.extend("Supuesto no interpretado: " + text for text in self.classification.get("uninterpreted_assumptions", ()))
        assigned = set(converted)
        missing = [u for u in (_sp(v) for v in self.unknowns)
                   if isinstance(u, AppliedUndef) and u not in assigned]
        status = ("rejected" if rejected else
                  "partial" if unresolved and partial_if_unassigned and missing else
                  "undetermined" if unresolved else "verified_on_domain")
        # A singular candidate cannot encode infinity in the scalar IR. Keep the
        # supplied rules and diagnostic instead, without inventing a zero residual.
        stored_rules = raw_rules if rejected else tuple((_ir(a), _ir(b)) for a, b in converted.items())
        coordinate_names = {str(_sp(c)) for c in self.ansatz.chart.coordinates} if self.ansatz else set()
        free = {str(s) for _, value in stored_rules for s in _sp(value).free_symbols
                if str(s) not in coordinate_names}
        assigned_names = {str(_sp(lhs)) for lhs, _ in stored_rules}
        free.update(name for name in self.classification.get("parameters", ()) if name not in assigned_names)
        return FormalSolution(
            stored_rules, status, tuple(residuals), tuple(conditions), tuple(dict.fromkeys(unresolved)),
            origin, tuple(dict.fromkeys(domain_conditions)), tuple(mixed_residuals),
            tuple(sorted(free)), tuple(dict.fromkeys(str(v) for v in branch_conditions)),
        )

    def export(self, output_root, *, compile_pdf=True, display_policy=None):
        from .reporting import export_solution
        return export_solution(self, output_root, compile_pdf=compile_pdf, display_policy=display_policy)


def _profile_rules(ansatz, specialization):
    if specialization is None:
        return (), ansatz
    effective = specialization.apply(ansatz)  # reuse stationary-field and metric validation
    if (specialization.scalar_field is not None and ansatz.scalar_field is not None
            and not isinstance(ansatz.scalar_field, Function)
            and specialization.scalar_field != ansatz.scalar_field):
        raise ValueError("No se puede recuperar un campo genérico desde un perfil ya sustituido. Use la proyección genérica de la corrida.")
    substitutions = []
    functions = {node for row in ansatz.metric_covariant for e in row for node in walk(e)
                 if isinstance(node, Function)}
    for name, expression in specialization.metric_functions.items():
        substitutions.extend((node, expression) for node in functions if node.name == name)
    if specialization.scalar_field is not None and ansatz.scalar_field is not None:
        substitutions.append((ansatz.scalar_field, specialization.scalar_field))
    return tuple(substitutions), effective


def _domain(expressions):
    guards = set()
    for expression in expressions:
        for node in walk(expression):
            if isinstance(node, Power) and isinstance(node.exponent, Number) and node.exponent.value < 0:
                if not isinstance(node.base, Number):
                    guards.add(node.base)
    return tuple(sorted(guards, key=lambda x: json.dumps(x.to_data(), sort_keys=True)))


def _parameter_scenarios(result: FieldEquationSolution, policy: SolverSearchPolicy):
    available = {}
    for unknown in result.unknowns:
        value = _sp(unknown)
        if isinstance(value, sp.Symbol):
            available[str(value)] = value
    choices = []
    for name in policy.zero_or_nonzero_parameters:
        if name in available:
            symbol = available[name]
            choices.append((
                (name + "=0", ((symbol, sp.Integer(0)),), (), (str(sp.Eq(symbol, 0)),)),
                (name + "!=0", (), (symbol,), (str(sp.Ne(symbol, 0)),)),
            ))
    for name in policy.required_nonzero_parameters:
        if name in available:
            symbol = available[name]
            choices.append(((name + "!=0", (), (symbol,), (str(sp.Ne(symbol, 0)),)),))
    combinations_ = product(*choices) if choices else ((),)
    scenarios = []
    for selected in combinations_:
        rules = tuple(pair for item in selected for pair in item[1])
        nonzero = tuple(value for item in selected for value in item[2])
        conditions = tuple(value for item in selected for value in item[3])
        name = ", ".join(item[0] for item in selected) or "generic"
        scenarios.append({"name": name, "rules": rules, "nonzero": nonzero,
                          "conditions": conditions})
        if len(scenarios) >= policy.max_parameter_scenarios:
            break
    return tuple(scenarios)


def _radial_metric_functions(result: FieldEquationSolution):
    if result.ansatz is None:
        return ()
    coordinate_names = {str(_sp(c)) for c in result.ansatz.chart.coordinates}
    functions = []
    for unknown in result.unknowns:
        value = _sp(unknown)
        if isinstance(value, AppliedUndef) and len(value.args) == 1 and str(value.args[0]) in coordinate_names:
            if any(_sp(entry).has(value) for row in result.ansatz.metric_covariant for entry in row):
                functions.append(value)
    return tuple(sorted(set(functions), key=sp.default_sort_key))


def _solution_identity(solution: FormalSolution):
    return json.dumps({"rules": [[a.to_data(), b.to_data()] for a, b in solution.rules],
                       "conditions": solution.branch_conditions, "origin": solution.origin},
                      sort_keys=True)


def _append_candidate(candidates, candidate, policy):
    if len(candidates) >= policy.max_candidates:
        return False
    identity = _solution_identity(candidate)
    if any(_solution_identity(previous) == identity for previous in candidates):
        return False
    candidates.append(candidate)
    return True


def _fit_template(result, function, template, coefficients, leading, scenario,
                  method, policy):
    profile = {_sp(a): _sp(b) for a, b in result.specialization_rules}
    scenario_rules = dict(scenario["rules"])
    coordinate = function.args[0]
    coefficient_equations = []
    for equation in result.original_equations:
        value = _substitute(_sp(equation.original), profile)
        value = _substitute(value, scenario_rules)
        value = _substitute(value, {function: template})
        if value == 0:
            continue
        if value.atoms(AppliedUndef):
            return (), "otras funciones desconocidas permanecen activas"
        if sp.count_ops(value) > policy.max_expression_ops:
            return (), "expresión excede el límite conservador de complejidad"
        numerator = sp.factor(sp.together(value).as_numer_denom()[0])
        try:
            polynomial = sp.Poly(numerator, coordinate)
        except sp.PolynomialError:
            return (), "los residuales no son polinomiales en la coordenada radial"
        coefficient_equations.extend(polynomial.all_coeffs())
    coefficient_equations = tuple(dict.fromkeys(sp.factor(v) for v in coefficient_equations if v != 0))
    if not coefficient_equations:
        fits = ({},)
    else:
        try:
            fits = sp.solve(coefficient_equations, coefficients, dict=True, simplify=False)
        except Exception as error:
            return (), "ajuste algebraico no evaluable: " + str(error)
    candidates = []
    for fit in fits[: policy.max_candidates]:
        expression = sp.factor(template.subs(fit, simultaneous=True))
        rules = dict(scenario_rules)
        rules[function] = expression
        required = (*scenario["nonzero"], sp.simplify(leading.subs(fit, simultaneous=True)))
        conditions = (*scenario["conditions"], str(sp.Ne(leading, 0)),
                      method + ": " + str(sp.Eq(function, expression)))
        candidate = result.verify(rules, origin=method, required_nonzero=required,
                                  branch_conditions=conditions, partial_if_unassigned=True)
        candidates.append(candidate)
    return tuple(candidates), "evaluated"


def _search_local_branches(result: FieldEquationSolution, policy: SolverSearchPolicy):
    candidates = []
    branch_records = []
    scenarios = _parameter_scenarios(result, policy)
    radial_functions = _radial_metric_functions(result)

    if policy.constant_branches:
        if not radial_functions:
            branch_records.append({"kind": "constant", "status": "not_applicable",
                                   "reason": "No hay función métrica radial desconocida."})
        for function in radial_functions:
            constant = sp.Symbol("C_" + str(function.func))
            for scenario in scenarios:
                rules = dict(scenario["rules"])
                rules[function] = constant
                candidate = result.verify(
                    rules, origin="constant_branch", required_nonzero=(*scenario["nonzero"], constant),
                    branch_conditions=(*scenario["conditions"], str(sp.Ne(constant, 0)),
                                       str(sp.Eq(function, constant))), partial_if_unassigned=True,
                )
                _append_candidate(candidates, candidate, policy)
                branch_records.append({"kind": "constant", "scenario": scenario["name"],
                                       "expression": str(sp.Eq(function, constant)),
                                       "status": candidate.status, "reason": "; ".join(candidate.unresolved)})
                zero_rules = dict(scenario["rules"])
                zero_rules[function] = sp.Integer(0)
                degenerate = result.verify(
                    zero_rules, origin="degenerate_metric_branch",
                    required_nonzero=scenario["nonzero"],
                    branch_conditions=(*scenario["conditions"], str(sp.Eq(function, 0))),
                    partial_if_unassigned=True,
                )
                _append_candidate(candidates, degenerate, policy)
                branch_records.append({"kind": "degenerate", "scenario": scenario["name"],
                                       "expression": str(sp.Eq(function, 0)),
                                       "status": degenerate.status,
                                       "reason": "; ".join(degenerate.unresolved)})

    if policy.factor_branches:
        factors = []
        for equation in result.equations:
            if equation.status in ("zero", "redundant"):
                continue
            numerator = sp.factor(sp.together(_sp(equation.reduced)).as_numer_denom()[0])
            factors.extend(f for f in sp.Mul.make_args(numerator) if f != 0 and not f.is_number)
        for factor in sorted(set(factors), key=sp.default_sort_key)[: policy.max_factor_branches]:
            produced = 0
            if sp.count_ops(factor) <= policy.max_expression_ops:
                for target in (_sp(u) for u in result.unknowns):
                    if not isinstance(target, (sp.Symbol, AppliedUndef)) or not factor.has(target):
                        continue
                    if any(derivative.has(target) for derivative in factor.atoms(sp.Derivative)):
                        continue
                    try:
                        roots = sp.solve(sp.Eq(factor, 0), target, dict=False, simplify=False)
                    except Exception:
                        roots = ()
                    for root in roots[:4]:
                        if root.has(target):
                            continue
                        candidate = result.verify(
                            {target: root}, origin="factor_branch",
                            branch_conditions=(str(sp.Eq(factor, 0)),), partial_if_unassigned=True,
                        )
                        produced += int(_append_candidate(candidates, candidate, policy))
            branch_records.append({"kind": "factor", "expression": str(sp.Eq(factor, 0)),
                                   "status": "evaluated" if produced else "pending",
                                   "candidate_count": produced,
                                   "reason": "" if produced else "No se obtuvo una regla segura sin resolver derivadas."})

    if policy.singular_branches:
        allowed = {_sp(u) for u in result.unknowns}
        for guard in result.nonzero_conditions:
            expression = _sp(guard)
            target = expression if expression in allowed else None
            if target is not None:
                candidate = result.verify(
                    {target: 0}, origin="singular_denominator_branch",
                    branch_conditions=(str(sp.Eq(expression, 0)),), partial_if_unassigned=True,
                )
            else:
                candidate = FormalSolution(
                    (), "rejected",
                    tuple((equation.key, equation.original) for equation in result.original_equations),
                    (guard,),
                    ("Rama singular excluida por el dominio de la métrica o un denominador.",),
                    "singular_denominator_branch", (),
                    tuple((f"mixed_{a}_{b}", value) for (a, b), value in result.mixed_components),
                    (), (str(sp.Eq(expression, 0)),),
                )
            _append_candidate(candidates, candidate, policy)
            branch_records.append({"kind": "singular", "expression": str(sp.Eq(expression, 0)),
                                   "status": "rejected",
                                   "reason": "Contradice una condición no nula del dominio."})

    for function in radial_functions:
        coordinate = function.args[0]
        for scenario in scenarios:
            for degree in policy.polynomial_degrees:
                if degree < 1:
                    continue
                coefficients = tuple(sp.Symbol(f"poly{function.func}D{degree}C{i}") for i in range(degree + 1))
                template = sum(coefficient * coordinate**i for i, coefficient in enumerate(coefficients))
                found, reason = _fit_template(result, function, template, coefficients, coefficients[-1],
                                              scenario, f"polynomial_degree_{degree}", policy)
                for candidate in found:
                    _append_candidate(candidates, candidate, policy)
                branch_records.append({"kind": "polynomial", "scenario": scenario["name"],
                                       "expression": str(sp.Eq(function, template)),
                                       "status": "evaluated" if reason == "evaluated" else "pending",
                                       "candidate_count": len(found), "reason": "" if reason == "evaluated" else reason})
            for exponent in policy.power_exponents:
                suffix = str(exponent).replace("-", "m")
                amplitude = sp.Symbol(f"power{function.func}E{suffix}A")
                offset = sp.Symbol(f"power{function.func}E{suffix}B")
                template = offset + amplitude * coordinate**exponent
                found, reason = _fit_template(result, function, template, (offset, amplitude), amplitude,
                                              scenario, f"power_{exponent}", policy)
                for candidate in found:
                    _append_candidate(candidates, candidate, policy)
                branch_records.append({"kind": "power", "scenario": scenario["name"],
                                       "expression": str(sp.Eq(function, template)),
                                       "status": "evaluated" if reason == "evaluated" else "pending",
                                       "candidate_count": len(found), "reason": "" if reason == "evaluated" else reason})

    summary = {
        "parameter_scenarios": [{"name": scenario["name"],
                                 "conditions": list(scenario["conditions"])} for scenario in scenarios],
        "branches": branch_records,
        "local_candidate_count": len(candidates),
        "completeness_proven": False,
        "completeness_reason": "Una búsqueda simbólica acotada no certifica que no existan otras ramas diferenciales o singulares.",
    }
    return tuple(candidates), summary


def _finalize_solution_status(result: FieldEquationSolution):
    counts = {name: sum(solution.status == name for solution in result.solutions)
              for name in ("verified_on_domain", "rejected", "undetermined", "partial")}
    pending_records = sum(branch.get("status") == "pending"
                          for branch in result.search_summary.get("branches", ()))
    completeness = bool(result.search_summary.get("completeness_proven", False))
    if counts["verified_on_domain"]:
        outcome = "verified_family_found" if completeness and not pending_records else "verified_with_pending_branches"
    elif counts["partial"] or counts["undetermined"]:
        outcome = "partially_solved"
    else:
        outcome = "no_verified_candidate"
    summary = dict(result.search_summary)
    summary.update({"outcome": outcome, "candidate_status_counts": counts,
                    "pending_branch_count": pending_records,
                    "has_verified_family": bool(counts["verified_on_domain"])})
    return replace(result, status=outcome, search_summary=summary)


def solve_field_equations(run, *, specialization: AnsatzSpecialization | None = None,
                          use_specialized=False, solve=True, wolfram_bridge=None,
                          eliminate=(), output_root=None, compile_pdf=True, display_policy=None,
                          search_policy: SolverSearchPolicy | None = None):
    """Reduce a completed EngineRun/RunPackage; never mutate or rerun it.

    By default consume the generic projection. use_specialized=True explicitly
    opts into the existing run specialization. A new specialization substitutes
    the already projected components, without deriving/projecting again.
    """
    package = getattr(run, "package", run)
    policy = search_policy or SolverSearchPolicy()
    if use_specialized and specialization is not None:
        raise ValueError("Use specialization sobre la proyección genérica, o use_specialized=True para la especialización existente; no ambos.")
    fingerprint = calculation_fingerprint(package.model, package.lagrangian,
                                         package.momenta, package.euler, package.noether)
    result = FieldEquationSolution(package.run_id, fingerprint, package.to_data(), None,
                                   search_policy=policy.to_data())
    view = package.specialized if use_specialized else package.projected
    diagnostics = []
    if view is None or view.ansatz_geometry is None:
        result = replace(result, status="unavailable", diagnostics=("No hay proyección con geometría serializada.",))
    else:
        ansatz = view.ansatz_geometry
        result = replace(result, ansatz=ansatz)
        metric, scalar = view.quantity("metric_euler"), view.quantity("scalar_euler")
        completion_note = None
        requested_profile, effective = _profile_rules(ansatz, specialization)
        # Missing generic projections are not zero. If the user explicitly
        # supplies a profile, calculate ONLY missing components from stored E IR.
        # Already available components are substituted and never recalculated.
        if specialization is not None and any(q.components is None for q in (metric, scalar)):
            from tensor_engine.derived import ProjectionStatus
            backend = None
            converted = []
            for quantity in (metric, scalar):
                try:
                    if quantity.components is None:
                        if backend is None:
                            backend = SympyComponentBackend.from_model(package.model, effective)
                        expression = getattr(package.euler, quantity.key)
                        reason = backend.projection_limit_reason(expression)
                        if reason:
                            raise ValueError(reason)
                        components = backend.evaluate(expression)
                    else:
                        replacements = {_sp(a): _sp(b) for a, b in requested_profile}
                        components = replace(quantity.components, values=tuple(
                            (position, _ir(_substitute(_sp(value), replacements)))
                            for position, value in quantity.components.values))
                    converted.append(replace(quantity, components=components, status=ProjectionStatus.COMPLETED,
                                             ansatz_name=effective.name, reason=""))
                except Exception as error:
                    converted.append(replace(quantity, components=None, status=ProjectionStatus.UNAVAILABLE, reason=str(error)))
            metric, scalar = converted
            ansatz = effective
            completion_note = "Componentes genéricas ausentes: proyectadas por primera vez desde E_ab/E_phi almacenados, con el perfil solicitado."
        for quantity in (metric, scalar):
            if quantity.components is None or quantity.status.value != "completed":
                diagnostics.append(quantity.key + ": " + (quantity.reason or "Proyección incompleta."))
        if diagnostics:
            available = []
            replacements = {_sp(a): _sp(b) for a, b in requested_profile}
            for quantity in (metric, scalar):
                if quantity.components is None:
                    continue
                if quantity.key == "scalar_euler":
                    expr = quantity.components.scalar
                    available.append(ReducedEquation("scalar", r"E_\phi", expr,
                                                     _ir(_substitute(_sp(expr), replacements)), "scalar"))
                else:
                    for a in range(ansatz.dimension):
                        for b in range(ansatz.dimension):
                            expr = quantity.components.component(a, b)
                            available.append(ReducedEquation(f"metric_{a}_{b}", f"E_{{{a} {b}}}", expr,
                                                             _ir(_substitute(_sp(expr), replacements)), "available_metric"))
            result = replace(result, ansatz=effective, status="unavailable", diagnostics=tuple(diagnostics),
                             original_equations=tuple(available), equations=analyze_redundancy(tuple(available)))
        else:
            # In the missing-component path all entries already belong to the
            # effective geometry, so do not specialize those entries twice.
            profile = () if completion_note else requested_profile
            profile_sp = {_sp(a): _sp(b) for a, b in profile}
            try:
                mixed = raise_metric_equation(metric.components, ansatz)
            except Exception as error:
                result = replace(result, status="unavailable", diagnostics=(str(error),))
            else:
                coordinates = tuple(_sp(c) for c in ansatz.chart.coordinates)
                names = tuple(c.name for c in ansatz.chart.coordinates)
                label = lambda a: {"tau": r"\tau", "varphi": r"\varphi"}.get(names[a], names[a])
                original = []
                for a in range(ansatz.dimension):
                    for b in range(ansatz.dimension):
                        expr = metric.components.component(a, b)
                        original.append(ReducedEquation(f"metric_{a}_{b}", f"E_{{{label(a)} {label(b)}}}",
                                                       expr, expr, "original_metric"))
                expr = scalar.components.scalar
                original.append(ReducedEquation("scalar", r"E_\phi", expr, expr, "scalar"))
                equations = []
                def append(key, tex, value, role):
                    canonical = _ir(value)
                    reduced = _ir(_substitute(value, profile_sp))
                    equations.append(ReducedEquation(key, tex, canonical, reduced, role))
                for a, b in combinations(range(ansatz.dimension), 2):
                    append(f"difference_{a}_{b}", f"E^{{{label(a)}}}_{{{label(a)}}}-E^{{{label(b)}}}_{{{label(b)}}}",
                           mixed[a, a] - mixed[b, b], "diagonal_difference")
                append("absolute_0", f"E^{{{label(0)}}}_{{{label(0)}}}", mixed[0, 0], "absolute")
                for equation in original:
                    if equation.role == "scalar" or equation.key.split("_")[-1] != equation.key.split("_")[-2]:
                        append(equation.key, equation.label, _sp(equation.original),
                               "scalar" if equation.role == "scalar" else "off_diagonal")
                equations = analyze_redundancy(tuple(equations))
                expressions = [_sp(e.reduced) for e in equations if e.status not in ("zero", "redundant")]
                fields = [_sp(e) for row in effective.metric_covariant for e in row]
                if effective.scalar_field is not None:
                    fields.append(_sp(effective.scalar_field))
                functions = sorted(set().union(*(v.atoms(AppliedUndef) for v in fields)), key=sp.default_sort_key)
                # Parameters in user profiles (e.g. q) are explicit free constants too.
                parameters = sorted(set().union(*(v.free_symbols for v in fields + expressions)) - set(coordinates), key=str)
                unknowns = functions + parameters
                classification = classify_system(expressions, unknowns, coordinates)
                classification["functions"] = [str(f) for f in functions]
                classification["parameters"] = [str(p) for p in parameters]
                classification["projection_source"] = completion_note or "Componentes existentes reutilizadas."
                predicates, unsupported_assumptions = _assumption_records(package.model, effective)
                classification["domain_assumptions"] = predicates
                classification["uninterpreted_assumptions"] = unsupported_assumptions
                guards = _domain([e.original for e in original] + [e.original for e in equations]
                                 + [e for row in ansatz.metric_covariant for e in row]
                                 + [e for _, e in requested_profile])
                determinant = sp.factor(sp.Matrix([[_sp(e) for e in row] for row in effective.metric_covariant]).det())
                try:
                    determinant_guards = tuple(_ir(factor) for factor, _ in sp.factor_list(determinant)[1]
                                               if not factor.is_number)
                except sp.PolynomialError:
                    determinant_guards = (_ir(determinant),)
                guards = tuple(dict.fromkeys((*guards, *determinant_guards)))
                # Factor equations without dividing: zero factors label possible branches,
                # not mutually independent solutions.
                factors = sorted({str(factor) for e in expressions
                                  for factor in sp.Mul.make_args(sp.factor(sp.together(e).as_numer_denom()[0]))
                                  if factor.free_symbols and factor != 0})
                classification["possible_zero_factors"] = factors
                classification["branch_note"] = "Factores candidatos; deben satisfacer todas las ecuaciones y el dominio."
                result = replace(result, ansatz=effective, original_equations=tuple(original), equations=equations,
                                 mixed_components=tuple(((a, b), _ir(mixed[a, b])) for a in range(ansatz.dimension) for b in range(ansatz.dimension)),
                                 unknowns=tuple(_ir(u) for u in unknowns), classification=classification,
                                 nonzero_conditions=guards, specialization_rules=profile,
                                 status="underdetermined" if classification["unconstrained_unknowns"] else "reduced")
                if solve:
                    try:
                        local_solutions, search_summary = _search_local_branches(result, policy)
                        result = replace(result, solutions=local_solutions, search_summary=search_summary)
                    except Exception as error:
                        result = replace(result, search_summary={
                            "completeness_proven": False,
                            "completeness_reason": "La exploración local no terminó: " + str(error),
                            "branches": (),
                        }, diagnostics=result.diagnostics + ("Búsqueda local: " + str(error),))
                    if policy.use_wolfram:
                        bridge = wolfram_bridge or FieldEquationWolframBridge(timeout_seconds=120)
                        result = _solve_wolfram(result, bridge, eliminate)
                    else:
                        result = replace(result, backend={"status": "disabled_by_policy"})
                    result = _finalize_solution_status(result)
    if output_root is not None:
        directory = result.export(output_root, compile_pdf=compile_pdf, display_policy=display_policy)
        result = replace(result, output_directory=directory)
    return result


def _solve_wolfram(result, bridge, eliminate):
    if not bridge.available:
        return replace(result, backend={"status": "unavailable"},
                       diagnostics=result.diagnostics + ("Wolfram Engine no disponible; se conservan ecuaciones reducidas.",))
    eliminated = tuple(_ir(e) for e in eliminate)
    if any(e not in result.unknowns for e in eliminated):
        raise ValueError("Solo pueden eliminarse incógnitas declaradas.")
    options = {"equations": [e.reduced.to_data() for e in result.equations if e.status not in ("zero", "redundant")],
               "assumptions": result.classification.get("domain_assumptions", []),
               "original_equations": [_ir(_substitute(_sp(e.original), {_sp(a): _sp(b) for a, b in result.specialization_rules})).to_data()
                                      for e in result.original_equations],
               "unknowns": [u.to_data() for u in result.unknowns],
               "eliminate": [e.to_data() for e in eliminated],
               "nonzero": [_ir(_substitute(_sp(e), {_sp(a): _sp(b) for a, b in result.specialization_rules})).to_data()
                           for e in result.nonzero_conditions],
               "time_limit": max(1, min(20, bridge.timeout_seconds / 8))}
    try:
        response = bridge.execute(bridge.build_request("solve_field_equations", options=options))
        solutions = list(result.solutions)
        for candidate in response.get("candidates", ()):
            try:
                rules = {expr_from_data(a): expr_from_data(b) for a, b in candidate["rules"]}
                constants = sorted((expr_from_data(c) for c in candidate.get("integration_constants", ())), key=lambda e: str(_sp(e)))
                if constants:
                    # Bijective renaming of arbitrary integration constants lets
                    # equal families share one record. No field equation is used.
                    prefix = "integrationConstant"
                    reserved = {str(_sp(u)) for u in result.unknowns} | {p["name"] for p in result.source_results["model"].get("parameters", ())}
                    while any(name.startswith(prefix) for name in reserved):
                        prefix += "X"
                    renaming = {_sp(c): sp.Symbol(prefix + str(i+1)) for i, c in enumerate(constants)}
                    rules = {a: _ir(sp.factor(_sp(b).xreplace(renaming))) for a, b in rules.items()}
                verified = result.verify(rules, origin=candidate.get("origin", "Wolfram"))
                if verified.status == "verified_on_domain" and candidate.get("verification") != "verified":
                    verified = replace(verified, status="undetermined", unresolved=("Wolfram no confirmó todos los residuales.",))
                if not any(_solution_identity(previous) == _solution_identity(verified) for previous in solutions):
                    solutions.append(verified)
            except Exception as error:
                response.setdefault("diagnostics", []).append("Candidato no verificable: " + str(error))
        # DSolve/Solve need not return every singular branch: never call this complete.
        return replace(result, backend=response, solutions=tuple(solutions),
                       search_summary={**result.search_summary,
                                       "wolfram_status": response.get("status", "unknown")},
                       diagnostics=result.diagnostics + tuple(response.get("diagnostics", ())))
    except Exception as error:
        return replace(result, backend={"status": "unavailable", "reason": str(error)},
                       diagnostics=result.diagnostics + ("Wolfram: " + str(error),))


# Notebook spelling requested by the user; both names have the same implementation.
solveFieldEquations = solve_field_equations
