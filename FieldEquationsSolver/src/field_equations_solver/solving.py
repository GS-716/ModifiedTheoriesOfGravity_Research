"""Optional, read-only reduction of already projected field equations.

No variational operations live here. Expressions cross the component boundary
using the existing scalar IR adapters. Solver exports are separate from runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import ast
from itertools import combinations
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

SOLVING_SCHEMA_VERSION = "1.0"


def _ir(value):
    return value if isinstance(value, Expr) else sympy_scalar_to_ir(sp.sympify(value))


def _sp(value):
    return ir_scalar_to_sympy(value)


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

    def to_data(self):
        return {"rules": [[a.to_data(), b.to_data()] for a, b in self.rules],
                "status": self.status, "residuals": [[k, v.to_data()] for k, v in self.residuals],
                "nonzero_conditions": [v.to_data() for v in self.nonzero_conditions],
                "unresolved": list(self.unresolved), "origin": self.origin,
                "domain_conditions": list(self.domain_conditions)}

    @classmethod
    def from_data(cls, data):
        return cls(tuple((expr_from_data(a), expr_from_data(b)) for a, b in data["rules"]),
                   data["status"], tuple((k, expr_from_data(v)) for k, v in data["residuals"]),
                   tuple(expr_from_data(v) for v in data["nonzero_conditions"]),
                   tuple(data["unresolved"]), data["origin"], tuple(data.get("domain_conditions", ())))


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
                "status": self.status, "diagnostics": list(self.diagnostics)}

    @classmethod
    def from_data(cls, data):
        if data["schema_version"] != SOLVING_SCHEMA_VERSION:
            raise ValueError("Versión de resolución no soportada.")
        return cls(data["source_run_id"], data["source_fingerprint"], data["source_results"],
                   None if data["ansatz"] is None else GeometryAnsatz.from_data(data["ansatz"]),
                   tuple(ReducedEquation.from_data(e) for e in data["original_equations"]),
                   tuple(ReducedEquation.from_data(e) for e in data["equations"]),
                   tuple((tuple(k), expr_from_data(v)) for k, v in data["mixed_components"]),
                   tuple(expr_from_data(v) for v in data["unknowns"]), data["classification"],
                   tuple(expr_from_data(v) for v in data["nonzero_conditions"]),
                   tuple((expr_from_data(a), expr_from_data(b)) for a, b in data["specialization_rules"]),
                   tuple(FormalSolution.from_data(s) for s in data["solutions"]),
                   data["backend"], data["status"], tuple(data["diagnostics"]))

    def verify(self, rules: Mapping, *, origin="user") -> FormalSolution:
        """Check every ORIGINAL covariant entry and scalar equation, even zeros."""
        raw_rules = tuple((_ir(a), _ir(b)) for a, b in rules.items())
        converted = {_sp(a): _sp(b) for a, b in raw_rules}
        converted = _resolve_solution_rules(converted)
        allowed = {_sp(u) for u in self.unknowns}
        if self.status != "unavailable" and any(key not in allowed for key in converted):
            raise ValueError("Las reglas solo pueden sustituir las incógnitas declaradas.")
        profile = {_sp(a): _sp(b) for a, b in self.specialization_rules}
        residuals, unresolved, conditions, domain_conditions = [], [], [], []
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
        for condition in (*self.nonzero_conditions, *_domain(tuple(v for _, v in raw_rules))):
            try:
                value = sp.simplify(_substitute(_substitute(_sp(condition), profile), converted))
                if value == 0 or value.has(sp.nan, sp.zoo):
                    rejected = True
                    unresolved.append("Dominio singular: " + str(condition))
                elif value.is_zero is not False:
                    conditions.append(_ir(value))
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
        status = "rejected" if rejected else "undetermined" if unresolved else "verified_on_domain"
        # A singular candidate cannot encode infinity in the scalar IR. Keep the
        # supplied rules and diagnostic instead, without inventing a zero residual.
        stored_rules = raw_rules if rejected else tuple((_ir(a), _ir(b)) for a, b in converted.items())
        return FormalSolution(stored_rules, status,
                              tuple(residuals), tuple(conditions), tuple(unresolved), origin, tuple(domain_conditions))

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


def solve_field_equations(run, *, specialization: AnsatzSpecialization | None = None,
                          use_specialized=False, solve=True, wolfram_bridge=None,
                          eliminate=(), output_root=None, compile_pdf=True, display_policy=None):
    """Reduce a completed EngineRun/RunPackage; never mutate or rerun it.

    By default consume the generic projection. use_specialized=True explicitly
    opts into the existing run specialization. A new specialization substitutes
    the already projected components, without deriving/projecting again.
    """
    package = getattr(run, "package", run)
    if use_specialized and specialization is not None:
        raise ValueError("Use specialization sobre la proyección genérica, o use_specialized=True para la especialización existente; no ambos.")
    fingerprint = calculation_fingerprint(package.model, package.lagrangian,
                                         package.momenta, package.euler, package.noether)
    result = FieldEquationSolution(package.run_id, fingerprint, package.to_data(), None)
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
                guards = tuple(dict.fromkeys((*guards, _ir(determinant))))
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
                    bridge = wolfram_bridge or FieldEquationWolframBridge(timeout_seconds=120)
                    result = _solve_wolfram(result, bridge, eliminate)
    if output_root is not None:
        directory = result.export(output_root, compile_pdf=compile_pdf, display_policy=display_policy)
        result = replace(result, output_directory=directory)
    return result


def _solve_wolfram(result, bridge, eliminate):
    if not bridge.available:
        return replace(result, status="symbolic", backend={"status": "unavailable"},
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
        solutions = []
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
                if not any(previous.rules == verified.rules and previous.status == verified.status for previous in solutions):
                    solutions.append(verified)
            except Exception as error:
                response.setdefault("diagnostics", []).append("Candidato no verificable: " + str(error))
        # DSolve/Solve need not return every singular branch: never call this complete.
        status = "formal_family" if any(s.status == "verified_on_domain" for s in solutions) else "partial"
        return replace(result, backend=response, solutions=tuple(solutions), status=status,
                       diagnostics=result.diagnostics + tuple(response.get("diagnostics", ())))
    except Exception as error:
        return replace(result, status="partial", backend={"status": "unavailable", "reason": str(error)},
                       diagnostics=result.diagnostics + ("Wolfram: " + str(error),))


# Notebook spelling requested by the user; both names have the same implementation.
solveFieldEquations = solve_field_equations
