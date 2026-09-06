"""Compact solver-only reports; the tensor run's exports are never overwritten."""
from dataclasses import asdict
import hashlib
import json
import re
from pathlib import Path

import sympy as sp

from tensor_engine.components import ir_scalar_to_sympy, sympy_scalar_to_ir
from tensor_engine.exporting import RunExporter, display_expr_to_latex
from tensor_engine.ir import Scalar, walk
from tensor_engine.model import ModelSpec
from tensor_engine.presentation import DisplayPolicy, PresentationBuilder


def _latex_text(text):
    replacements = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
                    "$": r"\$", "%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_",
                    "^": r"\textasciicircum{}", "~": r"\textasciitilde{}"}
    return "".join(replacements.get(c, c) for c in text)


_AUXILIARY_CONSTANT_PATTERNS = (
    re.compile(r"^C_[A-Za-z][A-Za-z0-9]*$"),
    re.compile(r"^poly[A-Za-z][A-Za-z0-9]*D[0-9]+C[0-9]+$"),
    re.compile(r"^power[A-Za-z][A-Za-z0-9]*E[A-Za-z0-9]+[AB]$"),
    re.compile(r"^integrationConstantX*[0-9]+$"),
)

_STATUS_LABELS = {
    "verified_on_domain": "verificada en el dominio declarado",
    "verified_with_pending_branches": "hay familias verificadas y ramas pendientes",
    "partially_solved": "sistema resuelto parcialmente",
    "no_verified_candidate": "ningún candidato verificado",
    "undetermined": "resultado indeterminado",
    "rejected": "candidato descartado",
    "partial": "resultado parcial",
}

_METHOD_LABELS = {
    "constant_branch": "búsqueda de soluciones constantes",
    "degenerate_metric_branch": "rama métrica degenerada",
    "singular_denominator_branch": "rama de denominador nulo",
}


def _is_auxiliary_constant(name):
    return any(pattern.fullmatch(name) for pattern in _AUXILIARY_CONSTANT_PATTERNS)


def _solution_constant_aliases(solution):
    """Assign readable, family-local labels without mutating solver expressions."""

    names = set()
    expressions = [item for rule in solution.rules for item in rule]
    expressions.extend(solution.nonzero_conditions)
    expressions.extend(value for _, value in solution.residuals)
    for expr in expressions:
        for node in walk(expr):
            if isinstance(node, Scalar) and _is_auxiliary_constant(node.name):
                names.add(node.name)
    names.update(name for name in solution.free_parameters if _is_auxiliary_constant(name))
    return {name: f"C_{position}" for position, name in enumerate(sorted(names), start=1)}


def _method_label(origin):
    if origin in _METHOD_LABELS:
        return _METHOD_LABELS[origin]
    polynomial = re.fullmatch(r"polynomial_degree_([0-9]+)", origin)
    if polynomial:
        return f"ansatz polinomial de grado {polynomial.group(1)}"
    power = re.fullmatch(r"power_(m?[0-9]+)", origin)
    if power:
        exponent = power.group(1).replace("m", "-")
        return f"ansatz de potencia con exponente {exponent}"
    return origin.replace("_", " ")


def solution_latex(result, display_policy=None):
    policy = display_policy or DisplayPolicy()
    model = ModelSpec.from_data(result.source_results["model"])
    builder = PresentationBuilder(model, policy)
    presentation = {}
    display_aliases = {}
    def render(key, expr, aliases=None):
        record = builder.expression(expr)
        presentation[key] = record.to_data()
        displayed = record.presentation
        if aliases:
            try:
                sympy_expression = ir_scalar_to_sympy(displayed)
                substitutions = {
                    sp.Symbol(source): sp.Symbol(
                        "integrationConstant" + target.removeprefix("C_")
                    )
                    for source, target in aliases.items()
                }
                displayed = sympy_scalar_to_ir(sympy_expression.xreplace(substitutions))
                display_aliases[key] = dict(aliases)
            except Exception:
                # Aliasing is strictly cosmetic. Unsupported IR keeps its original form.
                pass
        tex = display_expr_to_latex(displayed)
        return re.sub(r"\\mathrm\{integrationConstantX*\}_\{([0-9]+)\}", r"C_{\1}", tex)
    def equation(tex):
        # Break large sums at top-level terms in the IR if needed at call sites;
        # this width limit is the final guard against clipping arbitrary ansatz.
        return r"\[\adjustbox{max width=\linewidth}{$\displaystyle " + tex + r"$}\]"
    lines = [r"\documentclass[10pt]{article}", r"\usepackage[utf8]{inputenc}",
             r"\usepackage[T1]{fontenc}", r"\usepackage{amsmath,amssymb,adjustbox}",
             r"\usepackage[margin=2cm]{geometry}", r"\begin{document}",
             r"\begin{center}\large Reducción y resolución formal\end{center}",
             r"\textbf{Modelo:} " + _latex_text(model.name) + r"\par",
             equation(r"\text{Lagrangiano:}\qquad L=" + render("lagrangian", model.lagrangian))]
    if result.ansatz:
        lines.append(r"\textbf{Ansatz:} " + _latex_text(result.ansatz.name) + r"\par")
        if result.ansatz.scalar_field is not None:
            prefix = (r"\text{Especialización de }\phi:\quad " if result.specialization_rules else
                      r"\text{Campo escalar sin especialización posterior:}\quad ")
            lines.append(equation(prefix + r"\phi=" + render("profile", result.ansatz.scalar_field)))
    status_label = _STATUS_LABELS.get(result.status, result.status.replace("_", " "))
    lines.append(r"\textbf{Estado global:} " + _latex_text(status_label) + r"\par")
    domain_parts = []
    for index, condition in enumerate(result.nonzero_conditions):
        domain_parts.append(render(f"domain_{index}", condition) + r"\neq0")
    if domain_parts:
        lines.append(equation(r"\text{Dominio de trabajo:}\qquad " + r",\quad ".join(domain_parts)))
    else:
        lines.append(r"\textbf{Dominio de trabajo:} sin restricciones adicionales registradas.\par")
    assumptions = [item.get("source", "") for item in result.classification.get("domain_assumptions", ())]
    if assumptions:
        lines.append(r"\textbf{Supuestos declarados:} " + _latex_text("; ".join(assumptions)) + r"\par")
    lines.append(r"\section*{Ecuaciones combinadas}")
    for eq in result.equations:
        if eq.role == "diagonal_difference":
            lines.append(equation(eq.label + "=0"))
    lines.append(r"\section*{Combinaciones proyectadas y reducidas}")
    for eq in result.equations:
        if eq.role == "diagonal_difference":
            lines.append(equation(eq.label + "=" + render(eq.key, eq.reduced) + "=0"))
    lines.append(r"\section*{Ecuaciones absolutas y escalares}")
    for eq in result.equations:
        if eq.role != "diagonal_difference" and (eq.role != "off_diagonal" or eq.status != "zero"):
            lines.append(equation(eq.label + "=" + render(eq.key, eq.reduced) + "=0"))
    if any(e.role == "off_diagonal" for e in result.equations) and all(
            e.status == "zero" for e in result.equations if e.role == "off_diagonal"):
        lines.append(r"Componentes fuera de la diagonal: $0=0$.\par")
    lines.append(r"\section*{Familias verificadas}")
    verified_solutions = [solution for solution in result.solutions
                          if solution.status == "verified_on_domain"]
    if not verified_solutions:
        lines.append("No se obtuvo una familia formal verificada; se conservan las ecuaciones anteriores.\\par")
    for i, solution in enumerate(verified_solutions):
        aliases = _solution_constant_aliases(solution)
        lines.append(_latex_text(f"Familia {i + 1}: verificada en todas las ecuaciones, en el dominio declarado.") + r"\par")
        for j, (lhs, rhs) in enumerate(solution.rules):
            lines.append(equation(render(f"solution_{i}_{j}_lhs", lhs, aliases) + "="
                                  + render(f"solution_{i}_{j}", rhs, aliases)))
        lines.append(r"\textbf{Método:} " + _latex_text(_method_label(solution.origin)) + r"\par")
        parameters = list(solution.free_parameters)
        if parameters:
            labels = [render(f"solution_{i}_parameter_{j}", Scalar(parameter), aliases)
                      for j, parameter in enumerate(parameters)]
            lines.append(r"Parámetros libres en esta familia: $"
                         + r",\; ".join(labels) + r"$.\par")
        if solution.nonzero_conditions:
            restrictions = [render(f"solution_{i}_domain_{j}", condition, aliases) + r"\neq0"
                            for j, condition in enumerate(solution.nonzero_conditions)]
            lines.append(equation(r"\text{Restricciones no nulas:}\qquad "
                                  + r",\quad ".join(restrictions)))
        if solution.unresolved:
            lines.append(_latex_text("Pendiente: " + "; ".join(solution.unresolved)) + r"\par")
        if result.ansatz and result.ansatz.assumptions:
            lines.append(_latex_text("Supuestos: " + "; ".join(result.ansatz.assumptions)) + r"\par")
    counts = result.search_summary.get("candidate_status_counts", {})
    if counts:
        lines.append(r"\section*{Auditoría de ramas no aceptadas}")
        lines.append(_latex_text(
            "Candidatos conservados solo en results.json: "
            + ", ".join(f"{key}={value}" for key, value in counts.items() if key != "verified_on_domain")
            + "."
        ) + r"\par")
    classification = result.classification
    if classification:
        lines.append(_latex_text(f"Sistema: {classification['kind']}; orden máximo: {classification['max_derivative_order']}.") + r"\par")
        if classification.get("unconstrained_unknowns"):
            lines.append(_latex_text("Funciones o parámetros libres: " + ", ".join(classification["unconstrained_unknowns"])) + r"\par")
    for reason in result.diagnostics:
        lines.append(_latex_text(reason) + r"\par")
    completeness = result.search_summary.get(
        "completeness_reason",
        "La completitud de las familias y ramas singulares no está certificada.",
    )
    lines.append(_latex_text("Completitud: no demostrada. " + completeness) + r"\par")
    lines.append(r"\end{document}")
    return "\n".join(lines), {
        "policy": asdict(policy),
        "expressions": presentation,
        "display_aliases": display_aliases,
        "solver_summary": {
            "model": model.name,
            "ansatz": None if result.ansatz is None else result.ansatz.name,
            "status": result.status,
            "completeness_proven": result.search_summary.get("completeness_proven", False),
            "candidate_status_counts": result.search_summary.get("candidate_status_counts", {}),
        },
    }


def export_solution(result, output_root, *, compile_pdf=True, display_policy=None):
    data = result.to_data()
    content = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)
    digest = hashlib.sha256(content.encode()).hexdigest()
    # A new solver bundle leaves every byte in the tensor bundle intact.
    directory = Path(output_root) / ("field-equations-" + digest[:16])
    directory.mkdir(parents=True, exist_ok=True)
    tex, presentation = solution_latex(result, display_policy)
    for name, text in (("results.json", content), ("report.tex", tex),
                       ("presentation.json", json.dumps(presentation, ensure_ascii=False, indent=2))):
        RunExporter._write_atomic(directory / name, text)
    diagnostic = None
    if compile_pdf:
        _, diagnostic = RunExporter(directory)._compile_pdf(directory)
    elif (directory/"report.pdf").exists():
        # This PDF belongs to the previous presentation of this solver bundle.
        # Never let a changed report.tex be paired with a stale PDF.
        (directory/"report.pdf").unlink()
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()
             if p.name in {"results.json", "presentation.json", "report.tex", "report.pdf"}}
    manifest = {"schema_version": "1.1", "kind": "field_equation_solution",
                "source_run_id": result.source_run_id, "source_fingerprint": result.source_fingerprint,
                "result_sha256": digest, "files": files, "pdf_diagnostic": diagnostic,
                "global_status": result.status,
                "completeness_proven": result.search_summary.get("completeness_proven", False),
                "candidate_status_counts": result.search_summary.get("candidate_status_counts", {})}
    RunExporter._write_atomic(directory / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return directory
