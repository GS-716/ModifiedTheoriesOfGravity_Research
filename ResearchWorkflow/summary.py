"""Compact cross-model summary built exclusively from completed pipeline objects."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Mapping

import sympy as sp

from tensor_engine.components import ir_scalar_to_sympy, sympy_scalar_to_ir
from tensor_engine.exporting import RunExporter, display_expr_to_latex
from tensor_engine.ir import Function, Scalar, Variance, walk
from tensor_engine.presentation import (
    CompactProjection,
    DisplayPolicy,
    PresentationBuilder,
    independent_curvature_components,
)


@dataclass(frozen=True, slots=True)
class SummaryBundle:
    """Files produced by :func:`make_summary`."""

    output_directory: Path
    tex_path: Path
    json_path: Path
    manifest_path: Path
    pdf_path: Path | None
    model_count: int
    pdf_diagnostic: str | None = None


def _latex_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
        "$": r"\$", "%": r"\%", "&": r"\&", "#": r"\#",
        "_": r"\_", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _normalise_runs(runs: Mapping[str, object] | Iterable[object]) -> tuple[object, ...]:
    values = tuple(runs.values()) if isinstance(runs, Mapping) else tuple(runs)
    if not values:
        raise ValueError("makeSummary requiere al menos una corrida ejecutada en el notebook.")
    return values


def _normalise_solutions(
    solutions: Mapping[str, object] | Iterable[object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    if solutions is None:
        return {}, {}
    if isinstance(solutions, Mapping):
        values = tuple(solutions.values())
        by_name = dict(solutions)
    else:
        values = tuple(solutions)
        by_name = {}
    by_run_id = {
        str(solution.source_run_id): solution
        for solution in values
        if getattr(solution, "source_run_id", None)
    }
    return by_name, by_run_id


def _display(builder: PresentationBuilder, expression, *, assumptions=()) -> str:
    record = builder.expression(expression, assumptions=tuple(assumptions))
    return display_expr_to_latex(record.presentation)


def _component_label(label: str, free_indices, position: tuple[int, ...]) -> str:
    upper = "".join(
        str(value)
        for index, value in zip(free_indices, position, strict=True)
        if index.variance is Variance.UP
    )
    lower = "".join(
        str(value)
        for index, value in zip(free_indices, position, strict=True)
        if index.variance is Variance.DOWN
    )
    result = rf"\left[{label}\right]"
    if lower:
        result += rf"_{{{lower}}}"
    if upper:
        result += rf"^{{{upper}}}"
    return result


def _projection_lines(
    quantity,
    label: str,
    builder: PresentationBuilder,
    *,
    assumptions: tuple[str, ...],
    max_components: int,
    curvature: bool,
) -> tuple[list[str], dict]:
    if quantity.components is None:
        reason = quantity.reason or "el backend no produjo componentes"
        return [label + rf"\;\text{{no disponible: {_latex_text(reason)}}}"], {
            "status": quantity.status.value,
            "reason": reason,
            "components": [],
        }

    evaluation = quantity.components
    records = tuple(
        (position, builder.expression(expression, assumptions=assumptions))
        for position, expression in evaluation.values
    )
    independent = False
    if curvature:
        projection = CompactProjection(
            "completed",
            "Componentes reutilizadas desde la proyección del motor.",
            evaluation.free_indices,
            evaluation.dimension,
            records,
        )
        records, independent = independent_curvature_components(projection)

    if not records:
        return [label + "=0"], {
            "status": quantity.status.value,
            "independent_components": independent,
            "component_count": 0,
            "components": [],
        }

    shown = records[:max_components]
    lines = [
        _component_label(label, evaluation.free_indices, position)
        + "="
        + display_expr_to_latex(record.presentation)
        for position, record in shown
    ]
    if len(records) > len(shown):
        qualifier = " independientes" if independent else " no nulas"
        lines.append(
            rf"\text{{se muestran {len(shown)} de {len(records)} componentes{qualifier}}}"
        )
    return lines, {
        "status": quantity.status.value,
        "independent_components": independent,
        "component_count": len(records),
        "components": [
            {
                "position": list(position),
                "expression": record.canonical.to_data(),
            }
            for position, record in records
        ],
    }


_AUXILIARY_CONSTANT = re.compile(
    r"^(?:C_[A-Za-z][A-Za-z0-9]*|poly[A-Za-z][A-Za-z0-9]*D[0-9]+C[0-9]+|"
    r"power[A-Za-z][A-Za-z0-9]*E[A-Za-z0-9]+[AB]|integrationConstantX*[0-9]+)$"
)


def _solution_aliases(solution) -> dict[str, str]:
    names = {
        node.name
        for left, right in solution.rules
        for expression in (left, right)
        for node in walk(expression)
        if isinstance(node, Scalar) and _AUXILIARY_CONSTANT.fullmatch(node.name)
    }
    return {name: f"C_{position}" for position, name in enumerate(sorted(names), 1)}


def _display_solution_expression(builder, expression, aliases) -> str:
    record = builder.expression(expression)
    displayed = record.presentation
    if aliases:
        converted = ir_scalar_to_sympy(displayed)
        substitutions = {
            sp.Symbol(source): sp.Symbol(
                "integrationConstant" + target.removeprefix("C_")
            )
            for source, target in aliases.items()
        }
        displayed = sympy_scalar_to_ir(converted.xreplace(substitutions))
    tex = display_expr_to_latex(displayed)
    return re.sub(
        r"\\mathrm\{integrationConstantX*\}_\{([0-9]+)\}",
        r"C_{\1}",
        tex,
    )


def _solution_score(solution) -> tuple[int, int, str]:
    parameter_assignments = sum(
        not isinstance(left, Function) for left, _ in solution.rules
    )
    method_rank = 0 if "DSolve" in solution.origin else 1
    return parameter_assignments, method_rank, solution.origin


def _solution_lines(solution, builder: PresentationBuilder) -> tuple[list[str], dict]:
    if solution is None:
        return [r"\text{solver no ejecutado para este modelo}"], {"status": "not_run"}
    verified = [
        candidate
        for candidate in solution.solutions
        if candidate.status == "verified_on_domain"
    ]
    if not verified:
        return [rf"\text{{sin familia verificada; estado: {_latex_text(solution.status)}}}"], {
            "status": solution.status,
            "verified_families": 0,
        }
    selected = min(verified, key=_solution_score)
    aliases = _solution_aliases(selected)
    lines = [
        _display_solution_expression(builder, left, aliases)
        + "="
        + _display_solution_expression(builder, right, aliases)
        for left, right in selected.rules
    ]
    if not lines:
        lines = [r"\text{familia verificada sin reglas adicionales}"]
    return lines, {
        "status": solution.status,
        "verified_families": len(verified),
        "selected_origin": selected.origin,
        "selected_rules": [
            [left.to_data(), right.to_data()] for left, right in selected.rules
        ],
        "display_aliases": aliases,
    }


def _aligned(lines: list[str]) -> str:
    return r"\begin{aligned}" + r"\\[2pt]".join(lines) + r"\end{aligned}"


def make_summary(
    runs: Mapping[str, object] | Iterable[object],
    solutions: Mapping[str, object] | Iterable[object] | None = None,
    *,
    output_root: str | Path,
    display_policy: DisplayPolicy | None = None,
    max_components: int = 3,
    compile_pdf: bool = True,
) -> SummaryBundle:
    """Create one concise report without recomputing tensorial or solver results."""

    if (
        isinstance(max_components, bool)
        or not isinstance(max_components, int)
        or max_components < 1
    ):
        raise ValueError("max_components debe ser un entero positivo.")
    completed_runs = _normalise_runs(runs)
    solutions_by_name, solutions_by_run_id = _normalise_solutions(solutions)
    policy = display_policy or DisplayPolicy()
    abstract_lines: list[str] = []
    projected_lines: list[str] = []
    model_records = []

    for run in completed_runs:
        package = run.package
        if package.abstract is None or package.projected is None:
            raise ValueError(
                f"La corrida {package.run_id} no contiene las vistas abstracta y proyectada."
            )
        model_name = package.model.name
        builder = PresentationBuilder(package.model, policy)
        abstract_p4 = _display(builder, package.abstract.curvature_momentum)
        abstract_p2 = _display(builder, package.abstract.metric_momentum)
        abstract_lines.extend(
            (
                rf"\Needspace{{8\baselineskip}}\subsection*{{{_latex_text(model_name)}}}",
                r"\begin{dmath*}P^{abcd}=" + abstract_p4 + r"\end{dmath*}",
                r"\begin{dmath*}P_{ab}\equiv M_{ab}=" + abstract_p2 + r"\end{dmath*}",
            )
        )

        assumptions = (
            ()
            if package.projected.ansatz_geometry is None
            else package.projected.ansatz_geometry.assumptions
        )
        p4_lines, p4_data = _projection_lines(
            package.projected.curvature_momentum,
            r"P^{abcd}",
            builder,
            assumptions=assumptions,
            max_components=max_components,
            curvature=True,
        )
        p2_lines, p2_data = _projection_lines(
            package.projected.metric_momentum,
            r"P_{ab}",
            builder,
            assumptions=assumptions,
            max_components=max_components,
            curvature=False,
        )
        solution = solutions_by_name.get(model_name) or solutions_by_run_id.get(package.run_id)
        solution_lines, solution_data = _solution_lines(solution, builder)
        projected_lines.extend(
            (
                rf"\Needspace{{12\baselineskip}}\subsection*{{{_latex_text(model_name)}}}",
                rf"\textbf{{Ansatz:}} \texttt{{{_latex_text(package.projected.ansatz_name or 'sin nombre')}}}.\par",
                r"\[" + _aligned(p4_lines) + r"\]",
                r"\[" + _aligned(p2_lines) + r"\]",
                r"\noindent\textbf{Solución de las ecuaciones de campo:}\par",
                r"\[" + _aligned(solution_lines) + r"\]",
            )
        )
        model_records.append(
            {
                "model": model_name,
                "run_id": package.run_id,
                "ansatz": package.projected.ansatz_name,
                "abstract": {
                    "curvature_momentum": package.abstract.curvature_momentum.to_data(),
                    "metric_momentum": package.abstract.metric_momentum.to_data(),
                    "metric_momentum_label": "P_ab ≡ M_ab",
                },
                "projected": {
                    "curvature_momentum": p4_data,
                    "metric_momentum": p2_data,
                },
                "field_equations": solution_data,
            }
        )

    payload = {
        "schema_version": "1.0",
        "kind": "research_workflow_summary",
        "model_count": len(model_records),
        "max_components": max_components,
        "models": model_records,
    }
    canonical_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    output_directory = Path(output_root) / f"model-summary-{digest[:12]}"
    output_directory.mkdir(parents=True, exist_ok=True)

    tex = "\n".join(
        (
            r"\documentclass[10pt]{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{amsmath,amssymb,breqn,needspace}",
            r"\usepackage[margin=1.8cm]{geometry}",
            r"\setlength{\parindent}{0pt}",
            r"\begin{document}",
            r"\begin{center}{\Large Resumen de modelos}\end{center}",
            r"\small",
            r"\section*{Sin ansatz}",
            r"Para cada modelo se muestran $P^{abcd}$ y $P_{ab}\equiv M_{ab}$ en forma covariante.",
            *abstract_lines,
            r"\section*{Con ansatz}",
            r"Se muestran componentes ya proyectadas y una familia representativa verificada por el solver.",
            *projected_lines,
            r"\end{document}",
        )
    )
    RunExporter._write_atomic(output_directory / "report.tex", tex)
    RunExporter._write_atomic(output_directory / "summary.json", canonical_json)
    pdf_diagnostic = None
    pdf_path = None
    if compile_pdf:
        _, pdf_diagnostic = RunExporter(output_directory)._compile_pdf(output_directory)
        candidate = output_directory / "report.pdf"
        if candidate.is_file():
            pdf_path = candidate
    manifest = {
        "schema_version": "1.0",
        "kind": "research_workflow_summary",
        "summary_sha256": digest,
        "model_count": len(model_records),
        "pdf_diagnostic": pdf_diagnostic,
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output_directory.iterdir()
            if path.name in {"report.tex", "report.pdf", "summary.json"}
        },
    }
    manifest_path = output_directory / "manifest.json"
    RunExporter._write_atomic(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
    )
    return SummaryBundle(
        output_directory=output_directory,
        tex_path=output_directory / "report.tex",
        json_path=output_directory / "summary.json",
        manifest_path=manifest_path,
        pdf_path=pdf_path,
        model_count=len(model_records),
        pdf_diagnostic=pdf_diagnostic,
    )
