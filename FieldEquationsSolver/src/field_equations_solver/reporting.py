"""Compact solver-only reports; the tensor run's exports are never overwritten."""
from dataclasses import asdict
import hashlib
import json
import re
from pathlib import Path

from tensor_engine.exporting import RunExporter, display_expr_to_latex
from tensor_engine.model import ModelSpec
from tensor_engine.presentation import DisplayPolicy, PresentationBuilder


def _latex_text(text):
    replacements = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
                    "$": r"\$", "%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_",
                    "^": r"\textasciicircum{}", "~": r"\textasciitilde{}"}
    return "".join(replacements.get(c, c) for c in text)


def solution_latex(result, display_policy=None):
    policy = display_policy or DisplayPolicy()
    model = ModelSpec.from_data(result.source_results["model"])
    builder = PresentationBuilder(model, policy)
    presentation = {}
    def render(key, expr):
        record = builder.expression(expr)
        presentation[key] = record.to_data()
        tex = display_expr_to_latex(record.presentation)
        return re.sub(r"\\mathrm\{integrationConstantX*\}_\{([0-9]+)\}", r"C_{\1}", tex)
    def equation(tex):
        # Break large sums at top-level terms in the IR if needed at call sites;
        # this width limit is the final guard against clipping arbitrary ansatz.
        return r"\[\adjustbox{max width=\linewidth}{$\displaystyle " + tex + r"$}\]"
    lines = [r"\documentclass[10pt]{article}", r"\usepackage[utf8]{inputenc}",
             r"\usepackage[T1]{fontenc}", r"\usepackage{amsmath,amssymb,adjustbox}",
             r"\usepackage[margin=2cm]{geometry}", r"\begin{document}",
             r"\begin{center}\large Reducción y resolución formal\end{center}",
             _latex_text(result.source_run_id + " | " + result.status) + r"\par"]
    if result.ansatz:
        lines.append(_latex_text("Ansatz: " + result.ansatz.name) + r"\par")
        if result.ansatz.scalar_field is not None:
            lines.append(equation(r"\phi=" + render("profile", result.ansatz.scalar_field)))
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
    lines.append(r"\section*{Familias formales y verificación}")
    if not result.solutions:
        lines.append("No se obtuvo una familia formal verificada; se conservan las ecuaciones anteriores.\\par")
    for i, solution in enumerate(result.solutions):
        if solution.status != "verified_on_domain":
            lines.append(_latex_text(f"Candidato {i+1}: no verificado ({solution.status}); residuales completos en results.json.") + r"\par")
            continue
        lines.append(_latex_text(f"Familia {i + 1}: verificada en todas las ecuaciones, en el dominio declarado.") + r"\par")
        for j, (lhs, rhs) in enumerate(solution.rules):
            lines.append(equation(render(f"solution_{i}_{j}_lhs", lhs) + "=" + render(f"solution_{i}_{j}", rhs)))
        assigned = {getattr(lhs, "name", "") for lhs, _ in solution.rules}
        parameters = [name for name in result.classification.get("parameters", ()) if name not in assigned]
        if parameters:
            lines.append(_latex_text("Parámetros libres en esta familia: " + ", ".join(parameters)) + r"\par")
        if solution.nonzero_conditions:
            # Keep long domain formulae in JSON; the compact report states the
            # actual restrictions and the source ansatz without duplicating f.
            lines.append(r"Dominio: métrica no degenerada y denominadores no nulos.\par")
        if solution.unresolved:
            lines.append(_latex_text("Pendiente: " + "; ".join(solution.unresolved)) + r"\par")
        if result.ansatz and result.ansatz.assumptions:
            lines.append(_latex_text("Supuestos: " + "; ".join(result.ansatz.assumptions)) + r"\par")
    classification = result.classification
    if classification:
        lines.append(_latex_text(f"Sistema: {classification['kind']}; orden máximo: {classification['max_derivative_order']}.") + r"\par")
        if classification.get("unconstrained_unknowns"):
            lines.append(_latex_text("Funciones o parámetros libres: " + ", ".join(classification["unconstrained_unknowns"])) + r"\par")
    for reason in result.diagnostics:
        lines.append(_latex_text(reason) + r"\par")
    lines.append("Completitud de las familias y ramas singulares: no certificada.\\par")
    lines.append(r"\end{document}")
    return "\n".join(lines), {"policy": asdict(policy), "expressions": presentation}


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
    manifest = {"schema_version": "1.0", "kind": "field_equation_solution",
                "source_run_id": result.source_run_id, "source_fingerprint": result.source_fingerprint,
                "result_sha256": digest, "files": files, "pdf_diagnostic": diagnostic}
    RunExporter._write_atomic(directory / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return directory
