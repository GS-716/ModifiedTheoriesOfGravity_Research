from pathlib import Path
import re
import shutil
import subprocess
import sympy as sp
from IPython.display import display, Markdown, FileLink


def latex_legible(expr):
    tex = sp.latex(expr)
    tex = tex.replace(r"\mathrm{H}", r"\delta \mathrm{g}")
    tex = tex.replace("sqrt_{minus g}", r"\sqrt{-g}")
    tex = tex.replace("sqrt_{minus_g}", r"\sqrt{-g}")
    tex = tex.replace(r"\nabla\mathrm", r"\nabla \mathrm")
    tex = tex.replace(r"\delta\mathrm", r"\delta \mathrm")

    tokens = []
    for match in re.finditer(r"0_\{(\d+)\}", tex):
        token = match.group(0)
        if token not in tokens:
            tokens.append(token)

    names = ["i","j","k","l","m","n","r","s","t","u","v","w"]
    for pos, token in enumerate(tokens):
        new = names[pos] if pos < len(names) else rf"q_{{{pos}}}"
        tex = tex.replace(token, new)
    return tex


def block(lhs, rhs):
    eq = rf"{lhs} = {rhs}"
    if len(eq) > 240:
        return "\\[\n\\resizebox{\\textwidth}{!}{$\\displaystyle " + eq + "$}\n\\]\n"
    return "\\[\n\\boxed{" + eq + "}\n\\]\n"


def export_results(ctx, output_dir=None, show_preview=True, compile_pdf=True):
    S = ctx.S
    if output_dir is None:
        output_dir = Path.cwd()/"salidas_LL_completo"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    construction_items = [(r"m_{\max}", "m_input")]
    for k in range(ctx.max_order + 1):
        construction_items.append((rf"L_{k}", f"L_{k}_constructed"))
    construction_items.append((r"L=\sum_{k=0}^{m_{\max}}c_kL_k", "L_total_constructed"))

    groups = [
        ("Construcción de todos los términos y del Lagrangiano final", construction_items),
        ("Tensor P total y divergencia", [
            (r"P^{abcd}\ \text{(antes de canonizar)}", "P_total_product_rule_raw"),
            (r"P^{abcd}\ \text{(canonizado)}", "P_total_compact"),
            (r"\nabla_aP^{abcd}", "divergence_P_total"),
        ]),
        ("Identidad métrica y Rcal", [
            (r"P^{ab}", "P_metric_total_compact"),
            (r"\mathcal R^{ab}", "Rcal_total_compact"),
        ]),
        ("Derivación variacional completa", [
            (r"(\mathcal L_\xi L)_1", "Lie_L_route_1"),
            (r"(\mathcal L_\xi L)_2", "Lie_L_route_2"),
            (r"\delta L", "delta_L_total_unsplit"),
            (r"P\delta R\ \text{tras Palatini}", "palatini_sum"),
            (r"P\delta R\ \text{tras }\delta\Gamma", "palatini_metric_second_derivative"),
            (r"\delta v^j", "delta_v_vector"),
        ]),
        ("Resultado final", [
            (r"-2\nabla^m\nabla^nP_{amnb}", "minus2_double_divergence_P_ab"),
            (r"E_{ab}", "E_ab_raw"),
            (r"E^i{}_j\ \text{sin abreviaturas }L_k", "E_mixed_explicit_total"),
            (r"\sqrt{-g}E_{ab}\delta g^{ab}", "delta_action_bulk_integrand"),
        ]),
    ]

    missing = [key for _, items in groups for _, key in items if key not in S]
    if missing:
        raise KeyError(f"Faltan objetos para exportar: {missing}")

    if show_preview:
        display(Markdown("## Vista previa compacta"))
        for title, items in groups:
            display(Markdown(f"### {title}"))
            for lhs, key in items:
                display(Markdown(
                    f"**`S[{key!r}]`**\n\n\\[\n{lhs}={latex_legible(S[key])}\n\\]"
                ))

    sections = []
    for title, items in groups:
        sections.append(rf"\section*{{{title}}}")
        for lhs, key in items:
            key_tex = key.replace("_", r"\_")
            sections.append(rf"\noindent\texttt{{S[{key_tex}]}}")
            sections.append(block(lhs, latex_legible(S[key])))

    check_keys = [key for key in S if key.startswith("check_")]
    results = [(key, ctx.tsimplify(S[key])) for key in check_keys]
    all_ok = all(value == 0 for _, value in results)

    table = [
        r"\section*{Verificaciones automáticas}",
        rf"Total: {len(results)}. " + (r"\textbf{Todas dieron cero.}" if all_ok else r"\textbf{Hay verificaciones no nulas.}"),
        r"\begin{longtable}{p{0.78\textwidth}c}",
        r"\toprule",
        r"\textbf{Verificación} & \textbf{Resultado}\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Verificación} & \textbf{Resultado}\\",
        r"\midrule",
        r"\endhead",
    ]
    for key, value in results:
        key_tex = key.replace("_", r"\_")
        table.append(rf"\texttt{{{key_tex}}} & ${latex_legible(value)}$\\")
    table += [r"\bottomrule", r"\end{longtable}"]

    doc = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,mathtools}
\usepackage[a4paper,margin=1.7cm]{geometry}
\usepackage{graphicx,longtable,array,booktabs}
\setlength{\parindent}{0pt}
\allowdisplaybreaks
\begin{document}
\begin{center}
{\Large\bfseries Derivación simbólica del Lagrangiano completo de Lanczos--Lovelock}\\[0.5em]
{\small Construcción de todos los términos desde $L_0$ hasta $L_{m_{\max}}$}
\end{center}
""" + "\n\n".join(sections) + "\n\n" + "\n".join(table) + r"\end{document}"

    tex_path = output_dir/"resumen_derivacion_LL_completo.tex"
    snippet_path = output_dir/"ecuaciones_principales_LL_completo.tex"
    pdf_path = output_dir/"resumen_derivacion_LL_completo.pdf"

    tex_path.write_text(doc, encoding="utf-8")
    snippet_path.write_text("\n\n".join(sections), encoding="utf-8")

    compiled = False
    compile_message = ""
    if compile_pdf and shutil.which("pdflatex"):
        proc = subprocess.run(
            [shutil.which("pdflatex"), "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=output_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        compiled = proc.returncode == 0 and pdf_path.exists()
        if not compiled:
            compile_message = proc.stdout[-4000:]
    elif compile_pdf:
        compile_message = "No se encontró pdflatex."

    display(Markdown("## Archivos generados"))
    display(FileLink(str(tex_path), result_html_prefix="Documento LaTeX: "))
    display(FileLink(str(snippet_path), result_html_prefix="Fragmento LaTeX: "))
    if compiled:
        display(FileLink(str(pdf_path), result_html_prefix="PDF: "))
    elif compile_pdf:
        display(Markdown("**El PDF no se compiló automáticamente.**"))
        if compile_message:
            print(compile_message)

    print(f"Verificaciones: {len(results)}")
    print(f"Todas dieron cero: {all_ok}")
    print(f"Carpeta: {output_dir}")

    return {
        "tex": tex_path,
        "snippet": snippet_path,
        "pdf": pdf_path if compiled else None,
        "all_checks_zero": all_ok,
        "check_count": len(results),
    }
