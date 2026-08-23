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
    tex = tex.replace(r"\left(\frac{1}{2}\right)", r"\frac{1}{2}")
    tex = tex.replace(r"\left(- \frac{1}{2}\right)", r"-\frac{1}{2}")
    tex = tex.replace(r"\nabla\mathrm", r"\nabla \mathrm")
    tex = tex.replace(r"\delta\mathrm", r"\delta \mathrm")
    tex = re.sub(r"(?<![A-Za-z])GB(?![A-Za-z])", r"\\mathcal{G}", tex)

    tokens = []
    for match in re.finditer(r"0_\{(\d+)\}", tex):
        token = match.group(0)
        if token not in tokens:
            tokens.append(token)

    nombres = ["i", "j", "k", "l", "m", "n", "r", "s", "t", "u", "v", "w"]
    for pos, token in enumerate(tokens):
        nuevo = nombres[pos] if pos < len(nombres) else rf"q_{{{pos}}}"
        tex = tex.replace(token, nuevo)

    return tex


def bloque_ecuacion(lhs, rhs):
    ecuacion = rf"{lhs} = {rhs}"
    if len(ecuacion) > 220:
        return (
            "\\[\n"
            "\\resizebox{\\textwidth}{!}{$\\displaystyle "
            + ecuacion
            + "$}\n"
            "\\]\n"
        )
    return "\\[\n\\boxed{" + ecuacion + "}\n\\]\n"


RESUMEN_PRINCIPAL = [
    (
        "Entrada y derivadas escalares",
        [
            (r"L", "L_input"),
            (r"L_R", "L_R"),
            (r"L_{\mathcal G}", "L_GB"),
            (r"L_{RR}", "L_RR"),
            (r"L_{R\mathcal G}", "L_RGB"),
            (r"L_{\mathcal G\mathcal G}", "L_GBGB"),
        ],
    ),
    (
        "Construcción de las estructuras de curvatura",
        [
            (r"\dfrac{\partial R}{\partial R_{abcd}}", "dR_dRiemann"),
            (r"\dfrac{\partial (R_{ijkl}R^{ijkl})}{\partial R_{abcd}}", "P_Riemann2_abcd"),
            (r"\dfrac{\partial (R_{ij}R^{ij})}{\partial R_{abcd}}", "P_Ricci2_abcd"),
            (r"\dfrac{\partial R^2}{\partial R_{abcd}}", "P_R2_abcd"),
            (r"\dfrac{\partial \mathcal G}{\partial R_{abcd}}", "dGB_dRiemann"),
            (r"P^{abcd}", "P_abcd"),
        ],
    ),
    (
        "Derivadas métricas y Rcal",
        [
            (r"\dfrac{\partial R}{\partial g_{ab}}", "dR_dg_cov"),
            (r"\dfrac{\partial \mathcal G}{\partial g_{ab}}", "dGB_dg_cov"),
            (r"P^{ab}", "P_metric_ab"),
            (r"\mathcal{R}_{ab}", "Rcal_down_ab"),
        ],
    ),
    (
        "Derivada de Lie por las dos rutas",
        [
            (r"\nabla_m R", "nabla_R_from_Riemann"),
            (r"\nabla_m \mathcal G", "nabla_GB_from_Riemann"),
            (r"\left(\mathcal{L}_{\xi}L\right)_{\mathrm{ruta\ 1}}", "Lie_L_route_1"),
            (r"\left(\mathcal{L}_{\xi}L\right)_{\mathrm{ruta\ 2}}", "Lie_L_route_2"),
        ],
    ),
    (
        "Variación antes de integrar por partes",
        [
            (r"\delta L", "delta_L_total_unsplit"),
            (r"\delta\!\left(\sqrt{-g}L\right)", "delta_density_unsplit"),
            (r"\left(P\,\delta R\right)_{\mathrm{pieza\ métrica}}", "delta_R_split_metric_piece"),
            (r"\left(P\,\delta R\right)_{\mathrm{pieza\ conexión}}", "delta_R_split_connection_piece"),
        ],
    ),
    (
        "Palatini e integraciones por partes",
        [
            (r"\left(P\,\delta R\right)_{\mathrm{Palatini}}", "palatini_sum"),
            (r"\left(P\,\delta R\right)_{\delta\Gamma\ \mathrm{sustituido}}", "palatini_metric_second_derivative"),
            (r"I_{\mathrm{inicio}}", "ibp_start"),
            (r"\nabla_j B_1^{\,j}", "ibp1_divergence"),
            (r"\nabla_j B_2^{\,j}", "ibp2_divergence"),
            (r"\delta v^{\,j}", "delta_v_vector"),
        ],
    ),
    (
        "Resultado final",
        [
            (r"-2\nabla^m\nabla^n P_{amnb}", "minus2_double_divergence_P_ab"),
            (r"E_{ab}", "E_ab_raw"),
            (r"\sqrt{-g}\,E_{ab}\,\delta g^{ab}", "delta_action_bulk_integrand"),
            (r"\nabla_a P^{abcd}", "divergence_P_bcd"),
        ],
    ),
]


def export_results(
    ctx,
    mostrar_vista_previa=True,
    compilar_pdf=True,
    carpeta_salida=None,
):
    S = ctx.S
    if carpeta_salida is None:
        carpeta_salida = Path.cwd() / "salidas_FR_GB"
    else:
        carpeta_salida = Path(carpeta_salida)

    faltantes = [
        key
        for _, items in RESUMEN_PRINCIPAL
        for _, key in items
        if key not in S
    ]
    if faltantes:
        raise KeyError(f"Faltan objetos en S: {faltantes}")

    if mostrar_vista_previa:
        display(Markdown("## Vista previa compacta"))
        for titulo, items in RESUMEN_PRINCIPAL:
            display(Markdown(f"### {titulo}"))
            for lhs, key in items:
                rhs = latex_legible(S[key])
                display(Markdown(
                    f"**`S[{key!r}]`**\n\n"
                    f"\\[\n{lhs} = {rhs}\n\\]"
                ))

    ecuaciones = []
    for titulo, items in RESUMEN_PRINCIPAL:
        ecuaciones.append(rf"\section*{{{titulo}}}")
        for lhs, key in items:
            rhs = latex_legible(S[key])
            key_tex = key.replace("_", r"\_")
            ecuaciones.append(rf"\noindent\texttt{{S[{key_tex}]}}")
            ecuaciones.append(bloque_ecuacion(lhs, rhs))

    ecuaciones_tex = "\n\n".join(ecuaciones)

    check_keys = [key for key in S if key.startswith("check_")]
    check_results = [(key, ctx.tsimplify(S[key])) for key in check_keys]
    checks_ok = all(result == 0 for _, result in check_results)

    tabla_checks = [
        r"\section*{Verificaciones automáticas}",
        rf"\noindent Total de identidades comprobadas: {len(check_results)}. "
        + (
            r"\textbf{Todas dieron cero.}"
            if checks_ok
            else r"\textbf{Hay verificaciones que no dieron cero.}"
        ),
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

    for key, result in check_results:
        key_tex = key.replace("_", r"\_")
        tabla_checks.append(
            rf"\texttt{{{key_tex}}} & ${latex_legible(result)}$\\"
        )

    tabla_checks += [r"\bottomrule", r"\end{longtable}"]
    checks_tex = "\n".join(tabla_checks)

    documento = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,mathtools}
\usepackage[a4paper,margin=1.8cm]{geometry}
\usepackage{graphicx}
\usepackage{longtable,array,booktabs}
\setlength{\parindent}{0pt}
\allowdisplaybreaks

\begin{document}

\begin{center}
{\Large\bfseries Resumen de la derivación simbólica para teorías $L(R,\mathcal G)$}\\[0.5em]
{\small Generado directamente a partir de los objetos calculados en el notebook}
\end{center}

\section*{Lagrangiano usado}
""" + bloque_ecuacion(
        r"L_{\mathrm{input}}",
        latex_legible(S["L_input"])
    ) + "\n\n" + ecuaciones_tex + "\n\n" + checks_tex + r"""

\end{document}
"""

    carpeta_salida.mkdir(parents=True, exist_ok=True)

    snippet_path = carpeta_salida / "ecuaciones_principales_FR_GB.tex"
    tex_path = carpeta_salida / "resumen_derivacion_FR_GB.tex"
    pdf_path = carpeta_salida / "resumen_derivacion_FR_GB.pdf"

    snippet_path.write_text(ecuaciones_tex, encoding="utf-8")
    tex_path.write_text(documento, encoding="utf-8")

    compilado_pdf = False
    mensaje_compilacion = ""

    if compilar_pdf and shutil.which("pdflatex"):
        proceso = subprocess.run(
            [
                shutil.which("pdflatex"),
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=carpeta_salida,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        compilado_pdf = proceso.returncode == 0 and pdf_path.exists()
        if not compilado_pdf:
            mensaje_compilacion = proceso.stdout[-3000:]
    elif compilar_pdf:
        mensaje_compilacion = (
            "No se encontró pdflatex. Los archivos .tex sí fueron generados."
        )

    display(Markdown("## Archivos generados"))
    display(FileLink(str(tex_path), result_html_prefix="Documento LaTeX completo: "))
    display(FileLink(str(snippet_path), result_html_prefix="Fragmento LaTeX reutilizable: "))

    if compilado_pdf:
        display(FileLink(str(pdf_path), result_html_prefix="PDF compilado: "))
    elif compilar_pdf:
        display(Markdown(
            "**El PDF no se compiló automáticamente.** "
            "El `.tex` quedó guardado y puede compilarse en Overleaf o con `pdflatex`."
        ))
        if mensaje_compilacion:
            print(mensaje_compilacion)

    print(f"Verificaciones automáticas: {len(check_results)}")
    print(f"Todas dieron cero: {checks_ok}")
    print(f"Carpeta de salida: {carpeta_salida}")

    return {
        "snippet_path": snippet_path,
        "tex_path": tex_path,
        "pdf_path": pdf_path if compilado_pdf else None,
        "checks_ok": checks_ok,
        "check_results": check_results,
    }
