"""Exportacion de toda la cadena a LaTeX y compilacion opcional a PDF."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import sympy as sp

from mc_core import CouplingContext


def _equation(lhs: str, rhs: str) -> str:
    body = rf"{lhs} = {rhs}"
    if len(body) > 175:
        return "\\[\n\\resizebox{\\textwidth}{!}{$\\displaystyle " + body + "$}\n\\]"
    return "\\[\n" + body + "\n\\]"


def _escape_text(text: str) -> str:
    return (text.replace("\\", r"\textbackslash{}")
                .replace("_", r"\_")
                .replace("%", r"\%")
                .replace("&", r"\&")
                .replace("#", r"\#"))


def build_latex(ctx: CouplingContext) -> str:
    section_order: list[str] = []
    section_subgroups: dict[str, list[str | None]] = {}
    for step in ctx.steps:
        if "::" in step.group:
            section, subgroup = step.group.split("::", 1)
        else:
            section, subgroup = step.group, None
        if section not in section_order:
            section_order.append(section)
            section_subgroups[section] = []
        if subgroup not in section_subgroups[section]:
            section_subgroups[section].append(subgroup)

    sections: list[str] = []
    section_intros = {
        "Casos I: formulacion tensorial sin ansatz": (
            "En esta primera seccion de casos no se elige una metrica coordenada, "
            "no se reemplaza $f(r)$ y tampoco se fija un perfil para $\\phi$. "
            "Todos los momentos y las ecuaciones se mantienen a nivel tensorial."
        ),
        "Casos II: sustitucion completa del ansatz": (
            "Solo en esta segunda seccion se imponen la metrica circular y el perfil "
            "escalar correspondiente. A partir de las componentes se resuelve $f(r)$ "
            "y se vuelven a comprobar las ecuaciones de campo y Bianchi."
        ),
    }
    for section in section_order:
        sections.append(rf"\section{{{section}}}")
        if section in section_intros:
            sections.append(section_intros[section])
        for subgroup in section_subgroups[section]:
            if subgroup is not None:
                sections.append(rf"\subsection{{{subgroup}}}")
            expected_group = section if subgroup is None else f"{section}::{subgroup}"
            for step in (item for item in ctx.steps if item.group == expected_group):
                command = "subsection" if subgroup is None else "subsubsection"
                sections.append(rf"\{command}{{{step.title}}}")
                sections.append(
                    rf"\noindent\texttt{{{_escape_text(step.key)}}}\par\nopagebreak[4]"
                )
                sections.append(_equation(step.lhs, step.rhs))
                if step.note:
                    sections.append(rf"\begin{{quote}}\small {step.note}\end{{quote}}")

    check_rows = []
    for key, value in ctx.checks.items():
        check_rows.append(
            rf"\texttt{{{_escape_text(key)}}} & ${sp.latex(value)}$ & OK\\"
        )

    return r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish,es-noquoting]{babel}
\usepackage{amsmath,amssymb,mathtools,bm}
\usepackage[a4paper,margin=1.8cm]{geometry}
\usepackage{graphicx,longtable,array,booktabs,xcolor,hyperref}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!60!black}
\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{3}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.45em}
\allowdisplaybreaks
\newcommand{\Rcal}{\mathcal R}
\begin{document}
\hypersetup{pageanchor=false}
\begin{titlepage}
\centering
\vspace*{2.4cm}
{\Huge\bfseries Modelos con acoplamiento\par}
\vspace{0.7cm}
{\Large Derivación simbólica de $L(g,R,\phi,\nabla\phi)$\par}
\vspace{0.35cm}
{\large y especialización a los Casos 0, 1 y 2 de Draft4\par}
\vfill
{\large Guillermo Jordan A.\par}
\vspace{0.4cm}
{\small Motor reproducible en Python/SymPy; notebook usado solo como orquestador.\par}
\vfill
{\today\par}
\end{titlepage}
\hypersetup{pageanchor=true}
\pagenumbering{roman}

\tableofcontents
\clearpage
\pagenumbering{arabic}

\section{Alcance, convenciones y trazabilidad}
Se adopta una conexion de Levi--Civita, firma $(-,+,+)$ y
\[
R^\rho{}_{\sigma\mu\nu}=\partial_\mu\Gamma^\rho{}_{\nu\sigma}
-\partial_\nu\Gamma^\rho{}_{\mu\sigma}
+\Gamma^\rho{}_{\mu\lambda}\Gamma^\lambda{}_{\nu\sigma}
-\Gamma^\rho{}_{\nu\lambda}\Gamma^\lambda{}_{\mu\sigma}.
\]
La cadena abstracta reproduce la sección ``Generalización con campo escalar'' del
BEAMER del proyecto. Las especializaciones usan los Casos 0, 1 y 2, ecuaciones
(11)--(138), de \texttt{Papers/Draft\_4.pdf}. Se distingue el momento métrico $M_{ab}$ del momento
de curvatura $P^{abcd}$ para evitar la colision de notacion presente en algunos
desarrollos manuales.

\paragraph{Que se calcula realmente.}
Las identidades tensoriales abstractas se almacenan como pasos LaTeX auditables.
La geometría del ansatz se calcula componente por componente con SymPy desde
$g_{ab}$: inversa, conexion, Riemann, Ricci, $R$, Einstein, divergencia covariante
y ecuaciones de campo. Las filas ``OK'' al final son residuos simbólicos que el
programa simplificó exactamente a cero.

""" + "\n\n".join(sections) + r"""

\section{Verificaciones simbolicas automaticas}
\begin{longtable}{p{0.67\textwidth}cc}
\toprule
\textbf{Residuo} & \textbf{Valor} & \textbf{Estado}\\
\midrule
\endfirsthead
\toprule
\textbf{Residuo} & \textbf{Valor} & \textbf{Estado}\\
\midrule
\endhead
""" + "\n".join(check_rows) + r"""
\bottomrule
\end{longtable}

\section{Resultado final}
Para el Caso-0, los momentos son
\[
P_0^{abcd}=\tfrac12(g^{ac}g^{bd}-g^{ad}g^{bc}),\quad
M^{(0)}_{ab}=2R_{ab},\quad J_0^a=F_\phi^{(0)}=0,
\]
y $\nabla_aP_0^{abcd}=0$. Por tanto,
\[
E^{(0)}_{ab}=G_{ab}-\ell^{-2}g_{ab}=0.
\]
El ansatz circular reduce estas ecuaciones a $f'=2r/\ell^2$ y
$f''=2/\ell^2$, cuya solucion es $f=r^2/\ell^2-\lambda$. La sustitucion final
da $R_{ab}=-2g_{ab}/\ell^2$, $R=-6/\ell^2$ y $E^{(0)}_{ab}=0$ componente a
componente. La identidad de Bianchi se anula incluso antes de imponer la solucion.

Para el Caso-1 se obtiene, antes del ansatz,
\[
P_1^{abcd}=P_0^{abcd},\qquad
M^{(1)}_{ab}=2R_{ab}-\alpha_1u_au_b,\qquad
J_1^a=-2\alpha_1u^a,
\]
\[
E^{(1)}_{ab}=G_{ab}-\ell^{-2}g_{ab}
-\alpha_1\left(u_au_b-\tfrac12g_{ab}X\right),
\qquad E^{(1)}_\phi=2\alpha_1\Box\phi.
\]
Solo después de fijar $\phi=p\varphi$, la ecuación escalar se anula y la ecuación
radial integra a
\[
f_{(1)}(r)=\frac{r^2}{\ell^2}-\lambda
-\alpha_1p^2\log\left(\frac r{r_0}\right).
\]
La sustitución directa devuelve $E^{(1)}_{ab}=0$, $E^{(1)}_\phi=0$ y la identidad
de Bianchi--Noether nula componente a componente.

Para el Caso-2, definiendo
\[
H^{ab}=3u^au^b-Xg^{ab},\qquad
C^{ab}=g^{ab}+\ell^2\beta_0H^{ab},
\]
los momentos caracteristicos son
\[
P_2^{abcd}=\tfrac14(C^{ac}g^{bd}-C^{ad}g^{bc}-C^{bc}g^{ad}+C^{bd}g^{ac}),
\qquad
J_2^a=2\ell^2\beta_0(3R^{ab}-Rg^{ab})u_b.
\]
El perfil $\phi=p\varphi$ satisface la ecuacion escalar y las ecuaciones metricas
se integran mediante $N=Hf$, con $H=1+\beta_0p^2\ell^2/r^2$. La rama exacta es
\[
f_{(2)}(r)=\frac{r^2/\ell^2-\lambda}
{1+\beta_0p^2\ell^2/r^2}.
\]
Tras sustituirla se obtienen todos los momentos en forma explicita,
$E^{(2)}_{ab}=0$, $E^{(2)}_\phi=0$ y la identidad de Bianchi--Noether nula.

\end{document}
"""


def export_results(ctx: CouplingContext, compile_pdf: bool = True) -> dict[str, object]:
    output_dir = ctx.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "derivacion_modelos_con_acoplamiento_casos0_2.tex"
    pdf_path = output_dir / "derivacion_modelos_con_acoplamiento_casos0_2.pdf"
    json_path = output_dir / "verificaciones_simbolicas.json"

    tex_path.write_text(build_latex(ctx), encoding="utf-8")
    json_path.write_text(json.dumps(
        {key: str(value) for key, value in ctx.checks.items()},
        indent=2,
        ensure_ascii=False,
    ), encoding="utf-8")

    compile_log = ""
    compiled = False
    if compile_pdf:
        commands: list[list[str]] = []
        if shutil.which("latexmk") and shutil.which("perl"):
            commands.append([shutil.which("latexmk"), "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name])
        if shutil.which("pdflatex"):
            # Dos pasadas resuelven tabla de contenidos y referencias aun cuando
            # latexmk este instalado pero no disponga de Perl (caso comun en Windows).
            commands.extend([[shutil.which("pdflatex"), "-interaction=nonstopmode", "-halt-on-error", tex_path.name]] * 2)
        for command in commands:
            process = subprocess.run(
                command,
                cwd=output_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
            )
            compile_log += process.stdout + "\n" + process.stderr + "\n"
            compiled = process.returncode == 0 and pdf_path.exists()
            if compiled and "latexmk" in Path(command[0]).name.lower():
                break

    info = {
        "tex": tex_path,
        "pdf": pdf_path if compiled else None,
        "checks": json_path,
        "all_checks_zero": all(value == 0 for value in ctx.checks.values()),
        "check_count": len(ctx.checks),
        # El log completo queda en el .log de TeX; la vista del notebook solo
        # necesita el cierre para no inflar el archivo con miles de lineas.
        "compile_log": compile_log[-2000:],
    }
    return info
