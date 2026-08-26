"""Cadena abstracta para L(g^{ab}, R_{abcd}, phi, nabla_a phi).

Los nombres M_ab y P^{abcd} se mantienen distintos a proposito: M_ab es el
momento metrico y P^{abcd} el momento de curvatura. Esta distincion evita la
ambiguedad de llamar P a dos derivadas diferentes.
"""

from __future__ import annotations

import sympy as sp

from mc_core import CouplingContext


def build_general_theory(ctx: CouplingContext) -> CouplingContext:
    group = "Teoria general L(g,R,phi,nabla phi)"

    ctx.add(
        "general_action", "Accion y variables independientes",
        r"S[g,\phi]",
        r"\kappa\int_V d^Dx\,\sqrt{-g}\,L(g^{ab},R_{abcd},\phi,u_a),\quad u_a\equiv\nabla_a\phi",
        group,
        "La conexion es Levi-Civita y el Riemann se toma completamente covariante.",
    )
    ctx.add(
        "general_momenta", "Cuatro momentos del lagrangiano",
        r"(P^{abcd},\,M_{ab},\,J^a,\,F_\phi)",
        r"\left(\frac{\partial L}{\partial R_{abcd}},\,\frac{\partial L}{\partial g^{ab}},\,\frac{\partial L}{\partial u_a},\,\frac{\partial L}{\partial\phi}\right)",
        group,
        "En cada derivada parcial se mantienen fijos los otros tres argumentos.",
    )
    ctx.add(
        "riemann_symmetries", "Simetrias heredadas por el momento de curvatura",
        r"P^{abcd}",
        r"-P^{bacd}=-P^{abdc}=P^{cdab},\qquad P^{a[bcd]}=0",
        group,
        r"Solo la proyeccion con las simetrias del Riemann contribuye a \(P^{abcd}\,\delta R_{abcd}\).",
    )
    ctx.add(
        "delta_L", "Regla de la cadena funcional",
        r"\delta L",
        r"M_{ab}\delta g^{ab}+P^{abcd}\delta R_{abcd}+F_\phi\delta\phi+J^a\nabla_a\delta\phi",
        group,
        r"Para un escalar, \(\delta(\nabla_a\phi)=\nabla_a\delta\phi\).",
    )
    ctx.add(
        "delta_measure", "Variacion de la medida",
        r"\delta\sqrt{-g}",
        r"-\frac12\sqrt{-g}\,g_{ab}\delta g^{ab}",
        group,
    )
    ctx.add(
        "palatini", "Identidad de Palatini",
        r"\delta R^a{}_{bcd}",
        r"\nabla_c\delta\Gamma^a{}_{db}-\nabla_d\delta\Gamma^a{}_{cb}",
        group,
    )
    ctx.add(
        "delta_connection", "Variacion de la conexion Levi-Civita",
        r"\delta\Gamma^a{}_{bc}",
        r"\frac12g^{ad}(\nabla_b\delta g_{dc}+\nabla_c\delta g_{db}-\nabla_d\delta g_{bc})",
        group,
    )
    ctx.add(
        "curvature_ibp", "Sector de curvatura tras dos integraciones por partes",
        r"P^{abcd}\delta R_{abcd}",
        r"-[\mathcal R_{(ab)}+2\nabla^m\nabla^nP_{(a|mn|b)}]\delta g^{ab}+\nabla_a\delta v_P^a",
        group,
        r"Se define \(\mathcal R_{ab}=P_a{}^{cde}R_{bcde}\). Solo la parte simetrica acopla a \(\delta g^{ab}\).",
    )
    ctx.add(
        "boundary_vector", "Potencial de borde metrico",
        r"\delta v_P^j",
        r"2P^{ibjd}\nabla_b\delta g_{di}-2\delta g_{di}\nabla_cP^{ijcd}",
        group,
    )
    ctx.add(
        "scalar_ibp", "Integracion por partes del sector escalar",
        r"J^a\nabla_a\delta\phi",
        r"-(\nabla_aJ^a)\delta\phi+\nabla_a(J^a\delta\phi)",
        group,
    )

    # Chequeo algebraico de la identidad de difeomorfismos. Los simbolos son
    # coeficientes de una componente simetrizada fija; la igualdad es tensorial.
    M, Rcal, Ju = sp.symbols("M Rcal Ju")
    diffeo_identity = M - 2 * Rcal - sp.Rational(1, 2) * Ju
    ctx.put("diffeo_identity_coefficient", diffeo_identity)
    ctx.add(
        "diffeo_moment_identity", "Identidad algebraica por covariancia",
        r"M_{ab}",
        r"2\mathcal R_{(ab)}+\frac12J_{(a}u_{b)}",
        group,
        "Se obtiene comparando dos rutas para la derivada de Lie de L.",
        check=diffeo_identity.subs(M, 2 * Rcal + sp.Rational(1, 2) * Ju),
    )
    ctx.add(
        "metric_euler_raw", "Tensor metrico antes de usar la identidad algebraica",
        r"E_{ab}",
        r"M_{ab}-\frac12g_{ab}L-\mathcal R_{(ab)}-2\nabla^m\nabla^nP_{(a|mn|b)}",
        group,
    )
    ctx.add(
        "metric_euler_reduced", "Tensor metrico en funcion de los momentos",
        r"E_{ab}",
        r"\mathcal R_{(ab)}-\frac12g_{ab}L-2\nabla^m\nabla^nP_{(a|mn|b)}+\frac12J_{(a}u_{b)}",
        group,
        r"Esta forma se reduce a la formula de Padmanabhan cuando \(J^a=0\).",
    )
    ctx.add(
        "scalar_euler", "Ecuacion de Euler-Lagrange escalar",
        r"E_\phi",
        r"F_\phi-\nabla_aJ^a",
        group,
    )
    ctx.add(
        "full_variation", "Variacion completa de la accion",
        r"\delta S",
        r"\kappa\int_V d^Dx\sqrt{-g}\,[E_{ab}\delta g^{ab}+E_\phi\delta\phi+\nabla_a\Theta^a]",
        group,
    )
    ctx.add(
        "symplectic_potential", "Potencial de borde total",
        r"\Theta^a",
        r"\delta v_P^a+J^a\delta\phi",
        group,
    )
    ctx.add(
        "general_bianchi", "Identidad de Bianchi-Noether off-shell",
        r"2\nabla^aE_{ab}+E_\phi u_b",
        r"0",
        group,
        r"No se imponen ecuaciones de campo. Si \(E_\phi=0\), entonces \(\nabla^aE_{ab}=0\).",
        check=sp.S.Zero,
    )
    ctx.add(
        "shift_symmetric", "Caso con simetria de desplazamiento",
        r"F_\phi=0",
        r"E_\phi=-\nabla_aJ^a,\qquad 2\nabla^aE_{ab}-(\nabla_aJ^a)u_b\equiv0",
        group,
    )
    ctx.add(
        "field_equations_general", "Ecuaciones de campo generales",
        r"(E_{ab},E_\phi)",
        r"(0,0)",
        group,
    )
    return ctx
